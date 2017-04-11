import itertools
import os
import inspect

from six import StringIO


def get_inputs_helper(step, ignore, target):
    """
    Recursion helper used by get_inputs()
    """
    outputs = set()
    if not ignore and step.target == target:
        outputs.add(step)

    if ignore or not step.target:
        for i in step.inputs:
            outputs.update(get_inputs_helper(i, ignore=False, target=target))

    return outputs


def get_inputs(step, target):
    """
    Traverse input parents tree returning all steps which are targets or not targets
    (depending on argument target). Stop traversing at parent targets
    """
    return get_inputs_helper(step, ignore=True, target=target)


def get_drake_data(steps):
    """
    Returns: a dictionary of outputs mapped to inputs
    Note that an output is either a target or a leaf node in the
        step tree
    """
    output_inputs = {}
    if len(steps) == 0:
        return output_inputs

    for step in steps:
        output_inputs[step] = get_inputs(step, target=True)

    # recursively do the same for all the inputs
    inputs = set(itertools.chain(*output_inputs.values()))
    o = get_drake_data(inputs)
    output_inputs.update(o)

    return output_inputs


def to_drake_step(inputs, output):
    """
    Args:
        inputs: collection of input Steps
        output: output Step

    Returns: a string of the drake step for the given inputs and output
    """
    i = [output._yaml_filename]
    i.extend(map(lambda i: i._target_filename, list(inputs)))
    i.extend(output.dependencies)

    # add source file of output and its non-target inputs
    # if they're not in the drain library
    objects = get_inputs(output, target=False)
    objects.add(output)
    sources = set([os.path.abspath(inspect.getsourcefile(o.__class__)) for o in objects])
    i.extend([s for s in sources if not s.startswith(os.path.dirname(__file__))])

    output_str = '%' + output.__class__.__name__
    if output.name:
        output_str += ', %' + output.name
    if output.target:
        output_str += ', ' + os.path.join(output._target_filename)
    return '{output} <- {inputs} [method:drain]\n\n'.format(
            output=output_str, inputs=str.join(', ', i))


def to_drakefile(steps, preview=True, debug=False, input_drakefile=None, bindir=None):
    """
    Args:
        steps: collection of drain.step.Step objects for which to
            generate a drakefile
        preview: boolean, when False will create directories for output
            steps.  When True do not touch filesystem.
        debug: run python with '-m pdb'
        drakefile: path to drakefile to include
        bindir: path to drake binaries, defaults to ../bin/
    Returns:
        a string representation of the drakefile
    """
    data = get_drake_data(steps)
    drakefile = StringIO()

    if input_drakefile:
        drakefile.write('%context {}\n\n'.format(input_drakefile))

    if bindir is None:
        bindir = os.path.join(os.path.dirname(__file__), '..', 'bin')

    # if the step has a $OUTPUT, write drain.log to its directory
    drakefile.write("""drain()
    if [ $OUTPUT ]; then LOGFILE=$(dirname $OUTPUT)/drain.log; fi
    python %s %s/run_step.py $OUTPUT $INPUTS 2>&1 | tee $LOGFILE


""" % ('-m pdb' if debug else '', bindir))
    for output, inputs in data.items():
        if not preview:
            output.setup_dump()

        drakefile.write(to_drake_step(inputs, output))

    return drakefile.getvalue()


def is_target_filename(filename):
    return filename.endswith('/target')


def is_step_filename(filename):
    return filename.endswith('/step.yaml')
