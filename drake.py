import itertools
import os
import inspect

from drain.util import StringIO

# returns set of target steps
# used below by get_input_targets and get_output_targets
def get_targets(step, ignore):
    outputs = set()
    if not ignore and step.is_target():
        outputs.add(step)
    else:
        for i in step.inputs:
            outputs.update(get_targets(i, False))

    return outputs

# traverse input tree for closest parent targets
def get_input_targets(step):
    return get_targets(step, ignore=True)

# returns a dictionary of outputs mapped to inputs
# note that an output is either a target
# or a leaf node in the step tree
def get_drake_data(steps):
    output_inputs = {}
    if len(steps) == 0:
        return output_inputs

    for step in steps:
        output_inputs[step] = get_input_targets(step)

    # recursively do the same for all the inputs
    #    inputs |= i
    inputs = set(itertools.chain(*output_inputs.values()))
    o = get_drake_data(inputs)
    output_inputs.update(o)

    return output_inputs

# returns a drake step string for the given inputs and outputs
def to_drake_step(inputs, output):
    i = [output._target_yaml_filename]
    i.extend(map(lambda i: i._target_filename, list(inputs)))
    i.extend(output.dependencies)
    # add source file if it's not in the drain library
    # TODO: do this for all non-target inputs, too
    source = os.path.abspath(inspect.getsourcefile(output.__class__))
    if not source.startswith(os.path.dirname(__file__)):
        i.append(source)

    output_str = '%' + output.__class__.__name__
    if output.is_target():
        output_str += ', ' + os.path.join(output._target_filename)
    return '{output} <- {inputs} [method:drain]\n\n'.format(output=output_str, inputs=str.join(', ', i))

# if preview then don't create the dump directories and step yaml files
def to_drakefile(steps, preview=True, debug=False, bindir=None):
    data = get_drake_data(steps)
    drakefile = StringIO.StringIO()

    bindir = os.path.join(os.path.dirname(__file__), 'bin')
    drakefile.write("drain()\n\tpython %s %s/run_step.py $OUTPUT $INPUTS 2>&1\n\n" % ('-m pdb' if debug else '', bindir))
    for output, inputs in data.iteritems():
        if not preview:
            output.setup_dump()

        drakefile.write(to_drake_step(inputs, output))

    return drakefile.getvalue()
