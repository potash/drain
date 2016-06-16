from __future__ import absolute_import
import yaml

import inspect
import sys
import itertools
import pandas as pd
from cached_property import cached_property
from pprint import pformat

try:
    import StringIO
except ImportError:
    from io import StringIO

from sklearn.base import _pprint
import joblib
import os
import base64
import hashlib
import itertools
import logging
import shutil
import warnings
from tables import NaturalNameWarning

from drain import util

BASEDIR=None

# run the given step
# inputs should be loaded from disk
# output should be written to disk
# also loads targets from disk-- could make this optional
# recreate the dump directory before dumping
# if load_targets, assume all targets have been run and dumped
# TODO: move this to Step.execute()
def run(step, inputs=None, output=None, load_targets=False):
    if step == output:
        if os.path.exists(step._target_dump_dirname):
            shutil.rmtree(step._target_dump_dirname)
        if os.path.exists(step._target_filename):
            os.remove(step._target_filename)
        os.makedirs(step._target_dump_dirname)

    if inputs is None:
        inputs = []

    if not step.has_result():
        if (step in inputs or (load_targets and step.is_target())) and not step.has_result():
            logging.info('Loading\n\t%s' % str(step).replace('\n','\n\t'))
            step.load()
        else:
            for i in step.inputs:
                run(step=i, inputs=inputs, output=output, load_targets=load_targets)

            args, kwargs = step.map_inputs()
            logging.info('Running\n\t%s' % str(step).replace('\n','\n\t'))
            step.set_result(step.run(*args, **kwargs))

    if step == output:
        step.dump()
        util.touch(step._target_filename)

    return step.get_result()

def load(steps):
    """
    safely load steps, excluding those that fail
    """
    loaded = []
    for s in steps:
        try:
            s.load()
            loaded.append(s)
        except:
            pass
    return loaded

class Step(object):
    def __init__(self, name=None, target=False, **kwargs):
        """
        name and target are special because a Step's result 
        is independent of their values.
        """
        self._kwargs = kwargs
        self._name = name
        self._target = target

        for k in kwargs:
            setattr(self, k, kwargs[k])
        
        # avoid overriding inputs if it was set somewhere else
        if not hasattr(self, 'inputs'):
            self.inputs = []

        if not hasattr(self, 'dependencies'):
            self.dependencies = []

    @cached_property
    def _hasher(self):
        # TODO: check to make sure configure_yaml has been called!
        return hashlib.md5(yaml.dump(self).encode('utf-8'))

    @cached_property
    def _digest(self):
        return base64.urlsafe_b64encode(self._hasher.digest())

    @cached_property
    def named_steps(self):
        """
        returns a dictionary of name: step pairs
        recursively searches self and inputs
        doesn't use a visited set so that it can be a property, 
            also seems faster this way (needs testing)
        """
        named = {}

        for i in self.inputs:
            for name,step in i.named_steps.iteritems():
                if name in named and step != named[name]:
                    raise NameError('Multiple steps with the same name: %s' % name)
                named[name] = step
        
        if self.has_name():
            name = self.get_name()
            if name in named and named[name] != self:
                raise NameError('Multiple steps with the same name: %s' % name)
            named[name] = self

        return named

    def get_input(self, name):
        for i in self.inputs:
            step = i.get_input(name)
            if step is not None:
                return step

        if self.get_name() == name:
            return self

    def get_name(self):
        return self._name

    def has_name(self):
        return self._name is not None

    @cached_property
    def named_arguments(self):
        d = dict()
        named = self.named_steps

        for name, step in named.iteritems():
            for k,v in step.get_arguments().iteritems():
                d[(name, k)] = v

        return d

    # returns a shallow copy of _kwargs
    # any argument specified is excluded if False
    def get_arguments(self, **include):
        d = dict(self._kwargs)

        for k in include:
            if not include[k] and k in d:
                d.pop(k)
        return d

    def map_inputs(self):
        kwargs = {}
        args = []

        if hasattr(self, 'inputs_mapping'):
            inputs_mapping = util.make_list(self.inputs_mapping)
            
            diff = len(self.inputs) - len(inputs_mapping)
            if diff < 0:
                raise ValueError('Too many inputs_mappings')

            for input, mapping in zip(self.inputs, inputs_mapping):
                result = input.get_result()
                if isinstance(mapping, dict):
                    # pass through any missing keys, so {} is the identity
                    # do it first so that inputs_mapping overrides keys
                    for k in set(result.keys()).difference(set(mapping.keys())):
                        kwargs[k] = result[k]

                    for k in mapping:
                        if mapping[k] is not None:
                            kwargs[mapping[k]] = result[k]

                elif isinstance(mapping, basestring):
                    kwargs[mapping] = input.get_result()
                elif mapping is None: # drop Nones
                    pass
                else:
                    raise ValueError('Input mapping is neither dict nor str: %s' % mapping)

            mapped_inputs = len(inputs_mapping)
        else:
            mapped_inputs = 0

        for i in range(mapped_inputs, len(self.inputs)):
            result = self.inputs[i].get_result()
            # without a mapping we handle two cases
            # when the result is a dict merge it with a global dict
            if isinstance(result, dict):
                # but do not override
                kwargs.update({k:v for k,v in result.iteritems() if k not in kwargs})
            # otherwise use it as a positional argument
            else:
                args.append(result)

        return args, kwargs

    def get_result(self):
        return self._result

    def set_result(self, result):
        self._result = result

    def has_result(self):
        return hasattr(self, '_result')
    
    @cached_property
    def _target_dirname(self):
        if BASEDIR is None:
            raise ValueError('BASEDIR not initialized')

        return os.path.join(BASEDIR, self.__class__.__name__, self._digest[0:8])

    @cached_property
    def _target_yaml_filename(self):
        return os.path.join(self._target_dirname, 'step.yaml')

    @cached_property
    def _target_dump_dirname(self):
        return os.path.join(self._target_dirname, 'dump')

    @cached_property
    def _target_filename(self):
        return os.path.join(self._target_dirname, 'target')
        
    def run(self, *args, **kwargs):
        pass
    
    def is_target(self):
        return self._target
    
    def load(self, **kwargs):
        hdf_filename = os.path.join(self._target_dirname, 'dump', 'result.h5')
        if os.path.isfile(hdf_filename):
            store = pd.HDFStore(hdf_filename)
            keys = store.keys()
            if keys == ['/df']:
                self.set_result(store['df'])
            else:
                if set(keys) == set(map(lambda i: '/%s' % i, range(len(keys)))):
                    # keys are not necessarily ordered
                    self.set_result([store[str(k)] for k in range(len(keys))])
                else:
                    self.set_result({k[1:]:store[k] for k in keys})
                
        else:
            self.set_result(joblib.load(os.path.join(self._target_dirname, 'dump', 'result.pkl')))

    def setup_dump(self):
        dumpdir = self._target_dump_dirname
        if not os.path.isdir(dumpdir):
            os.makedirs(dumpdir)
            
        dump = False
        yaml_filename = self._target_yaml_filename
        
        if not os.path.isfile(yaml_filename):
            dump = True
        else:
            with open(yaml_filename) as f:
                if f.read() != yaml.dump(self):
                    logging.warning('Existing step.yaml does not match hash, regenerating')
                    dump = True
        
        if dump:
            with open(yaml_filename, 'w') as f:
                yaml.dump(self, f)

    def dump(self):
        self.setup_dump()
        result = self.get_result()
        if isinstance(result, pd.DataFrame):
            result.to_hdf(os.path.join(self._target_dump_dirname, 'result.h5'), 'df')
        elif hasattr(result, '__iter__') and is_dataframe_collection(result):
            if not isinstance(result, dict):
                keys = map(str, range(len(result)))
                values = result
            else:
                keys = result.keys()
                values = result.values()

            store = pd.HDFStore(os.path.join(self._target_dump_dirname, 'result.h5'))
            # ignore NaturalNameWarning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=NaturalNameWarning)
                for key, df in zip(keys, values):
                    store.put(key, df, mode='w')
        else:
            joblib.dump(self.get_result(), os.path.join(self._target_dump_dirname, 'result.pkl'))

    def __repr__(self):
        class_name = self.__class__.__name__

        return '%s(%s)' % (class_name, 
                _pprint(self._kwargs, offset=len(class_name)),)

    def __hash__(self):
        return int(self._hasher.hexdigest(), 16)
    
    def __eq__(self, other):
        if not isinstance(other, Step):
            return False
        else:
            return util.eqattr(self, other, '__class__') and util.eqattr(self, other, '_kwargs')

    def __ne__(self, other):
        return not self.__eq__(other)

def is_dataframe_collection(l):
    if isinstance(l, dict):
        l = l.values()

    for i in l:
        if not isinstance(i, pd.DataFrame):
            return False
    return True

class Construct(Step):
    def __init__(self, __class_name__, name=None, target=False, **kwargs):
        Step.__init__(self, __class_name__=__class_name__, name=name, target=target, **kwargs)

    def run(self, **update_kwargs):
        kwargs = self.get_arguments(inputs=False, inputs_mapping=False)
        kwargs.update(update_kwargs)
        cls = util.get_attr(kwargs.pop('__class_name__'))
        return cls(**kwargs)

class Echo(Step):
    def run(self, *args, **kwargs):
        for i in self.inputs:
            print('%s: %s' % (i, i.get_result()))


class Scalar(Step):
    def __init__(self, value, **kwargs):
        Step.__init__(self, value=value, **kwargs)

    def run(self):
        return self.value

class Add(Step):
    def run(self, *values):
        return sum(values)

class Divide(Step):
    def run(self, numerator, denominator):
        return numerator / denominator
   
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

# if step is a target returns set containing step
# else returns empty set
def get_output_targets(step):
    return get_targets(step, ignore=False)

# returns two Step:set<Step> dicts
# output_inputs: maps output to inputs
# no_output_inputs: maps no_output step with *multiple* target inputs to them
def get_drake_data_helper(steps):
    output_inputs = {}
    no_output_inputs = {}

    for step in steps:
        if step.is_target():
            if step not in output_inputs:
                output_inputs[step] = get_input_targets(step)
        else:
            outputs = get_output_targets(step)
            no_output_inputs[step] = outputs

    # recursively do the same for all the inputs
    inputs = set()
    for i in itertools.chain(output_inputs.values(), no_output_inputs.values()):
        inputs |= i

    if len(inputs) > 0:
        o1, o2 = get_drake_data_helper(inputs)
        util.dict_update_union(output_inputs, o1)
        util.dict_update_union(no_output_inputs, o2)


    return output_inputs, no_output_inputs

# returns a dictionary mapping outputs to their inputs
# an output is any target in the step tree
# a no-output is a leaf node of the step tree which is not a target
def get_drake_data(steps):
    drake_data = {}
    output_inputs, no_output_inputs = get_drake_data_helper(steps)

    return util.merge_dicts(output_inputs, no_output_inputs)

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

