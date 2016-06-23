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
        Args:
            name (str): Optional. Used for accessing this Step in a tree of Steps.
            target (bool): Optional. Sets whether this Step will be cached on the disk.
            kwargs: Every argument in kwargs will become part of this Step's serialization signature.
                    Objects have to be YAML-serializable for this to work.
                    Every kwarg becomes an attribute of this Step object. (Thus,
                    avoid names that are probably already taken.)
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
        """ Returns this Step's unique hash, which identifies the 
        Step's dump on disk. Depends on the constructor's kwargs. """
        return base64.urlsafe_b64encode(self._hasher.digest())

    @cached_property
    def named_steps(self):
        """ Gives a dictionary that maps names of Steps (if they have been set) to
        references to the various Steps. The dict includes this Step (if it's named), too.

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
            
            if len(self.inputs) < len(inputs_mapping):
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
        raise NotImplementedError
    
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

# checks if l is a collection of DataFrames or a DataFrame-valued dictionary
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

