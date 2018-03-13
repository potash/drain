import yaml

import collections
import inspect
import pandas as pd
from cached_property import cached_property
from six import string_types
from six.moves import zip_longest

from sklearn.base import _pprint
import joblib
import os
import traceback
import hashlib
import logging
import shutil
import warnings
from tables import NaturalNameWarning

from . import util
import drain

_STEP_CACHE = {}


def load(steps, reload=False):
    """
    safely load steps in place, excluding those that fail
    Args:
        steps: the steps to load
    """
    # work on collections by default for fewer isinstance() calls per call to load()
    if reload:
        _STEP_CACHE.clear()

    if callable(steps):
        steps = steps()

    if not isinstance(steps, collections.Iterable):
        return load([steps])[0]

    loaded = []
    for s in steps:
        digest = s._digest
        if digest in _STEP_CACHE:
            loaded.append(_STEP_CACHE[digest])
        else:
            try:
                s.load()
                _STEP_CACHE[digest] = s
                loaded.append(s)
            except(Exception):
                logging.warn('Error during step load:\n%s' %
                             util.indent(traceback.format_exc()))

    return loaded


class Step(object):
    def __new__(cls, *args, **kwargs):
        # use inspection to get positional argument names
        argspec = inspect.getargspec(cls.__init__)
        nargs = zip(argspec.args[1:len(args)+1], args)
        kwargs.update(nargs)

        obj = object.__new__(cls)
        obj._kwargs = kwargs

        return obj

    def __init__(self, inputs=None, dependencies=None, **kwargs):
        """
        initialize name and target attributes
        and set all arguments as attributes
        """
        self.target = False
        self.name = None

        if not hasattr(self, 'inputs'):
            self.inputs = inputs if inputs is not None else []
        self.dependencies = dependencies if dependencies is not None else []

        for k, v in kwargs.items():
            setattr(self, k, v)

    def execute(self, inputs=None, output=None, load_targets=False):
        """
        Run this step, recursively running or loading inputs.
        Used in bin/run_step.py which is run by drake.
        Args:
            inputs: collection of steps that should be loaded
            output: step that should be dumped after it is run
            load_targets (boolean): load all steps which are targets.
                This argument is not used by run_step.py because target
                does not get serialized. But it can be useful for
                running steps directly.
        """
        if self == output:
            if os.path.exists(self._dump_dirname):
                shutil.rmtree(self._dump_dirname)
            if os.path.exists(self._target_filename):
                os.remove(self._target_filename)
            os.makedirs(self._dump_dirname)

        if inputs is None:
            inputs = []

        if not hasattr(self, 'result'):
            if self in inputs or (load_targets and self.target):
                logging.info('Loading\n%s' % util.indent(str(self)))
                self.load()
            else:
                for i in self.inputs:
                    i.execute(inputs=inputs, output=output,
                              load_targets=load_targets)

                args = merge_results(self.inputs)
                logging.info('Running\n%s' % util.indent(str(self)))
                self.result = self.run(*args.args, **args.kwargs)

        if self == output:
            logging.info('Dumping\n%s' % util.indent(str(self)))
            self.dump()
            util.touch(self._target_filename)

    @cached_property
    def _hasher(self):
        return hashlib.md5(yaml.dump(self).encode('utf-8'))

    @cached_property
    def _digest(self):
        """ Returns this Step's unique hash, which identifies the
        Step's dump on disk. Depends on the constructor's kwargs. """
        return self._hasher.hexdigest()

    def get_input(self, value, _search=None):
        """
        Searches the tree for a step
        Args:
            value: The value to search for. If value is a string then the search looks for
                a step of that name. If the value is a type, it looks for a step
                of that type.
        Returns: The first step found via a depth-first search.
        """
        if _search is None:
            if isinstance(value, string_types):
                _search = lambda s: s.name  # noqa: E731
            elif isinstance(value, type):
                _search = type

        for i in self.inputs:
            step = i.get_input(value, _search)
            if step is not None:
                return step

        if _search(self) == value:
            return self

    def get_inputs(self, _visited=None):
        """
        Returns: the set of all input steps
        """
        if _visited is None:
            _visited = set()

        _visited.add(self)

        for i in self.inputs:
            if i not in _visited:
                i.get_inputs(_visited)

        return _visited

    # returns a shallow copy of _kwargs
    # any argument specified is excluded if False
    def get_arguments(self, **include):
        """
        return a shallow copy of self._kwargs
        passing {key}=False will pop the {key} from the dict
        e.g. get_arguments(inputs=False) returns all keywords except inputs
        """
        d = dict(self._kwargs)

        for k in include:
            if not include[k] and k in d:
                d.pop(k)
        return d

    @cached_property
    def _output_dirname(self):
        if drain.PATH is None:
            raise ValueError('drain.PATH not set')

        return os.path.join(drain.PATH, self.__class__.__name__, self._digest[0:8])

    @cached_property
    def _yaml_filename(self):
        return os.path.join(self._output_dirname, 'step.yaml')

    @cached_property
    def _dump_dirname(self):
        return os.path.join(self._output_dirname, 'dump')

    @cached_property
    def _target_filename(self):
        return os.path.join(self._output_dirname, 'target')

    def run(self):
        raise NotImplementedError

    def load(self):
        """
        Load this step's result from its dump directory
        """
        hdf_filename = os.path.join(self._dump_dirname, 'result.h5')
        if os.path.isfile(hdf_filename):
            store = pd.HDFStore(hdf_filename, mode='r')
            keys = store.keys()
            if keys == ['/df']:
                self.result = store['df']
            else:
                if set(keys) == set(map(lambda i: '/%s' % i, range(len(keys)))):
                    # keys are not necessarily ordered
                    self.result = [store[str(k)] for k in range(len(keys))]
                else:
                    self.result = {k[1:]: store[k] for k in keys}

        else:
            self.result = joblib.load(
                    os.path.join(self._output_dirname, 'dump', 'result.pkl'))

    def setup_dump(self):
        """
        Set up dump, creating directories and writing step.yaml file
        containing yaml dump of this step.

        {drain.PATH}/{self._digest}/
            step.yaml
            dump/
        """
        dumpdir = self._dump_dirname
        if not os.path.isdir(dumpdir):
            os.makedirs(dumpdir)

        dump = False
        yaml_filename = self._yaml_filename

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
        if isinstance(self.result, pd.DataFrame):
            self.result.to_hdf(os.path.join(self._dump_dirname, 'result.h5'), 'df')
        elif util.is_instance_collection(self.result, [pd.Series, pd.DataFrame]):
            if not isinstance(self.result, dict):
                keys = map(str, range(len(self.result)))
                values = self.result
            else:
                keys = self.result.keys()
                values = self.result.values()

            store = pd.HDFStore(os.path.join(self._dump_dirname, 'result.h5'))
            # ignore NaturalNameWarning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=NaturalNameWarning)
                for key, df in zip(keys, values):
                    store.put(key, df, mode='w')
                store.close()
        else:
            joblib.dump(self.result, os.path.join(self._dump_dirname, 'result.pkl'))

    def __repr__(self):
        class_name = self.__class__.__name__
        args = _pprint(self._kwargs, offset=len(class_name))
        return '%s(%s)' % (class_name,
                           args.replace('\\n', '\n'))

    def __hash__(self):
        return int(self._hasher.hexdigest(), 16)

    def __eq__(self, other):
        if not isinstance(other, Step):
            return False
        else:
            return util.eqattr(self, other, '__class__') and util.eqattr(self, other, '_kwargs')

    def __ne__(self, other):
        return not self.__eq__(other)


class Arguments(object):
    """
    A simple wrapper for positional and keyword arguments
    """
    def __init__(self, args=None, kwargs=None):
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}


def _simplify_arguments(arguments):
    """
    If positional or keyword arguments are empty return only one or the other.
    """
    if len(arguments.args) == 0:
        return arguments.kwargs
    elif len(arguments.kwargs) == 0:
        return arguments.args
    else:
        return arguments


def merge_results(inputs, arguments=None):
        """
        Merges results to form arguments to run(). There are two cases for each result:
         - dictionary: dictionaries get merged and passed as keyword arguments
         - list: lists get concatenated to positional arguments
         - Arguments: kwargs gets merged and args gets appended
         - else: concatenated and passed as postitional arguments
        Args:
            inputs: the inputs whose results to merge
            arguments: an optional existing Arguments object to merge into
        """
        if arguments is None:
            arguments = Arguments()

        args = arguments.args
        kwargs = arguments.kwargs

        for i in inputs:
            # without a mapping we handle two cases
            # when the result is a dict merge it with a global dict
            if isinstance(i.result, dict):
                # but do not override
                kwargs.update({k: v for k, v in i.result.items() if k not in kwargs})
            elif isinstance(i.result, list):
                args.extend(i.result)
            elif isinstance(i.result, Arguments):
                args.extend(i.result.args)
                kwargs.update({k: v for k, v in i.result.kwargs.items() if k not in kwargs})
            # otherwise use it as a positional argument
            else:
                args.append(i.result)

        return arguments


class GetItem(Step):
    """
    Given a step that returns a dict, this Step grabs a single value from it.
    """
    def __init__(self, step, key=None):
        inputs = [step]
        if isinstance(key, Step):
            inputs.append(key)

        Step.__init__(self, step=step, key=key, inputs=inputs)

    def run(self, *args, **kwargs):
        key = self.key.result if isinstance(self.key, Step) else self.key
        return self.step.result[key]


class MapResults(Step):
    """
    This step maps the results of its inputs into a new form of arguments and keyword arguments.
    It is a useful connector between steps.
    """

    DEFAULT = 1

    def __init__(self, inputs, mapping):
        """
        Args:
            inputs: input step or list of steps whose results will be mapped
            mapping: the mapping contains a list of maps. Each map can be:
                - dictionary: when a result has keyword entries, use this to remap them.
                    The keys are the source key and the values are the destination keys.
                    Missing keys are treated as the identity.
                    A destination of None removes the corresponding result.
                    Thus an empty dictionary is the identity.
                - list: when a result has positional entries, use a list to map them to keyword
                    arguments. Each entry is a string, or None removes the entry.
                - string: the entire result gets mapped to keyword of the given string.
                - None: remove results from the given input.
                - MapResults.DEFAULT: default mapping (as in merge_results())

        Additional entries are mapped in the default way.
        """
        Step.__init__(self, inputs=util.make_list(inputs), mapping=mapping)

    def run(self, *args, **kwargs):
        mapping = util.make_list(self.mapping)
        arguments = Arguments()
        args = arguments.args
        kwargs = arguments.kwargs

        if len(self.inputs) < len(mapping):
            raise ValueError('Too many maps')

        for input, m in zip_longest(self.inputs, mapping, fillvalue=self.DEFAULT):
            result = input.result
            if isinstance(m, dict):
                # pass through any missing keys, so {} is the identity
                # do it first so that mapping overrides keys
                for k in set(result.keys()).\
                        difference(set(m.keys())):
                    kwargs[k] = result[k]

                for k in m:
                    if m[k] is not None:
                        kwargs[m[k]] = result[k]
            elif isinstance(m, list):
                if len(m) > len(result):
                    raise ValueError("More keywords than results")
                for kw, r in zip(m, result):
                    if kw is not None:
                        kwargs[kw] = r
                # unmapped args are appended to positional args
                args.extend(result[len(m):])
            elif isinstance(m, string_types):
                kwargs[m] = input.result
            elif m is None:  # drop Nones
                pass
            elif m is self.DEFAULT:
                merge_results([input], arguments)
            else:
                raise ValueError('Input mapping is neither dict nor str: %s' % m)

        return _simplify_arguments(arguments)


class Call(Step):
    def __init__(self, _base, _method_name=None, inputs=None, **kwargs):
        """
        Args:
            _base: The base from which to call.
                If a drain Step, then use its result.
            _method_name: Optional method name.
                If _base is to be called, leave this None.
        """
        if inputs is None:
            inputs = []
        if isinstance(_base, Step):
            inputs = [_base] + inputs

        Step.__init__(self, _base=_base, _method_name=_method_name,
                      inputs=inputs, **kwargs)

    def run(self, *args, **update_kwargs):
        kwargs = self.get_arguments(
                _base=False, _method_name=False, inputs=False)
        kwargs.update(update_kwargs)

        if isinstance(self._base, Step):
            call = self._base.result
            args = args[1:]
        elif isinstance(self._base, str):
            call = util.get_attr(self._base)
        else:
            call = self._base

        if self._method_name is not None:
            call = getattr(call, self._method_name)

        return call(*args, **kwargs)


def _expand_inputs(step, steps=None):
    """
    Returns the set of this step and all steps passed to the constructor (recursively).
    """
    if steps is None:
        steps = set()

    for arg in step._kwargs.values():
        if isinstance(arg, Step):
            _expand_inputs(arg, steps=steps)
        elif util.is_instance_collection(arg, Step):
            for s in util.get_collection_values(arg):
                _expand_inputs(s, steps=steps)

    steps.add(step)
    return steps


def _collect_kwargs(step, drop_duplicate_names=True):
    """
    Collect the kwargs of this step and inputs passed to the constructor (recursively)
    Returns: dictionary of name: kwargs pairs where name is the name of
        a step and kwargs is its kwargs minus inputs. If the step doesn't have
        a name __class__.__name__ is used.
    """
    dicts = {}
    duplicates = set()

    for s in _expand_inputs(step):
        name = s.name if s.name is not None else s.__class__.__name__
        if name in dicts.keys():
            if drop_duplicate_names:
                duplicates.add(name)
            else:
                raise ValueError("Duplicate step names: %s" % name)

        d = dict(s._kwargs)
        d = {k: v for k, v in d.items()
             if not (isinstance(v, Step) or util.is_instance_collection(v, Step))}
        dicts[name] = d

    dicts = {k: v for k, v in dicts.items() if k not in duplicates}
    return dicts


def _output_dirnames(workflow=None, leaf=False):
    """
    Args:
        workflow: optional collection of steps
        leaf: only include leaves of the workflow

    Returns: If workflow is specified, returns output directories for all target
        steps in the workflow. If no workflow specified, returns all extant
        output directories in drain.PATH.
    """
    if workflow is None:
        dirs = set()
        for cls in os.listdir(drain.PATH):
            for step in os.listdir(os.path.join(drain.PATH, cls)):
                dirs.add(os.path.join(drain.PATH, cls, step))
        return dirs
    else:
        if leaf:
            steps = [step for step in workflow if step.target]
        else:
            steps = util.union(step.get_inputs() for step in workflow if step.target)

        return set(step._output_dirname for step in steps)
