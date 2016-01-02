import yaml
import itertools
from pprint import pformat
import util
import joblib
import os
import base64
import hashlib
import itertools
import logging

# TODO:
#    - create Drakefile from collection of steps by searching for targets and leaves
#    - random grid search
#    - get steps by name
#    - optional args like njobs that don't affect output? allow them to be excluded from yaml, hash, eq
#    - don't initialize yaml twice

BASEDIR=None

def run(step, targets=[]):
    if not step.has_output():
        if step in targets:
            logging.info('Loading %s' % step)
            step.load()
        else:
            for i in step.inputs:
                run(i, targets)
            logging.info('Running %s' % step)
            step.run()
    
    return step.output

class Step(object):
    def __init__(self, name=None, target=False, dirname=None, **kwargs):
        self.__kwargs__ = kwargs
        self.__name__ = name
        self.__target__ = target

        self.dirname = dirname
        for k in kwargs:
            setattr(self, k, kwargs[k])
        
        if 'inputs' not in kwargs:
            self.inputs = []

    def get_dirname(self):
        if self.dirname is not None:
            return self.dirname
        else:
            if BASEDIR is None:
                raise ValueError('BASEDIR not initialized')
            return os.path.join(BASEDIR, self.__class__.__name__, self.__dirname__())
        
    def run(self):
        raise NotImplementedError()
    
    def has_output(self):
        return hasattr(self, 'output')

    def is_target(self):
        return self.__target__
    
    def load(self, **kwargs):
        self.output = joblib.load(os.path.join(self.get_dirname(), 'dump', 'output.pkl'), **kwargs)
    
    def create_dump(self):
        dumpdir = os.path.join(self.get_dirname(), 'dump')
        if not os.path.isdir(dumpdir):
            os.makedirs(dumpdir)
            
        dump = False
        yaml_filename = os.path.join(self.get_dirname(), 'step.yaml')
        
        if not os.path.isfile(yaml_filename):
            dump = True
        else:
            with open(yaml_filename) as other:
                if yaml.load(other) != self:
                    logging.warning('Existing step.yaml does not match hash, regenerating')
                    dunmp = True
        
        if dump:
            with open(yaml_filename, 'w') as f:
                yaml.dump(self, f)

    def dump(self, **kwargs):
        self.create_dump()
        joblib.dump(self.output, os.path.join(self.get_dirname(), 'dump', 'output.pkl'), **kwargs)

    def __repr__(self):
        return '{name}({args})'.format(name=self.__class__.__name__,
                args=str.join(',', ('%s=%s' % i for i in self.__kwargs__.iteritems())))
    
    def __dirname__(self):
        hasher = hashlib.md5(yaml.dump(self))
        return base64.urlsafe_b64encode(hasher.digest())[0:8]
    
    def __hash__(self):
        return hash(yaml.dump(self)) # pyyaml dumps dicts in sorted order so this works
    
    def __eq__(self, other):
        return util.eqattr(self, other, '__kwargs__')

# temporary holder of step arguments
# used to expand ArgumentCollections by search()
# and to set inputs by serial()
class StepTemplate(object):
    def __init__(self, cls=Step, name=None, target=False, **kwargs):
        self.cls = cls
        self.name = name
        self.target = target
        self.kwargs = kwargs

    def copy(self, **kwargs):
        kwargs = util.merge_dicts(self.kwargs, kwargs)
        return StepTemplate(cls=self.cls, name=self.name, target=self.target, **kwargs)
    
    def construct(self):
        if 'inputs' in self.kwargs:
            self.kwargs['inputs'] = [i.construct() for i in self.kwargs['inputs']]
        return self.cls(target=self.target, name=self.name, **self.kwargs)

    def __repr__(self):
        return '{name}({args})'.format(name=self.cls.__name__,
                args=str.join(',', ('%s=%s' % i for i in self.kwargs.iteritems())))
 
    def __eq__(self, other):
        for attr in ('cls', 'name', 'target', 'kwargs'):
            if not util.eqattr(self, other, attr):
                print attr
                return False
        return True

class Add(Step):
    def __init__(self, value, **kwargs):
        Step.__init__(self, value=value, **kwargs)

    def run(self):
        r = self.value
        for i in self.inputs:
            r += i.output
            
        self.output = r
        
class ArgumentCollection(object):
    def __init__(self, collection):
        if not hasattr(collection, '__iter__'):
            raise ValueError('Not a collection: %s' % collection)
        self.collection = collection    

def argument_product(args):
    for k in args:
        if isinstance(args[k], dict):
            args[k] = ArgumentCollection(argument_product(args[k]))
    
    product_vars = [k for k in args if isinstance(args[k], ArgumentCollection)]
    for k in product_vars:
        args[k] = args[k].collection
    dicts = util.dict_product(args, product_vars)
    return dicts

def step_product(step):
    return [step.copy(**d) for d in argument_product(step.kwargs)]

# a list of steps
# expands ArgumentCollections
def parallel(*inputs):
    return list(itertools.chain(*map(util.make_list, inputs)))

def search(*inputs):
    return parallel(*map(step_product, inputs))

# compose the list of steps by setting s(n).inputs = s(n-1).get_steps()
def serial(*inputs):
    psteps = None
    for steps in map(util.make_list, inputs):
        if psteps is not None:
            steps = map(lambda s: s.copy(inputs=psteps), steps)
        psteps = steps
    return psteps
    
# take the product of steps from each step
def product(*inputs):
    return list(itertools.product(*map(util.make_list, inputs)))
    
def step_multi_representer(dumper, data):
    tag = '!obj:%s.%s' % (data.__class__.__module__, data.__class__.__name__)
    return dumper.represent_mapping(tag, data.__kwargs__)

def step_multi_constructor(loader, tag_suffix, node):
    if tag_suffix == '':
        tag_suffix = ':__main__.Step'
    cls = util.get_attr(tag_suffix[1:])
    args = loader.construct_mapping(node)
    return StepTemplate(cls, **args)

def get_sequence_constructor(method):
    def constructor(loader, node):
        seq = loader.construct_sequence(node)
        return method(*seq)
    return constructor

def range_constructor(loader, node):
    args = loader.construct_sequence(node)
    return ArgumentCollection(xrange(*args))

def list_constructor(loader, node):
    args = loader.construct_sequence(node)
    return ArgumentCollection(args)

def powerset_constructor(loader, node):
    args = loader.construct_sequence(node)
    return ArgumentCollection(itertools.chain.from_iterable(itertools.combinations(args, r) for r in range(len(args)+1)))

def initialize(basedir):
    global BASEDIR
    BASEDIR = basedir
    yaml.add_multi_representer(Step, step_multi_representer)
    yaml.add_multi_constructor('!step', step_multi_constructor)
    
    yaml.add_constructor('!parallel', get_sequence_constructor(parallel))
    yaml.add_constructor('!search', get_sequence_constructor(search))
    yaml.add_constructor('!serial', get_sequence_constructor(serial))
    yaml.add_constructor('!product', get_sequence_constructor(product))

    yaml.add_constructor('!range', range_constructor)
    yaml.add_constructor('!powerset', powerset_constructor)
    yaml.add_constructor('!list', list_constructor)

def to_drake_step(step):
    inputs = [step.yaml_filename] # first input is the step itself
    input_targets = get_input_targets(step)
    inputs.extend(map(lambda i: os.path.join(i.dirname, 'target'), input_targets)) # followed by its input drake steps
    
    output = os.path.join(step.dirname, 'target') if step.__target__ else ''
    return '{output} <- {inputs} [method:drain]'.format(output=output, inputs=str.join(', ', inputs))

def get_targets(step, ignore):
    outputs = set()
    if not ignore and step.is_target():
        outputs.add(step)
    else:
        for i in step.inputs:
            outputs.update(get_targets(i, False))

    return outputs

def get_input_targets(step):
    return get_targets(step, ignore=True)

def get_output_targets(step):
    return get_targets(step, ignore=False)

def to_drake_helper(steps):
    output_inputs = {}
    output_no_outputs = {}
    no_output_inputs = {}

    for step in steps:
        if step.is_target():
            if step not in output_inputs:
                output_inputs[step] = get_input_targets(step)
            print output_inputs
        else:
            outputs = get_output_targets(step)
            if len(outputs) == 1:
                output = outputs.pop()
                if output in output_no_outputs:
                    output_no_outputs[output].add(step)
                else:
                    output_no_outputs[output] = {step}
                    if output not in output_inputs:
                        output_inputs[output] = get_input_targets(output)
            else:
                no_output_inputs[step] = outputs

    # recursively do the same for all the inputs
    inputs = set()
    for i in itertools.chain(output_inputs.values(), no_output_inputs.values()):
        inputs |= i

    if len(inputs) > 0:
        o1, o2, o3 = to_drake_helper(inputs)
        util.dict_update_union(output_inputs, o1)
        util.dict_update_union(output_no_outputs, o2)
        util.dict_update_union(no_output_inputs, o3)


    return output_inputs, output_no_outputs, no_output_inputs

def to_drake(steps):
    output_inputs, output_no_outputs, no_output_inputs = to_drake_helper(steps)
    for output, inputs in output_inputs.iteritems():
        if output in output_no_outputs:
            no_outputs = output_no_outputs[output]
        else:
            no_outputs = set()
        print to_drake_step(output, inputs, no_outputs)

    for no_output, inputs in no_output_inputs.iteritems():
        print to_drake_step(None, inputs, set(no_output))

#def to_drake_step(inputs, output, no_outputs):
#    if len(no_outputs) > 0:
#        collect them into one step yaml
#    else:
#        use step yaml
#    output/target <- step.yaml, input/target, input2/target, ...
