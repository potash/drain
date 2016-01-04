import sys
import os
import yaml
from drain import drain

def is_target(filename):
    return filename.endswith('/target')

def is_step(filename):
    return filename.endswith('/step.yaml')

# given the filename of a step or a target, load it
def get_step(filename):
    yaml_filename = os.path.join(os.path.dirname(filename), 'step.yaml')
    return drain.from_yaml(yaml_filename)

if len(sys.argv) == 1:
    raise ValueError('Need at least one argument')

args = sys.argv[1:]

basedir = os.path.dirname(os.path.dirname(os.path.dirname(args[0])))
print basedir
drain.initialize(basedir)

if is_target(args[0]):
    output = get_step(args[0])
    args = args[1:]
else:
    output = None

if not is_step(args[0]):
    raise ValueError('Need a step to run')

step = get_step(args[0])
inputs = map(get_step, args[1:])

drain.run(step=step, output=output, inputs=inputs)
