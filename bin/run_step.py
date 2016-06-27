import sys
import os
from os.path import dirname
import logging

import drain.step
import drain.yaml

def is_target(filename):
    return filename.endswith('/target')

def is_step(filename):
    return filename.endswith('/step.yaml')

# given the filename of a step or a target, load it
def get_step(filename):
    yaml_filename = os.path.join(dirname(filename), 'step.yaml')
    return drain.yaml.load(yaml_filename)

if len(sys.argv) == 1:
    raise ValueError('Need at least one argument')

args = sys.argv[1:]

drain.step.OUTPUTDIR = dirname(dirname(dirname(args[0])))
drain.yaml.configure()

if is_target(args[0]):
    output = get_step(args[0])
    args = args[1:]
else:
    output = None

if not is_step(args[0]):
    raise ValueError('Need a step to run')

step = get_step(args[0])
inputs = []
for i in args[1:]:
    if is_step(i) or is_target(i):
        inputs.append(get_step(i))

drain.step.run(step=step, output=output, inputs=inputs)
