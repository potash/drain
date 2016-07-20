import sys
from os.path import dirname

import drain.step
import drain.yaml
from drain.drake import is_target_filename, is_step_filename

if len(sys.argv) == 1:
    raise ValueError('Need at least one argument')

args = sys.argv[1:]

drain.step.OUTPUTDIR = dirname(dirname(dirname(args[0])))
drain.yaml.configure()

if is_target_filename(args[0]):
    output = drain.yaml.load(args[0])
    args = args[1:]
else:
    output = None

if not is_step_filename(args[0]):
    raise ValueError('Need a step to run')

step = drain.yaml.load(args[0])
inputs = []
for i in args[1:]:
    if is_step_filename(i) or is_target_filename(i):
        inputs.append(drain.yaml.load(i))

step.execute(output=output, inputs=inputs)
