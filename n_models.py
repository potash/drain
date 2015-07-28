#!/usr/bin/python
# Usage: n_models.py <param_file>
# Prints the number of models to be run by this parameter file

import yaml
import sys

with open(sys.argv[1]) as f:
    param_dicts = yaml.load(f)
    
outputs = param_dicts['outputs']
n = 0
for output in outputs:
    m = 1
    m *= reduce(lambda x,y: x*y, map(lambda x: len(x), output['transform'].values()))
    m *= reduce(lambda x,y: x*y, map(lambda x: len(x), output['model'].values()))
    n += m

print 'Running {} models...'.format(n)
