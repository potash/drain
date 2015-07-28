#!/usr/bin/python

import os
import json
import itertools
import sys
import yaml
    
def dict_product(d):
    items = sorted(d.items())
    keys, values = zip(*items)
    a = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return a

datadir = sys.argv[1]
outputdir = sys.argv[2]

with open(sys.argv[3]) as f:
    param_dicts = yaml.load(f)

data = param_dicts['data']

outputs = param_dicts['outputs']

# TODO only do this and the output below if corresponding arg switches are present
data['source'] = 'pkl'
data['directory'] = datadir

for output in outputs:
    transforms = output['transform']
    models = output['model']

    transforms_product = dict_product(transforms)
    
    for model_params in dict_product(models):
        for transform_params in transforms_product:
            params = {'transform': transform_params, 'model': model_params, 'data': data}
            
            h = hex(hash(yaml.dump(params)))
            params['output'] = os.path.join(outputdir, h[h.index('x')+1:])
            
            print yaml.dump(params).replace('\n','\\n')
