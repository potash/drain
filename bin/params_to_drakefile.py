import os
import json
import itertools
import sys
import yaml
from drain import util
    
# cartesian product of dict whose values are lists
def dict_product(d):
    items = sorted(d.items())
    keys, values = zip(*items)
    a = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return a

# concatenate a list of dict products
def list_dict_product(l):
    return list(itertools.chain(*[dict_product(d) for d in l]))

def drake_step(basedir, params, method):
    h = util.hash_yaml_dict(params)
    d = os.path.join(basedir, method, h + '/')

    if not os.path.exists(d):
        os.makedirs(d)
    
    dirname = os.path.join(d, 'output/')
    params_file = os.path.join(d, 'params.yaml')

    with file(params_file, 'w') as f:
        yaml.dump(params, f)

    return dirname + ' <- ' + params_file + ' [method:' + method + ']'
    
with open(sys.argv[1]) as f:
    params = yaml.load(f)

outputdir = sys.argv[2]

drakefile = sys.argv[3]

runs = []

data = list_dict_product(params['data'])
transforms = list_dict_product(params['transforms'])
models = list_dict_product(params['models'])
metrics = list_dict_product(params['metrics'])

# TODO: include drain/bin/Drakefile containing drake method definitions

# data steps
for d in data:
    print drake_step(outputdir, d, 'data')

# model steps
runs = []
i = 0
for d,t,m in itertools.product(data,transforms,models):
    i = i + 1
    p = {'data': d, 'transform':t, 'model':m, 'metrics':metrics}

    print drake_step(outputdir, p, 'model')
