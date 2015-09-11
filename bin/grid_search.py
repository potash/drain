import os
import json
import itertools
import sys
import yaml
from drain import util
    
# cartesian product of dict whose values are lists
def dict_product(d):
    items = sorted(d.items())
    if len(items) == 0:
        return [{}]
    keys, values = zip(*items)
    a = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return a

# concatenate a list of dict products
def list_dict_product(l):
    return list(itertools.chain(*[dict_product(d) for d in l]))

def params_dir(basedir, params, method):
    h = util.hash_yaml_dict(params)
    d = os.path.join(basedir, method, h + '/')
    return d

def drake_step(basedir, params, method, inputs=None):
    d = params_dir(basedir, params, method)

    if not os.path.exists(d):
        os.makedirs(d)
    
    dirname = os.path.join(d, 'output/')
    params_file = os.path.join(d, 'params.yaml')

    if not os.path.isfile(params_file):
        with file(params_file, 'w') as f:
            yaml.dump(params, f)

    inputs = ', ' + str.join(', ', inputs) if inputs is not None else ''

    return dirname + ' <- ' + params_file + inputs + ' [method:' + method + ']'
    
with open(sys.argv[1]) as f:
    params = yaml.load(f)

outputdir = sys.argv[2]

runs = []

data = list_dict_product(params['data'])
transforms = list_dict_product(params['transforms'])
models = list_dict_product(params['models'])
metrics = list_dict_product(params['metrics'])

#TODO include a project specific Drakefile via cmd arg
bindir = os.path.abspath(os.path.dirname(sys.argv[0]))
print """
data()
    python {bindir}/read_write_data.py $INPUT $OUTPUT
model()
    python {bindir}/run_model.py $INPUT $OUTPUT $INPUT1
""".format(bindir=bindir)

# data steps
for d in data:
    p = {'data': d}
    print drake_step(outputdir, p, 'data')

# model steps
runs = []
i = 0
for d,t,m in itertools.product(data,transforms,models):
    i = i + 1
    p = {'data': d, 'transform':t, 'model':m, 'metrics':metrics}
    d = {'data': d}
    datadir = os.path.join(params_dir(outputdir, d, 'data'), 'output/') # use data dir for drake dependency

    print drake_step(outputdir, p, 'model', inputs=[datadir])
