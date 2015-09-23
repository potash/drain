import os
import json
import itertools
import sys
import yaml
import argparse
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

def drake_step(basedir, params, method, inputs=None, tagdir=None):
    d = params_dir(basedir, params, method)

    if not os.path.exists(d):
        os.makedirs(d)
    
    dirname = os.path.join(d, 'output/')
    params_file = os.path.join(d, 'params.yaml')

    if not os.path.isfile(params_file):
        with file(params_file, 'w') as f:
            yaml.dump(params, f)

    if tagdir is not None and not os.path.exists(tagdir):
        os.symlink(d, tagdir)

    inputs = ', !' + str.join(', !', inputs) if inputs is not None else ''

    return '!'+dirname + ' <- ' + '!'+params_file + inputs + ' [method:' + method + ']\n\n'

# write the grid search drakefile to drakefile
# drakein is the optional dependent drakefile
def grid_search(params, outputdir, drakefile, drakein=None, tag=None):
    data = list_dict_product(params['data'])
    transforms = list_dict_product(params['transforms'])
    models = list_dict_product(params['models'])
    metrics = list_dict_product(params['metrics'])
    
    if drakein is not None:
        dirname, basename = os.path.split(os.path.abspath(drakein))
        drakefile.write("BASE={}\n".format(dirname))
        drakefile.write("%include $[BASE]/{}\n".format(basename))
    
    #TODO include a project specific Drakefile via cmd arg
    bindir = os.path.abspath(os.path.dirname(sys.argv[0]))
    drakefile.write("""
PYTHONUNBUFFERED=Y
data()
    python {bindir}/read_write_data.py $INPUT $OUTPUT
model()
    python {bindir}/run_model.py $INPUT $OUTPUT $INPUT1\n
""".format(bindir=bindir))
    
    # data steps
    for d in data:
        p = {'data': d}
        drakefile.write(drake_step(outputdir, p, 'data'))
    
    if tag is not None:
        tagdir = os.path.join(outputdir, 'tag', tag)
        if not os.path.exists(tagdir):
            os.makedirs(tagdir)

    # model steps
    i = 0
    for d,t,m in itertools.product(data,transforms,models):
        i = i + 1
        p = {'data': d, 'transform':t, 'model':m, 'metrics':metrics}
        d = {'data': d}
        datadir = os.path.join(params_dir(outputdir, d, 'data'), 'output/') # use data dir for drake dependency
        tagdir = os.path.join(outputdir, 'tag', tag, util.hash_yaml_dict(d)) if tag is not None else None
    
        drakefile.write(drake_step(outputdir, p, 'model', inputs=[datadir], tagdir=tagdir))

parser = argparse.ArgumentParser(description='Use this script to generate a Drakefile for grid search')
parser.add_argument('drakeoutput', type=str, help='output drakefile')
parser.add_argument('params', type=str, help='yaml params filename')
parser.add_argument('outputdir', type=str, help='output directory')
parser.add_argument('--Drakeinput', type=str, default=None, help='dependent drakefile')
parser.add_argument('--drakeargs', type=str, default=None, help='parameters to pass to drake (via stdout)')
parser.add_argument('--tag', type=str, default=None, help='tag name for model outputs')
args = parser.parse_args()

with open(args.params) as f:
    params = yaml.load(f)

outputdir = os.path.abspath(args.outputdir)

with open(args.drakeoutput, 'w') as drakefile:
    grid_search(params, outputdir, drakefile, args.Drakeinput, args.tag)

if args.drakeargs is not None:
    print args.drakeargs
