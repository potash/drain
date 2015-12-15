import os
import json
import itertools
import sys
import yaml
import argparse
import shutil

from drain import util
from drain.model import params_dir
    
# cartesian product of dict whose values are lists
def dict_product(d):
    items = sorted(d.items())
    if len(items) == 0:
        return [{}]
    keys, values = zip(*items)
    a = [dict_filter_none(dict(zip(keys, v))) for v in itertools.product(*values)]
    return a

# filter none values from dict
def dict_filter_none(d):
    return {k:v for k,v in d.items() if v is not None}

# concatenate a list of dict products
def list_dict_product(l):
    return list(itertools.chain(*[dict_product(d) for d in l]))

def check_tag(tag):
    if tag.find('.') > -1:
        raise argparse.ArgumentTypeError('%s is an invalid tag name' % tag)
    else:
        return tag

# return True if params_file does not exist
# or if params_file content is different from params
# that is possible if the params_file was corrupted
# or if metrics changed (because those don't count for hash)
def params_new(params, params_file):
    if not os.path.isfile(params_file):
        return True
    else:
        with open(params_file) as params2:
            if yaml.load(params2) != params:
                return True
    return False

def drake_step(basedir, params, method, inputs=[], tagdir=None, preview=False):
    d = params_dir(basedir, params, method)

    if not os.path.exists(d) and not preview:
        os.makedirs(d)
    
    #dirname = os.path.join(d, 'output/')
    target = os.path.join(d, 'target')
    params_file = os.path.join(d, 'params.yaml')

    if params_new(params, params_file) and not preview:
        with open(params_file, 'w') as f:
            yaml.dump(params, f)

    if tagdir is not None and not os.path.exists(tagdir) and not preview:
        os.symlink(d, tagdir)

    cls = util.get_attr(params[method]['name'])
    if hasattr(cls, 'DEPENDENCIES'):
        inputs = inputs + cls.DEPENDENCIES

    inputs = ', !' + str.join(', !', inputs) if len(inputs) > 0 else ''

    return '!'+target + ' <- ' + '!'+params_file + inputs + ' [method:' + method + ']\n\n'

# write the grid search drakefile to drakefile
# drakein is the optional dependent drakefile
def grid_search(params, outputdir, drakefile, drakein=None, tag=None, python_args=None, overwrite_tag=False, preview=False):
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
PYTHONUNBUFFERED=Y\n
data()
    python {python_args} {bindir}/read_write_data.py $INPUT $(dirname $OUTPUT)/output/
model()
    python {python_args} {bindir}/run_model.py $INPUT $(dirname $OUTPUT)/output/ $(dirname $INPUT1)/output/ \n
""".format(bindir=bindir, python_args=python_args))
    
    # data steps
    for d in data:
        p = {'data': d}
        drakefile.write(drake_step(outputdir, p, 'data', preview=preview))
    
    if tag is not None:
        tagdir = os.path.join(outputdir, 'tag', tag)
        if overwrite_tag and os.path.exists(tagdir) and not preview:
            shutil.rmtree(tagdir)
        if not os.path.exists(tagdir) and not preview:
            os.makedirs(tagdir)
            

    # model steps
    i = 0
    #for d,t,m in itertools.product(data,transforms,models):
    for t,m in itertools.product(transforms,models):
        i = i + 1
        #p = {'data': d, 'transform':t, 'model':m, 'metrics':metrics}
        p = {'transform':t, 'model':m, 'metrics':metrics}
        #d = {'data': d}
        #datadir = os.path.join(params_dir(outputdir, d, 'data'), 'target') # use data dir for drake dependency
        tagdir = os.path.join(outputdir, 'tag', tag, util.hash_yaml(p)) if tag is not None else None
    
        #drakefile.write(drake_step(outputdir, p, 'model', inputs=[datadir], tagdir=tagdir, preview=preview))
        drakefile.write(drake_step(outputdir, p, 'model', inputs=[], tagdir=tagdir, preview=preview))

def powerset_constructor(loader, node):
    args = loader.construct_sequence(node)
    return list(itertools.chain.from_iterable(itertools.combinations(args, r) for r in range(len(args)+1)))

def range_constructor(loader, node):
    args = loader.construct_sequence(node)
    return range(*args)

def yaml_init():
    yaml.add_constructor('!powerset', powerset_constructor)
    yaml.add_constructor('!range', range_constructor)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use this script to generate a Drakefile for grid search')
    
    parser.add_argument('--drakeoutput', type=str, help='internally used temp file for drake workflow')
    parser.add_argument('--drakeargsfile', type=str, help='internally used temp file for drake arguments')
    parser.add_argument('-D', '--Drakeinput', type=str, default=None, help='dependent drakefile')
    parser.add_argument('-t', '--tag', default=None, help='tag name for model outputs', type=check_tag)
    parser.add_argument('-o', '--overwrite', action='store_true', help='overwrite tag')
    parser.add_argument('-d', '--debug', action='store_true', help='run python -m pdb')
    parser.add_argument('-P', '--preview', action='store_true', help='Preview Drakefile and pass --preview to drake')
    
    parser.add_argument('params', type=str, help='yaml params filename')
    parser.add_argument('outputdir', type=str, help='output directory')
    #parser.add_argument('drakeargs', nargs='?', type=str, default=None, help='parameters to pass to drake via --drakeargsfile')
    args, drake_args = parser.parse_known_args()
    
    yaml_init()
    with open(args.params) as f:
        params = yaml.load(f)
    
    outputdir = os.path.abspath(args.outputdir)
    
    if args.Drakeinput is None and os.path.exists('Drakefile'):
        args.Drakeinput = 'Drakefile'

    pyargs = '-m pdb' if args.debug else ''
    
    with open(args.drakeoutput, 'w') as drakefile:
        if args.preview:
            drakefile = sys.stdout
        grid_search(params, outputdir, drakefile, args.Drakeinput, args.tag, 
                python_args=pyargs, overwrite_tag=args.overwrite, preview=args.preview)
   
    if drake_args is not None and args.drakeargsfile is not None and not args.preview:
        with open(args.drakeargsfile, 'w') as drakeargsfile:
            drakeargsfile.write(str.join(' ', drake_args))
