import os
import yaml

import json
import itertools
import sys
import argparse
import imp

from drain import step

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use this script to generate a Drakefile for grid search')
    
    parser.add_argument('--drakeoutput', type=str, help='internally used temp file for drake workflow')
    parser.add_argument('--drakeargsfile', type=str, help='internally used temp file for drake arguments')
    parser.add_argument('-D', '--Drakeinput', type=str, default=None, help='dependent drakefile')
    parser.add_argument('-d', '--debug', action='store_true', help='run python -m pdb')
    parser.add_argument('-P', '--preview', action='store_true', help='Preview Drakefile')
    parser.add_argument('-n', '--name', help='Name to store this workflow under')
    parser.add_argument('--basedir', type=str, help='output base directory')
    
    parser.add_argument('steps', type=str, help='yaml params filename')

    #parser.add_argument('drakeargs', nargs='?', type=str, default=None, help='parameters to pass to drake via --drakeargsfile')
    args, drake_args = parser.parse_known_args()

    if args.drakeoutput is None or args.drakeargsfile is None:
        args.preview = True

    step.BASEDIR = os.path.abspath(args.basedir)
    step.configure_yaml()

    if args.steps.endswith('.yaml'):
        steps = step.from_yaml(args.steps)
    else:
        filename, fname = args.steps.split('::')
        mod = imp.load_source('steps', filename)
        fn = getattr(mod, fname)
        steps = fn()

    if args.Drakeinput is None and os.path.exists('Drakefile'):
        args.Drakeinput = 'Drakefile'

    workflow = step.to_drakefile(steps, preview=args.preview, debug=args.debug)

    if not args.preview:
        with open(args.drakeoutput, 'w') as drakefile:
            drakefile.write(workflow)
    else:
        sys.stdout.write(workflow)

    # need PYTHONUNBUFFERED for pdb interactivity
    if args.debug:
        drake_args = list(drake_args) if drake_args is not None else []
        drake_args.insert(0, '-v PYTHONUNBUFFERED=Y')
   
    if args.drakeargsfile is not None and not args.preview:
        with open(args.drakeargsfile, 'w') as drakeargsfile:
            drakeargsfile.write(str.join(' ', drake_args))

    if args.name is not None and not args.preview:
        steps_dirname = os.path.join(args.basedir, '.steps')
        if not os.path.isdir(steps_dirname):
            os.makedirs(steps_dirname)

        with open(os.path.join(steps_dirname, '%s.yaml' % args.name), 'w') as steps_file:
            step.configure_yaml(dump_all_args=True)
            yaml.dump(steps, steps_file)
