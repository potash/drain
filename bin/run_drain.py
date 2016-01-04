import os
import yaml

import json
import itertools
import sys
import argparse

from drain import drain

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use this script to generate a Drakefile for grid search')
    
    parser.add_argument('--drakeoutput', type=str, help='internally used temp file for drake workflow')
    parser.add_argument('--drakeargsfile', type=str, help='internally used temp file for drake arguments')
    parser.add_argument('-D', '--Drakeinput', type=str, default=None, help='dependent drakefile')
    parser.add_argument('-d', '--debug', action='store_true', help='run python -m pdb')
    parser.add_argument('-P', '--preview', action='store_true', help='Preview Drakefile')
    
    parser.add_argument('steps', type=str, help='yaml params filename')
    parser.add_argument('outputdir', type=str, help='output base directory')

    #parser.add_argument('drakeargs', nargs='?', type=str, default=None, help='parameters to pass to drake via --drakeargsfile')
    args, drake_args = parser.parse_known_args()
    outputdir = os.path.abspath(args.outputdir)
    
    with open(args.steps) as f:
        drain.initialize(outputdir)
        templates = yaml.load(f)
        steps = [t.construct() for t in templates]

    if args.Drakeinput is None and os.path.exists('Drakefile'):
        args.Drakeinput = 'Drakefile'

    pyargs = '-m pdb' if args.debug else '' # TODO use this
    
    workflow = drain.to_drakefile(steps, preview=args.preview)

    if not args.preview:
        with open(args.drakeoutput, 'w') as drakefile:
            drakefile.write(workflow)
    else:
        sys.stdout.write(workflow)
   
    if drake_args is not None and args.drakeargsfile is not None and not args.preview:
        with open(args.drakeargsfile, 'w') as drakeargsfile:
            drakeargsfile.write(str.join(' ', drake_args))
