import os
import sys
import argparse
import importlib
import logging

from drain import step, util, drake, serialize
import drain

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Use this script to generate a Drakefile for grid search')
    #internally used temp file for drake workflow
    parser.add_argument('--drakeoutput', type=str, help=argparse.SUPPRESS)
    # internally used temp file for drake arguments
    parser.add_argument('--drakeargsfile', type=str, help=argparse.SUPPRESS)

    parser.add_argument('--drakefile', type=str, default=None, help='Specifies a dependent drake workflow file (defaults to ./Drakefile, if present).')
    parser.add_argument('--ignore-drakefile', action='store_true', help='Ignore ./Drakefile')
    parser.add_argument('--debug', action='store_true', help='Executes steps with Python debugger.')
    parser.add_argument('--preview', action='store_true', help='Prints the drake workflow that would run, then stops.')
    parser.add_argument('--path', type=str, help='output base directory')
    
    parser.add_argument('steps', type=str, help='The steps to run. The name of a method returning either a drain Step object or collection thereof. Alternatively, the path to a YAML serialization of a step. To specify multiple paths, separate with semi-colons.')

    args, drake_args = parser.parse_known_args()

    if args.drakeoutput is None or args.drakeargsfile is None:
        args.preview = True

    if args.path:
        drain.PATH = os.path.abspath(args.path)
    elif drain.PATH is None:
        raise ValueError('Must pass path argument or set DRAINPATH environment variable')

    steps = []
    for s in args.steps.split(';'):
        if s.endswith('.yaml'):
            steps += serialize.load(s)
        else:
            modulename, fname = s.split('::')
            mod = importlib.import_module(modulename)
            s_attr = getattr(mod, fname)
            # if s is callable, it should return a collection of Steps
            # otherwise assume it is a collection of Steps
            ss = util.make_list(s_attr() if hasattr(s_attr, '__call__') else s_attr)
            logging.info('Loaded %s with %s leaf steps' % (s, len(ss)))
            steps += ss

    if args.drakefile is None and not args.ignore_drakefile and os.path.exists('Drakefile'):
        args.drakefile = 'Drakefile'
    drakefile = os.path.abspath(args.drakefile) if args.drakefile else None

    workflow = drake.to_drakefile(steps, 
                                  preview=args.preview, 
                                  debug=args.debug, 
                                  input_drakefile=drakefile,
                                  bindir=os.path.dirname(__file__))

    if not args.preview:
        with open(args.drakeoutput, 'w') as drakefile:
            logging.info('Writing drake workflow %s' % args.drakeoutput)
            drakefile.write(workflow)
    else:
        sys.stdout.write(workflow)

    drake_args = list(drake_args) if drake_args is not None else []
    # need PYTHONUNBUFFERED for pdb interactivity
    if args.debug:
        drake_args.insert(0, '-v PYTHONUNBUFFERED=Y')

    # set basedir for drakefile to get around issue in comments of:
    # https://github.com/Factual/drake/pull/211
    if args.drakefile is not None:
        drake_args.insert(0, '--base=%s' % os.path.dirname(args.drakefile))
   
    if args.drakeargsfile is not None and not args.preview:
        with open(args.drakeargsfile, 'w') as args:
            args.write(str.join(' ', drake_args))
