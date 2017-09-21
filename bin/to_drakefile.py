import os
import sys
import argparse
import importlib
import logging
import shutil
from six.moves import input

from drain import step, util, drake, serialize
import drain

workflows_help = "Each workflow is either: the name of a method returning either a drain Step object or collection thereof; or the path to a YAML serialization of a step. To specify multiple workflows, separate with semi-colons."

def parse_workflows(workflows):
    steps = []
    for s in workflows.split(';'):
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
    return steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')

    parser_exec = subparsers.add_parser('execute', help='Execute the specified workflows.')
    #internally used temp file for drake workflow
    parser_exec.add_argument('--drakeoutput', type=str, 
                             default=os.environ.get('DRAKE_FILE', None),
                             help=argparse.SUPPRESS)
    # internally used temp file for drake arguments
    parser_exec.add_argument('--drakeargsfile', type=str, 
                             default=os.environ.get('DRAKE_ARGS_FILE', None),
                             help=argparse.SUPPRESS)

    parser_exec.add_argument('--drakefile', type=str, default=None, help='Specify a dependent drake workflow file (defaults to ./Drakefile, if present).')
    parser_exec.add_argument('--ignore-drakefile', action='store_true', help='Ignore ./Drakefile')
    parser_exec.add_argument('--debug', action='store_true', help='Execute steps with Python debugger.')
    parser_exec.add_argument('--preview', action='store_true', help='Print the drake workflow that would run, then stops.')
    parser_exec.add_argument('--path', type=str, help='Output base directory. If not specified, use $DRAINPATH environment variable.')
    parser_exec.add_argument('workflows', type=str, help=workflows_help)
   
    parser_keep = subparsers.add_parser('keep', help='Delete all outputs except those which are targets of the specified workflows.')
    parser_keep.add_argument('--path', type=str, help='Output base directory. If not specified, use $DRAINPATH environment variable.')
    parser_keep.add_argument('workflows', type=str, help=workflows_help)

    args, drake_args = parser.parse_known_args()
    if args.path:
        drain.PATH = os.path.abspath(args.path)
    elif drain.PATH is None:
        raise ValueError('Must pass path argument or set DRAINPATH environment variable')

    steps = parse_workflows(args.workflows)

    if args.command == 'execute':
        if args.drakefile is None and not args.ignore_drakefile and os.path.exists('Drakefile'):
            args.drakefile = 'Drakefile'
        drakefile = os.path.abspath(args.drakefile) if args.drakefile else None

        workflow = drake.to_drakefile(steps, 
                                      preview=args.preview, 
                                      debug=args.debug, 
                                      input_drakefile=drakefile,
                                      bindir=os.path.dirname(__file__))

        if args.preview:
            sys.stdout.write(workflow)
        else:
            with open(args.drakeoutput, 'w') as drakefile:
                logging.info('Writing drake workflow %s' % args.drakeoutput)
                drakefile.write(workflow)

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

    elif args.command == 'keep':
        rm_dirs = step._output_dirnames().difference(
                step._output_dirnames(steps))
        if len(rm_dirs) == 0:
            print("No outputs to remove.")
            quit()

        print('The following outputs will be REMOVED:')
        for d in rm_dirs:
            print("  " + d)
        response = input("Do you want to continue? [Y/n] ")
        if response.upper() == 'Y':
            for d in rm_dirs:
                shutil.rmtree(d)
        else:
            print("Abort.")
            exit()
