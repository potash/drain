import os
import sys
import argparse
import logging
import signal

from drain import step, util, drake, serialize
import drain

workflows_help = "Each workflow is either: the name of a method returning either a drain Step object or collection thereof; or the path to a YAML serialization of a step."

def parse_workflows(workflows):
    steps = []
    for s in workflows:
        if s.endswith('.yaml'):
            steps += serialize.load(s)
        else:
            modulename = s[:s.split('(')[0].rfind('.')]
            exec('import %s' % modulename)
            if s.find('(') < 0:
                s += '()'
            ss = util.make_list(eval('%s' % s))
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
    parser_exec.add_argument('-w', '--workflow', action='append', help=workflows_help, required=True)
   
    parser_list = subparsers.add_parser('list', help='Print step directories. Pipe into rm (for cleanup), du (for disk usage), etc.')
    parser_list.add_argument('--path', type=str, help='Output base directory. If not specified, use $DRAINPATH environment variable.')
    parser_list.add_argument('--complete', action='store_true', help='Only include steps missing that have not been completed.')
    parser_list.add_argument('--invert', action='store_true', help='Print the inverse (complement) of the specified workflows.')
    parser_list.add_argument('-w', '--workflow', action='append', help=workflows_help, required=False)
    parser_list.add_argument('--leaf', action='store_true', help='With --workflow, only include leaves.')

    args, drake_args = parser.parse_known_args()
    if args.path:
        drain.PATH = os.path.abspath(args.path)
    elif drain.PATH is None:
        raise ValueError('Must pass path argument or set DRAINPATH environment variable')

    if args.command == 'execute':
        steps = parse_workflows(args.workflow)
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

    elif args.command == 'list':
        if args.workflow is not None:
            steps = parse_workflows(args.workflow)
            dirs = step._output_dirnames(steps, leaf=args.leaf)
        else:
            dirs = step._output_dirnames()

        if args.complete:
            dirs = [d for d in dirs if os.path.exists(os.path.join(d, 'target'))]
            
        if args.invert:
            dirs = step._output_dirnames().difference(dirs)
        
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)
        for d in dirs:
            print(d)
