import sys
import os
import inspect
from drain import util

aggregator_name = sys.argv[1]
params = sys.argv[2:]

aggregator_cls = util.get_attr(aggregator_name)
aggregator = aggregator_cls(*params)
aggregate = os.path.join(os.path.dirname(sys.argv[0]), 'aggregate.py') # path to aggregate.py executable

for date in aggregator.dates:
    inputs = inspect.getsourcefile(aggregator_cls)
    if hasattr(aggregator, 'DEPENDENCIES'):
        inputs = str.join(', ', [inputs] + aggregator.DEPENDENCIES)
    print '{} <- {}'.format(aggregator.filenames[date], inputs)
    print '    python {} {} {} {}'.format(aggregate, aggregator_name, date, str.join(' ', params))
