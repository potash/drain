import sys
import dateutil
import logging
import pandas as pd

from drain import util

aggregator_name = sys.argv[1]
date = pd.to_datetime(dateutil.parser.parse(sys.argv[2]))
basedir = sys.argv[3]

logging.info('Initializing %s' % aggregator_name)
aggregator = util.get_attr(aggregator_name)(basedir)
aggregator.write_date(date)
