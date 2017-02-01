import logging
import os
from . import serialize

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=0)

serialize.configure()

if 'DRAINPATH' in os.environ:
    PATH = os.path.abspath(os.environ['DRAINPATH'])
else:
    PATH = None
