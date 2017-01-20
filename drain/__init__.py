import logging
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level=0)

from . import serialize
serialize.configure()

PATH = None
