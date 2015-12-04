import yaml
import os
import argparse
from copy import deepcopy
from drain import util
import logging

import warnings
from pandas.io.pytables import PerformanceWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)

parser = argparse.ArgumentParser(description='Use this script to read and write ModelData for caching.')
parser.add_argument('input', type=str, help='yaml filename')
parser.add_argument('basedir', type=str, help='directory to cache in')
args = parser.parse_args()

with open(args.input) as f:
    params_orig = yaml.load(f)

params = deepcopy(params_orig)
data_name = params['data'].pop('name')

logging.info('Loading ' + data_name + ' with parameters:\n\t' + str(params['data']))

data = util.get_attr(data_name)(**params['data'])
data.read()

if not os.path.exists(args.basedir):
    os.makedirs(args.basedir)

logging.info('Writing ' + data_name)
data.write(args.basedir)
logging.info(data_name + ' written.')
