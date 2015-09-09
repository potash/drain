import yaml
import os
import argparse
from copy import deepcopy
from drain import util

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(description='Use this script to read and write ModelData for caching.')
parser.add_argument('input', type=str, help='yaml filename')
parser.add_argument('basedir', type=str, help='directory to cache in')
args = parser.parse_args()

with open(args.input) as f:
    params_orig = yaml.load(f)

params = deepcopy(params_orig)
data_name = params['data'].pop('name')

print 'Loading ' + data_name
print '    with parameters ' + str(params['data'])

data = util.get_class(data_name)(**params['data'])
data.read()

data.write(args.basedir)
