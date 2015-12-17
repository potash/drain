import yaml
import pandas as pd
from pandas.io.pytables import PerformanceWarning
from sklearn.externals import joblib
import os
import argparse
import inspect
from copy import deepcopy
import logging

from drain import model
from drain import util

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PerformanceWarning)

from pprint import pformat
def pformat_indent(o):
    return '\t' + pformat(o).replace('\n', '\n\t')

parser = argparse.ArgumentParser(description='Use this script to run a single model.')
parser.add_argument('params', type=str, help='yaml params filename')
parser.add_argument('outputdir', type=str, help='output directory')
args = parser.parse_args()

with open(args.params) as f:
    params = yaml.load(f)

data_name = params['transform'].pop('name')
model_name = params['model'].pop('name')
model_cache = params['model'].pop('cache') if 'cache' in params['model'] else False

logging.info('Reading ' + data_name + ' with parameters:\n' + pformat_indent(params['transform']))

data = util.get_attr(data_name)(**params['transform'])
#data.read(args.datadir)

logging.info('Tranforming')

basedir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.join(args.outputdir)))))
data.run()
train,test = data.cv

logging.info('Training ' + model_name +
    '\n\twith parameters ' + str(params['model']) + 
    '\n\ton ' + str(train.sum()) + ' examples'+
    '\n\twith ' + str(len(data.X.columns)) + ' features')

estimator = util.get_attr(model_name)(**params['model'])

if 'sample_weight' in inspect.getargspec(estimator.fit) and hasattr(data, 'sample_weight'):
    estimator.fit(data.X[train],data.y[train], data.sample_weight[train])
else:
    estimator.fit(data.X[train],data.y[train])

logging.info('Validating model'+
    '\n\ton ' + str(test.sum()) + ' examples')

y_score = pd.Series(model.y_score(estimator, data.X[test]), index=data.X[test].index)

# create a single y dataframe with train and test and other masks
y = pd.DataFrame({'true': data.y[test]})
y['score'] = y_score
if hasattr(data, 'aux'):
    y = y.join(data.aux)
#y['train'] = train
#y['test'] = test

#run = {'data': data, 'estimator': estimator, 'y': y.reset_index()}
run = model.ModelRun(data=data, estimator=estimator, y=y.reset_index())

# print metrics
common_params = {'run': run} # stuff passed to every metric
for metric in params['metrics']:
    metric_name = metric.pop('name')
    metric_fn = util.get_attr(metric_name)

    # get the subset of args that this metric wants
    #kwds = inspect.getargspec(metric_fn).args
    #mp = {k:v for k,v in metric.iteritems() if k in kwds and v is not None}
    mp = {k:v for k,v in metric.iteritems()}
    #cp = {k:v for k,v in common_params.iteritems() if k in kwds}
    #p = dict(mp, **cp)
    p = dict(common_params, **mp)
    v = metric_fn(**p)

#    mp = map(lambda k: str(k) + '=' + str(mp[k]),mp)
    mp = map(lambda k: str(k) + '=' + str(metric[k]),metric)
    print '\t' + metric_fn.__name__ + '(' + str.join(',', mp) + '): ' + str(v)

if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)
    
if model_cache:
    logging.info('Dumping estimator')
    joblib.dump(estimator, os.path.join(args.outputdir, 'estimator.pkl'))

logging.info('Dumping output')
y.to_hdf(os.path.join(args.outputdir, 'y.hdf'), 'y', mode='w')
model.feature_importance(estimator, data.X).to_csv(os.path.join(args.outputdir, 'features.csv'), index=False)

target = os.path.join(os.path.dirname(args.params), 'target')
util.touch(target)
