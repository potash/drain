import yaml
import pandas as pd
from sklearn.externals import joblib
import os
import argparse
import inspect
from copy import deepcopy

from drain import model
from drain import util

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(description='Use this script to run a single model.')
parser.add_argument('params', type=str, help='yaml params filename')
parser.add_argument('outputdir', type=str, help='output directory')
parser.add_argument('datadir', type=str, default=None, help='datadir filename')
args = parser.parse_args()

with open(args.params) as f:
    params = yaml.load(f)

data_name = params['data'].pop('name')
model_name = params['model'].pop('name')

print 'Loading ' + data_name
print '    with parameters ' + str(params['data'])

data = util.get_attr(data_name)(**params['data'])

data.read(args.datadir)

print 'Tranforming with parameters ' + str(params['transform'])
data.transform(**params['transform'])

train,test = data.cv

print 'Training ' + model_name
print '    with parameters ' + str(params['model'])
print '    on ' + str(train.sum()) + ' examples'
print '    with ' + str(len(data.X.columns)) + ' features'

estimator = util.get_attr(model_name)(**params['model'])

if 'sample_weight' in inspect.getargspec(estimator.fit) and hasattr(data, 'sample_weight'):
    estimator.fit(data.X[train],data.y[train], data.sample_weight[train])
else:
    estimator.fit(data.X[train],data.y[train])

print 'Validating model'
print '    on ' + str(test.sum()) + ' examples'

y_score = pd.Series(model.y_score(estimator, data.X), index=data.X.index)

# print metrics
common_params = {'data':data, 'estimator':estimator, 'y_score':y_score[test], 'y_true':data.y[test]} # stuff passed to every metric
for metric in params['metrics']:
    metric_name = metric.pop('name')
    metric_fn = util.get_attr(metric_name)

    # get the subset of args that this metric wants
    kwds = inspect.getargspec(metric_fn).args
    mp = {k:v for k,v in metric.iteritems() if k in kwds and v is not None}
    cp = {k:v for k,v in common_params.iteritems() if k in kwds}
    p = dict(mp, **cp)
    v = metric_fn(**dict(mp, **cp))

    mp = map(lambda k: str(k) + '=' + str(mp[k]),mp)
    print '    ' + metric_fn.__name__ + '(' + str.join(',', mp) + '): ' + str(v)

if not os.path.exists(args.outputdir):
    os.makedirs(args.outputdir)
        
joblib.dump(estimator, os.path.join(args.outputdir, 'estimator.pkl'))

# write output
y = pd.DataFrame({'score':y_score, 'true': data.y})
y.to_csv(os.path.join(args.outputdir, 'y.csv'), index=True)

train.to_csv(os.path.join(args.outputdir, 'train.csv'), index=True)
test.to_csv(os.path.join(args.outputdir, 'test.csv'), index=True)
pd.DataFrame(columns=data.X.columns).to_csv(os.path.join(args.outputdir, 'columns.csv'),index=False)
