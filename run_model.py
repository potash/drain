#!/usr/bin/python

import yaml
import pandas as pd
from sklearn.externals import joblib
import os
import argparse
import inspect
from copy import deepcopy

import model
import util

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

parser = argparse.ArgumentParser(description='Use this script to run a single model.')
parser.add_argument('--file', action='store_true')
parser.add_argument('input', type=str, help='filename or yaml string')
args = parser.parse_args()

if args.file:
    with open(args.input) as f:
        params_orig = yaml.load(f)
else:
    params_orig =  yaml.load(args.input.replace('\\n', '\n'))

params = deepcopy(params_orig)
data_name = params['data'].pop('name')
model_name = params['model'].pop('name')

print 'Loading ' + data_name
print '    with parameters ' + str(params['data'])

data = util.get_class(data_name)(**params['data'])

data.read()

print 'Tranforming with parameter ' + str(params['transform'])
data.transform(**params['transform'])

train,test = data.cv

print 'Training ' + model_name
print '    with parameters ' + str(params['model'])
print '    on ' + str(train.sum()) + ' examples'
print '    with ' + str(len(data.X.columns)) + ' features'

estimator = util.get_class(model_name)(**params['model'])

if 'sample_weight' in inspect.getargspec(estimator.fit) and hasattr(data, 'sample_weight'):
    estimator.fit(data.X[train],data.y[train], data.sample_weight[train])
else:
    estimator.fit(data.X[train],data.y[train])

print 'Validating model'
print '    on ' + str(test.sum()) + ' examples'

y_score = pd.Series(model.y_score(estimator, data.X), index=data.X.index)

p = [.005,.01,.02,.05]
precisions = model.precision(data.y[test], y_score[test], p)

print '    baseline: ' + str(model.baseline(data.y[test]))
print '    precision: ' + str(', '.join('%s=%.2f' % t for t in zip(p, precisions)))
print '    auc: ' + str(model.auc(data.y[test], y_score[test]))

if 'output' in params:
    print 'Writing results in ' + params['output']
    if not os.path.exists(params['output']):
        os.makedirs(params['output'])
        
    with open( os.path.join(params['output'], 'params.yaml'), 'w') as outfile:
        yaml.dump(params_orig, outfile)
    joblib.dump(estimator, os.path.join(params['output'], 'estimator.pkl'))
    y = pd.DataFrame({'score':y_score, 'true': data.y})
    y.to_csv(os.path.join(params['output'], 'y.csv'), index=True)
    train.to_csv(os.path.join(params['output'], 'train.csv'), index=True)
    test.to_csv(os.path.join(params['output'], 'test.csv'), index=True)
    pd.DataFrame(columns=data.X.columns).to_csv(os.path.join(params['output'], 'columns.csv'),index=False)
