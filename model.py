import os
import datetime
import math
from copy import deepcopy

import pandas as pd
import numpy as np
from sklearn import metrics
from statsmodels.discrete.discrete_model import Logit

from drain import util

def y_score(estimator, X):
    if hasattr(estimator, 'decision_function'):
        return estimator.decision_function(X)
    else:
        y = estimator.predict_proba(X)
        return y[:,1]

# given a params dict and a basedir and a method, return a directory for storing the method output
# generally this is basedir/method/#/
# where '#' is the hash of the yaml dump of the params dict
# in the special case of method='model', the metrics key is dropped before hashing
def params_dir(basedir, params, method):
    if method == 'model' and 'metrics' in params:
        params = deepcopy(params)
        params.pop('metrics')

    h = util.hash_yaml_dict(params)
    d = os.path.join(basedir, method, h + '/')
    return d

def sk_tree(X,y, params={'max_depth':3}):
    clf = tree.DecisionTreeClassifier(**params)
    return clf.fit(X, y)

def feature_importance(estimator, X):
    if hasattr(estimator, 'coef_'):
        i = estimator.coef_[0]
    elif hasattr(estimator, 'feature_importances_'):
        i = estimator.feature_importances_
    else:
        i = [np.nan]*len(X.columns)

    return pd.DataFrame({'feature': X.columns, 'importance': i}).sort_values('importance', ascending=False)


class LogisticRegression(object):
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        self.model = Logit(y, X)
        self.result = self.model.fit()
    
    def predict_proba(self, X):
        return self.result.predict(X)

from sklearn.externals.joblib import Parallel, delayed
from sklearn.ensemble.forest import _parallel_helper

def _proximity_parallel_helper(train_nodes, t, k):
    d = (train_nodes == t).sum(axis=1)
    n = d.argsort()[::-1][:k]
    
    return d[n], n #distance, neighbors

def _proximity_helper(train_nodes, test_nodes, k):
    results = Parallel(n_jobs=16, backend='threading')(delayed(_proximity_parallel_helper)(train_nodes, t, k) for t in test_nodes)
    distance, neighbors = zip(*results)
    return np.array(distance), np.array(neighbors)

# store nodes in run
def apply_forest(run):
    run['nodes'] = pd.DataFrame(run.estimator.apply(run['data'].X), index=run['data'].X.index)
    
# look for nodes in training set proximal to the given nodes
def proximity(run, ix, k):
    if 'nodes' not in run:
        apply_forest(run)
    distance, neighbors = _proximity_helper(run['nodes'][run.y.train].values, run['nodes'].loc[ix].values, k)
    neighbors = run['nodes'][run.y.train].irow(neighbors.flatten()).index
    neighbors = [neighbors[k*i:k*(i+1)] for i in range(len(ix))]
    return distance, neighbors

# subset a model "y" dataframe
def y_subset(y, masks=[], filters={}, test=True, 
        dropna=False, outcome='true',
        k=None, p=None, ascending=False, score='score'):

    masks2=[]
    for mask in masks:
        masks2.append(util.get_series(y, mask))

    for column, value in filters.iteritems():
        masks2.append(util.get_series(y, column) == value)

    if test:
        masks2.append(y['test'])

    mask = util.intersect(masks2)
    y = y[mask]

    if dropna:
        y = y.dropna(subset=[outcome])

    if k is not None and p is not None:
        raise ValueError("precision: cannot specify both k and p")
    elif k is not None:
        k = k
    elif p is not None:
        k = int(p*len(y))
    else:
        k = None

    if k is not None:
        y = y.sort_values(score, ascending=ascending).head(k)

    return y

def true_score(y, outcome='true', score='score', **subset_args):
    y = y_subset(y, outcome=outcome, score=score, **subset_args) 
    return util.to_float(y[outcome], y[score])
