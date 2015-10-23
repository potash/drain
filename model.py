import os
import datetime
import math

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

    return pd.DataFrame({'feature': X.columns, 'importance': i}).sort('importance', ascending=False)


class LogisticRegression(object):
    def __init__(self):
        pass

    def fit(self, X, y, **kwargs):
        self.model = Logit(y, X)
        self.result = self.model.fit()
    
    def predict_proba(self, X):
        return self.result.predict(X)
