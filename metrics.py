import numpy as np
import pandas as pd
import sklearn.metrics
import util
from util import to_float

def _argsort(y_score, k=None):
    ranks = y_score.argsort()
    if k is None:
        k = len(y_score)

    argsort = ranks[::-1][0:k]
    return argsort

# avoid sorting when just want the top all
def _argtop(y_score, k=None):
    if k is None:
        return slice(0, len(y_score))
    else:
        return _argsort(y_score, k)

def count(y, countna=False):
    if not countna:
        return (~np.isnan(to_float(y))).sum()
    else:
        return len(y)

def count_series(y_true, y_score, countna=False):
    y_true, y_score = to_float(y_true, y_score)
    top = _argtop(y_score)

    if not countna:
        a = (~np.isnan(y_true[top])).cumsum()
    else:
        a = range(1, len(y)+1)

    return pd.Series(a, index=range(1, len(a)+1))


def baseline(y_true):
    if len(y_true) > 0:
        return np.nansum(y_true)/count(y_true, countna=False)
    else:
        return 0.0

def auc(y_true, y_score):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    return sklearn.metrics.auc(fpr, tpr)

# return_bounds for missing labels:
# first element is lower bound (assuming unlabeled examples are all False)
# second is precision of labeled examples only
# third is upper bound (assuming unlabeled examples are all True) 
def precision(y_true, y_score, k=None, return_bounds=False):
    y_true, y_score = to_float(y_true, y_score)
    top = _argtop(y_score, k)

    n = np.nan_to_num(y_true[top]).sum() # fill missing labels with 0
    d = (~np.isnan(y_true[top])).sum()     # count number of labelsa
    p = n/d

    if return_bounds:
        k = len(y_true) if k is None else k
        bounds = (n/k, (n+k-d)/k) if k != 0 else (np.nan, np.nan)
        return p, d, bounds
    else:
        return p

# TODO extrapolate here
def precision_series(y_true, y_score, k=None):
    y_true, y_score = to_float(y_true, y_score)
    top = _argsort(y_score, k)

    n = np.nan_to_num(y_true[top]).cumsum() # fill missing labels with 0
    d = (~np.isnan(y_true[top])).cumsum()     # count number of labelsa
    return pd.Series(n/d, index=np.arange(1,len(n)+1))

def recall(y_true, y_score, k=None, value=True):
    y_true, y_score = to_float(y_true, y_score)
    top = _argtop(y_score, k)

    if not value:
        y_true = 1-y_true

    return np.nan_to_num(y_true[top]).sum()

def recall_series(y_true, y_score, k=None, value=True):
    y_true, y_score = to_float(y_true, y_score)
    top = _argsort(y_score, k)

    if not value:
        y_true = 1-y_true

    a = np.nan_to_num(y_true[top]).cumsum()
    return pd.Series(a, index=np.arange(1,len(a)+1))
