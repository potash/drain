import numpy as np
import pandas as pd
import sklearn.metrics
from drain import util
from drain.util import to_float

def _top_k(y_score, k=None):
    ranks = y_score.argsort()
    
    if k is None:
        k = len(y_score)
    top_k = ranks[::-1][0:k]

    return top_k

def count(y, dropna=False):
    if dropna:
        return (~np.isnan(to_float(y))).sum()
    else:
        return len(y)

def count_series(y_true, y_score, dropna=False):
    y_true, y_score = to_float(y_true, y_score)
    top_k = _top_k(y_score)

    if dropna:
        a = (~np.isnan(y_true[top_k])).cumsum()
    else:
        a = range(1, len(y)+1)

    return pd.Series(a, index=range(1, len(a)+1))


def baseline(y_true):
    if len(y_true) > 0:
        return np.nansum(y_true)/count(y_true, dropna=True)
    else:
        return 0.0

def auc(y_true, y_score):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    return sklearn.metrics.auc(fpr, tpr)

def top_k(y_true, y_score, k, extrapolate=False):
    if len(y_true) != len(y_score):
        raise ValueError('Labels and scores must have same lengths: %s != %s' 
                 % (len(y_true), len(y_score)))
    if k == 0:
        return (0,0)

    y_true, y_score = to_float(y_true, y_score)

    labeled = ~np.isnan(y_true)
    n = len(y_true) if extrapolate else labeled.sum()
    if not extrapolate and k > n:
        raise ValueError('Cannot calculate precision at %d > %d'% (k,n))

    if extrapolate:
        ranks = y_score.argsort()
        top = ranks[-k:]
        labeled_top = ~np.isnan(y_true[top])

        return y_true[top][labeled_top].sum(), labeled_top.sum()

    else:
        y_true = y_true[labeled]
        y_score = y_score[labeled]
        ranks = y_score.argsort()
        top = ranks[-k:]

        return y_true[top].sum(), k

# when extrapolate is True, return a triple
# first element is lower bound (assuming unlabeled examples are all False)
# second is precision of labeled examples only
# third is upper bound (assuming unlabeled examples are all True) 
def precision_at_k(y_true, y_score, k, extrapolate=False, return_bounds=True):
    n,d = top_k(y_true, y_score, k, extrapolate)
    p = n*1./d if d != 0 else np.nan

    if extrapolate:
        bounds = (n/k, (n+k-d)/k) if k != 0 else (np.nan, np.nan)
        if return_bounds:
            return p, d, bounds
        else:
            return p
    else:
        return p

# TODO extrapolate here
def precision_series(y_true, y_score, k=None):
    y_true, y_score = to_float(y_true, y_score)
    top_k = _top_k(y_score, k)

    n = np.nan_to_num(y_true[top_k]).cumsum() # fill missing labels with 0
    d = (~np.isnan(y_true[top_k])).cumsum()     # count number of labelsa
    return pd.Series(n/d, index=np.arange(1,len(n)+1))

def recall(y_true, y_score, k=None, value=True):
    y_true, y_score = to_float(y_true, y_score)
    top_k = _top_k(y_score, k)

    if not value:
        y_true = 1-y_true

    return np.nan_to_num(y_true[top_k]).sum()

def recall_series(y_true, y_score, k=None, value=True):
    y_true, y_score = to_float(y_true, y_score)
    top_k = _top_k(y_score, k)

    if not value:
        y_true = 1-y_true

    a = np.nan_to_num(y_true[top_k]).cumsum()
    return pd.Series(a, index=np.arange(1,len(a)+1))
