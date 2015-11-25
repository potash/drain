import numpy as np
import pandas as pd
import sklearn.metrics
from drain import util

# cast numpy arrays to float32
# if there's more than one, return an array
def to_float(*args):
    floats = [np.array(a, dtype=np.float32) for a in args]
    return floats[0] if len(floats) == 1 else floats

def count_notnull(series):
    return (~np.isnan(to_float(series))).sum()

def baseline(run, masks=[], test=True, outcome='true'):
    y_true,y_score = _mask(run, masks, test, outcome)
    y_true,y_score = to_float(y_true, y_score)

    if len(y_true) > 0:
        return np.nansum(y_true)/count_notnull(y_true)
    else:
        return 0.0

# return size of dataset
# if dropna=True, only count rows where outcome is not nan
def count(run, masks=[], test=True, outcome='true', dropna=False):
    y_true,y_score = _mask(run, masks, test, outcome)
    if dropna:
        return count_notnull(y_true)
    else:
        return len(y_true)

def auc(run, masks=[], test=True, outcome='true'):
    y_true, y_score = _mask(run, masks, test, outcome)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    return sklearn.metrics.auc(fpr, tpr)

def precision(run, k=None, p=None, masks=[], test=True, outcome='true', extrapolate=False):
    y_true, y_score = _mask(run, masks, test, outcome)

    # deal with k or p
    if k is not None and p is not None:
        raise ValueError("precision: cannot specify both k and p")
    elif k is not None:
        k = k
    elif p is not None:
        k = int(p*len(y_true))
    else:
        raise ValueError("precision must specify either k or p")

    k = min(k, len(y_true) if extrapolate else count_notnull(y_true) )

    return precision_at_k(y_true, y_score, k, extrapolate)

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
def precision_at_k(y_true, y_score, k, extrapolate=False):
    n,d = top_k(y_true, y_score, k, extrapolate)
    p = n*1./d if d != 0 else np.nan

    if extrapolate:
        bounds = (n/k, (n+k-d)/k) if k != 0 else (np.nan, np.nan)
        return d, p, bounds
    else:
        return p

# TODO extrapolate here
def precision_series(y_true, y_score, k=None):
    y_true, y_score = to_float(y_true, y_score)
    ranks = y_score.argsort()

    if k is None:
        k = len(y_true)

    top_k = ranks[::-1][0:k]
    return pd.Series(y_true[top_k].cumsum()*1.0/np.arange(1,k+1), index=np.arange(1,k+1))

def _mask(run, masks, test, outcome='true'):
    masks2 = []
    d = isinstance(masks, dict)
    for mask in masks:
        series = util.get_series(run['y'], mask)
        if d:
            series = series == masks[mask]
        masks2.append(series)

    if test:
        masks2.append(run['y']['test'])

    mask = reduce(lambda a,b: a & b, masks2)
    y_true = run['y'][mask][outcome]
    y_score = run['y'][mask]['score']

    return y_true, y_score
