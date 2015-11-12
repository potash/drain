import numpy as np
import pandas as pd
import sklearn.metrics
from drain import util

def baseline(run, masks=[], test=True, outcome='true'):
    y_true,y_score = _mask(run, masks, test, outcome)
    if len(y_true) > 0:
        return y_true.sum()*1.0/len(y_true)
    else:
        return 0.0

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

    k = min(k, len(y_true))

    return precision_at_k(y_true.values, y_score.values, k, extrapolate)

def top_k(y_true, y_score, k, extrapolate=False):
    if len(y_true) != len(y_score):
        raise ValueError('Labels and scores must have same lengths: %s != %s' 
                 % (len(y_true), len(y_score)))

    if k == 0:
        return (0,0)

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

def precision_at_k(y_true, y_score, k, extrapolate=False):
    n,d = top_k(y_true, y_score, k, extrapolate)
    p = n*1.0/d

    if extrapolate:
        s = np.sqrt(p*(1 - p) / d)
        return p,s
    else:
        return p

def precision_series(y_true, y_score, k):
    ranks = y_score.argsort()
    top_k = ranks[::-1][0:k]
    return pd.Series(y_true[top_k].cumsum()*1.0/np.arange(1,k+1), index=np.arange(1,k+1))

def _mask(run, masks, test, outcome='true'):
    if test:
        masks = masks + ['test']

    masks2 = []
    for mask in masks:
        if mask in run['y'].columns:
            masks2.append(run['y'][mask])
        else:
            masks2.append(util.index_as_series(run['y'], mask))

    mask = reduce(lambda a,b: a & b, masks2)
    return run['y'][mask][outcome], run['y'][mask]['score']
