import numpy as np
import pandas as pd
import sklearn.metrics
from drain import util

def baseline(run, masks=[], test=True):
    y_true,y_score = _mask(run, masks, test)
    if len(y_true) > 0:
        return y_true.sum()*1.0/len(y_true)
    else:
        return 0.0

def auc(run, masks=[], test=True):
    y_true, y_score = _mask(run, masks, test)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_score)
    return sklearn.metrics.auc(fpr, tpr)

def precision(run, k=None, p=None, masks=[], test=True):
    y_true, y_score = _mask(run, masks, test)

    # deal with k or p
    if k is not None and p is not None:
        raise ValueError("precision: cannot specify both k and p")
    elif k is not None:
        k = k
    elif p is not None:
        k = int(p*len(y_true))
    else:
        raise ValueError("precision must specify either k or p")

    return precision_at_k(y_true.values, y_score.values, k)

def precision_at_k(y_true, y_score, k):
    ranks = y_score.argsort()
    top_k = ranks[-k:]
    return y_true[top_k].sum()*1.0/k

def precision_series(y_true, y_score, k):
    ranks = y_score.argsort()
    top_k = ranks[::-1][0:k]
    return pd.Series(y_true[top_k].cumsum()*1.0/np.arange(1,k+1), index=np.arange(1,k+1))

def _mask(run, masks, test):
    if test:
        masks = masks + ['test']

    masks2 = []
    for mask in masks:
        if mask in run['y'].columns:
            masks2.append(run['y'][mask])
        else:
            masks2.append(util.index_as_series(run['y'], mask))

    mask = reduce(lambda a,b: a & b, masks2)
    return run['y'][mask]['true'], run['y'][mask]['score']
