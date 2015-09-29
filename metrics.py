import numpy as np
import pandas as pd

def baseline(y_true, y_score, data, masks=None):
    if masks is not None:
        y_true,y_score = _mask(y_true, y_score, data, masks)
    if len(y_true) > 0:
        return y_true.sum()*1.0/len(y_true)
    else:
        return 0.0

def precision(y_true, y_score, data, k=None, p=None, masks=None):
    # deal with k or p
    if k is not None and p is not None:
        raise ValueError("precision: cannot specify both k and p")
    elif k is not None:
        k = k
    elif p is not None:
        k = int(p*len(y_true))
    else:
        raise ValueError("precision must specify either k or p")

    if masks is not None:
        y_true, y_score = _mask(y_true, y_score, data, masks)

    return precision_at_k(y_true.values, y_score.values, k)

def precision_at_k(y_true, y_score, k):
    ranks = y_score.argsort()
    top_k = ranks[-k:]
    return y_true[top_k].sum()*1.0/k

def precision_series(y_true, y_score, k):
    ranks = y_score.argsort()
    top_k = ranks[::-1][0:k]
    return pd.Series(y_true[top_k].cumsum()*1.0/np.arange(1,k+1), index=np.arange(1,k+1))

def _mask(y_true, y_score, data, masks):
    mask = reduce(lambda a,b: a & b, (data.masks[mask] for mask in masks))
    mask = mask.loc[y_true.index]
    return y_true[mask], y_score[mask]
