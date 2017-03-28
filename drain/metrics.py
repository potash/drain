import numpy as np
import pandas as pd
import sklearn.metrics

from drain.util import to_float

"""Methods that calculate metrics for classification models.

All metrics are functions of two numpy arrays of floats of equal length:
    - y_true: the labels, either 0 or 1 or NaN
    - y_score: the scores, which can take any non-NaN number.
All metrics have been implemented to support missing labels.
"""


def _argsort(y_score, k=None):
    """
    Returns the indexes in descending order of the top k score
        or all scores if k is None
    """
    ranks = y_score.argsort()
    argsort = ranks[::-1]
    if k is not None:
        argsort = argsort[0:k]

    return argsort


def _argtop(y_score, k=None):
    """
    Returns the indexes of the top k scores (not necessarily sorted)
    """
    # avoid sorting when just want the top all
    if k is None:
        return slice(0, len(y_score))
    else:
        return _argsort(y_score, k)


def count(y_true, y_score=None, countna=False):
    """
    Counts the number of examples. If countna is False then only count labeled examples,
    i.e. those with y_true not NaN
    """
    if not countna:
        return (~np.isnan(to_float(y_true))).sum()
    else:
        return len(y_true)


def count_series(y_true, y_score, countna=False):
    """
    Returns series whose i-th entry is the number of examples in the top i
    """
    y_true, y_score = to_float(y_true, y_score)
    top = _argsort(y_score)

    if not countna:
        a = (~np.isnan(y_true[top])).cumsum()
    else:
        a = range(1, len(y_true)+1)

    return pd.Series(a, index=range(1, len(a)+1))


def baseline(y_true, y_score=None):
    """
    Number of positive labels divided by number of labels,
        or zero if there are no labels
    """
    if len(y_true) > 0:
        return np.nansum(y_true)/count(y_true, countna=False)
    else:
        return 0.0


def roc_auc(y_true, y_score):
    """
    Returns are under the ROC curve
    """
    notnull = ~np.isnan(y_true)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true[notnull], y_score[notnull])
    return sklearn.metrics.auc(fpr, tpr)


def precision(y_true, y_score, k=None, return_bounds=False):
    """
    If return_bounds is False then returns precision on the
        labeled examples in the top k.
    If return_bounds is True the returns a tuple containing:
        - precision on the labeled examples in the top k
        - number of labeled examples in the top k
        - lower bound of precision in the top k, assuming all
            unlabaled examples are False
        - upper bound of precision in the top k, assuming all
            unlabaled examples are True
    """
    y_true, y_score = to_float(y_true, y_score)
    top = _argtop(y_score, k)

    n = np.nan_to_num(y_true[top]).sum()    # fill missing labels with 0
    d = (~np.isnan(y_true[top])).sum()      # count number of labels
    p = n/d

    if return_bounds:
        k = len(y_true) if k is None else k
        bounds = (n/k, (n+k-d)/k) if k != 0 else (np.nan, np.nan)
        return p, d, bounds[0], bounds[1]
    else:
        return p


def precision_series(y_true, y_score, k=None):
    """
    Returns series of length k whose i-th entry is the precision in the top i
    TODO: extrapolate here
    """
    y_true, y_score = to_float(y_true, y_score)
    top = _argsort(y_score, k)

    n = np.nan_to_num(y_true[top]).cumsum()  # fill missing labels with 0
    d = (~np.isnan(y_true[top])).cumsum()    # count number of labels
    return pd.Series(n/d, index=np.arange(1, len(n)+1))


def recall(y_true, y_score, k=None, value=True):
    """
    Returns recall (number of positive examples) in the top k
    If value is False then counts number of negative examples
    TODO: add prop argument to return recall proportion instead of count
    """
    y_true, y_score = to_float(y_true, y_score)
    top = _argtop(y_score, k)

    if not value:
        y_true = 1-y_true

    r = np.nan_to_num(y_true[top]).sum()

    return r


def recall_series(y_true, y_score, k=None, value=True):
    """
    Returns series of length k whose i-th entry is the recall in the top i
    """
    y_true, y_score = to_float(y_true, y_score)
    top = _argsort(y_score, k)

    if not value:
        y_true = 1-y_true

    a = np.nan_to_num(y_true[top]).cumsum()
    return pd.Series(a, index=np.arange(1, len(a)+1))
