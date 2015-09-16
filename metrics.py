def baseline(y_true):
    return y_true.sum()*1.0/len(y_true)

def precision(y_true, y_score, data, k=None, p=None, mask=None):
    # deal with k or p
    if k is not None and p is not None:
        raise ValueError("precision: cannot specify both k and p")
    elif k is not None:
        k = k
    elif p is not None:
        k = int(p*len(y_true))
    else:
        raise ValueError("precision must specify either k or p")

    if mask is not None:
        y_true, y_score = _mask(y_true, y_score, data, mask)

    return precision_at_k(y_true.values, y_score.values, k)

def precision_at_k(y_true, y_score, k):
    ranks = y_score.argsort()
    top_k = ranks[-k:]
    return y_true[top_k].sum()*1.0/k

def _mask(y_true, y_score, data, mask):
    mask = (data.X[mask] > 0).loc[y_true.index]
    return y_true[mask], y_score[mask]