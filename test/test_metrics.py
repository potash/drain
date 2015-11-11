import numpy as np
from drain.metrics import top_k, precision_at_k

from numpy.testing import assert_almost_equal
import pytest

def test_top_k():
    y_true = np.array([True, True, False, False])
    y_score = np.array([.25, 1, 0, .5])

    assert top_k(y_true, y_score,3) == (2,3)

# missing labels with extrapolate=False
def test_top_k_null():
    y_true = np.array([True, False, np.nan])
    y_score = np.array([1,0,.5])
    
    assert top_k(y_true, y_score,2) == (1,2)

#missing labels with extrapolate=True
def test_top_k_extrapolate():
    y_true = np.array([True, False, np.nan])
    y_score = np.array([1,0,.5])
    
    assert top_k(y_true, y_score,2, extrapolate=True) == (1,1)

# all labels missing in top k with extrapolate=True
# same as precision at k=0, i.e. infinite
def test_top_k_extrapolate_empty():
    y_true = np.array([np.nan, False, np.nan])
    y_score = np.array([1,0,.5])

    assert top_k(y_true, y_score, 2, extrapolate=True) == (0,0)

# when k=0 precision should be infinite
def test_top_k_0():
    y_true = np.array([True])
    y_score = np.array([1])

    assert top_k(y_true, y_score, 0) == (0,0)

# should throw error when number of labels is less than k
def test_top_k_label_error():
    y_true = np.array([True, False, np.nan])
    y_score = np.array([1,0,.5])

    with pytest.raises(ValueError):
        top_k(y_true, y_score, 3)

# lengths of labels and scores must match
def test_top_k_len_error():
    y_true = np.array([True, False])
    y_score = np.array([1,0,.5])

    with pytest.raises(ValueError):
        top_k(y_true, y_score, 3)

def test_precision_at_k():
    y_true = np.array([True, True, False])
    y_score = np.array([1, 0, .5])

    assert precision_at_k(y_true, y_score,2) == .5

def test_precision_at_k_extrapolate():
    y_true = np.array([True, False, np.nan])
    y_score = np.array([1, 0, .5])

    p,s = precision_at_k(y_true, y_score,3, extrapolate=True)
    assert p == .5
    assert_almost_equal(s, 1/np.sqrt(8))
