import numpy as np
from drain.metrics import precision

from numpy.testing import assert_almost_equal
import pytest

def test_precision():
    y_true = np.array([True, True, False])
    y_score = np.array([1, 0, .5])

    assert precision(y_true, y_score,k=2) == .5

def test_precision_nan():
    y_true = np.array([True, False, np.nan])
    y_score = np.array([1, 0, .5])

    p = precision(y_true, y_score,None, return_bounds=True)
    assert p == (.5, 2, 1.0/3, 2.0/3)
