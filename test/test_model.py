import pandas as pd
import numpy as np

from drain.model import y_subset

y = pd.DataFrame({
    'score':[.1,1,.2,.3,0],
    'true':[True, False, np.nan,False,True],
    'test':[True, True, True, False, False],
    'mask':[False, True, False, False, False],
    'filter': [0,0,1,1,2]
})

def test_subset_mask():
    assert set(y_subset(y, masks=['mask']).index) == set([1])

def test_subset_filters():
    assert set(y_subset(y, filters={'filter':1}).index) == set([2])

def test_subset_dropna():
   assert set(y_subset(y, dropna=True).index) == set([0,1])

def test_subset_k():
   assert set(y_subset(y, k=2).index) == set([1,2])

#def test_subset():
