import pandas as pd
import numpy as np

from drain.model import y_subset

y = pd.DataFrame({
    'score':[.1,1,.2,.3,0],
    'true':[True, False, np.nan,False,True],
    'test':[True, True, True, False, False],
    'attr':[False, True, False, False, False],
})

def test_subset_query():
    assert set(y_subset(y, query='attr').index) == set([1])

def test_subset_dropna():
   assert set(y_subset(y, dropna=True).index) == set([0,1,3,4])

def test_subset_k():
   assert set(y_subset(y, k=2).index) == set([1,3])
