from drain import data,step
import tempfile
import os
from datetime import date
import pandas as pd
import numpy as np

def test_to_hdf():
    d = data.ClassificationData()
    h = data.ToHDF(inputs=[d], target=True)

    h.setup_dump()
    h.execute()

    r0, r1 = h.result, d.result

    for key in r1.keys():
       assert r0[key].equals(r1[key])

def test_date_select():
    df = pd.DataFrame({'date':pd.to_datetime(
            [date(2013,m,1) for m in range(1,13)])})
    assert np.array_equal(data.date_select(df, 'date', date(2013,4,1), 'all').values, df.values[0:3])

    # test it on a pandas timestamp column too
    df['date'] = pd.to_datetime(df['date'])
    assert np.array_equal(data.date_select(df, 'date', date(2013,4,1), 'all').values, df.values[0:3])

def test_binarize_inplace():
    df = pd.DataFrame({'a':['b','c']})
    data.binarize(df, ['a'], inplace=True)
    assert df.columns.tolist() == ['a_b', 'a_c']

def test_binarize_drop():
    df = pd.DataFrame({'a':['b','c']})
    data.binarize(df, ['a'], drop=False, inplace=True)
    assert df.columns.tolist() == ['a', 'a_b', 'a_c']

def test_binarize_all_classes():
    df = pd.DataFrame({'a':['b','c']})
    data.binarize(df, ['a'], all_classes=False, inplace=True)
    assert df.columns.tolist() == ['a_b']

def test_binarize_not_inplace():
    df = pd.DataFrame({'a':['b','c']})
    df2 = data.binarize(df, ['a'], inplace=False)
    assert df.columns.tolist() == ['a']
    assert df2.columns.tolist() == ['a_b', 'a_c']

def test_impute():
    df = pd.DataFrame([1.0,None,3.0])
    i = data.impute(df)
    assert i.equals(pd.DataFrame([1.0,2.0,3.0]))

def test_impute_train():
    df = pd.DataFrame([1.0,None,3.0])
    i = data.impute(df, train=pd.Series([True,False,False]))
    assert i.equals(pd.DataFrame([1.0,1.0,3.0]))

def test_impute_value():
    df = pd.DataFrame([1.0,None,3.0])
    i = data.impute(df, value=pd.Series({0:10.0}))
    assert i.equals(pd.DataFrame([1.0,10.0,3.0]))
