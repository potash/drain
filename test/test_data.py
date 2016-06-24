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

    r0, r1 = h.get_result(), d.get_result()

    for key in r1.keys():
       assert r0[key].equals(r1[key])

def test_date_select():
    df = pd.DataFrame({'date':pd.to_datetime(
            [date(2013,m,1) for m in range(1,13)])})
    assert np.array_equal(data.date_select(df, 'date', date(2013,4,1), 'all').values, df.values[0:3])

    # test it on a pandas timestamp column too
    df['date'] = pd.to_datetime(df['date'])
    assert np.array_equal(data.date_select(df, 'date', date(2013,4,1), 'all').values, df.values[0:3])

