import pandas as pd
from drain.aggregate import *
from itertools import product
from pandas.util.testing import assert_frame_equal


def test_aggregator(crime_df, small_df):
    aggregates = [
         Count(), 
         Count('Arrest'), 
         Count([lambda c: c['Primary Type'] == 'THEFT',
                lambda c: c['Primary Type'] == 'ASSAULT'], 
                ['theft', 'assault'], prop=True),
        Aggregate(['Latitude', 'Longitude'], ['min','max','mean'])
    ]


    aggregates = [
        Count(),
        Aggregate(['score', lambda x: x.score**2],'sum', ['score', 'lambda'])
        ]

    ag = Aggregator(small_df, aggregates).aggregate('name')
    df = pd.DataFrame({'count':[2,1,1],
                       'score_sum': [0.3,0.4,1.0],
                       'lambda_sum': [0.05, 0.16, 1.0]},
                        index=['Anne','Ben','Charlie'])
    df = df.reindex_axis(sorted(df.columns), axis=1)
    ag = ag.reindex_axis(sorted(ag.columns), axis=1)
    df.index.name = 'name'
    assert_frame_equal(ag, df)

def test_count(small_df):

    aggregates = [Count('score', name='myname'),
                  Count(lambda x: x.score+1, name='lambdaname'),
                  Count('score', prop=True,name='foo'),
                  Count('score', prop='arrests'),
                  Count('score', prop=lambda x: x.arrests+1, prop_only=True, prop_name='propname')]

    ag = Aggregator(small_df, aggregates).aggregate('name')

    df = pd.DataFrame({'myname_count':[0.3,0.4,1.0],
                       'lambdaname_count': [2.3,1.4,2.0],
                       'foo_count': [0.3,0.4,1.0],
                       'foo_prop': [0.15,0.4,1.0],
                       'score_count': [0.3,0.4,1.0],
                       'score_prop_arrests': [0.1,0.2,0.2],
                       'score_prop_propname': [0.3/5,0.4/3,1.0/6]},
                        index=['Anne','Ben','Charlie'])
    df = df.reindex_axis(sorted(df.columns), axis=1)
    ag = ag.reindex_axis(sorted(ag.columns), axis=1)
    df.index.name = 'name'
    assert_frame_equal(ag, df)


def test_fraction(crime_df):
    n = Aggregate('Arrest', 'sum')
    d = Aggregate('ID', 'count')

    t = (True, False)
    for i_n,i_f,i_d in product(t,t,t):
        if not i_n and not i_f and not i_d:
            continue
        f = Fraction(n,d, include_numerator=i_n, include_denominator=i_d, include_fraction=i_f)
        a = Aggregator(crime_df, [f])
        df = a.aggregate('District')
        print df

