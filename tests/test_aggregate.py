import pandas as pd
from drain.aggregate import *
from itertools import product
from pandas.util.testing import assert_frame_equal

def test_aggregator(small_df):

    aggregates = [
        Count(),
        Aggregate(['score', lambda x: x.score**2],'sum', ['score', 'lambda']),
        Aggregate(lambda x: x.arrests%2,'sum', name='typetest')
        ]

    ag = Aggregator(small_df, aggregates).aggregate('name')
    df = pd.DataFrame({'count':[2,1,1],
                       'score_sum': [0.3,0.4,1.0],
                       'lambda_sum': [0.05, 0.16, 1.0],
                       'typetest_sum': [True, False, True]},
                        index=['Anne','Ben','Charlie'], dtype=np.float32)
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
                       'score_per_arrests': [0.1,0.2,0.2],
                       'score_per_propname': [0.3/5,0.4/3,1.0/6]},
                        index=['Anne','Ben','Charlie'], dtype=np.float32)
    df = df.reindex_axis(sorted(df.columns), axis=1)
    ag = ag.reindex_axis(sorted(ag.columns), axis=1)
    df.index.name = 'name'
    assert_frame_equal(ag, df)

def test_fraction(small_df):
    n = Aggregate('Arrest', 'sum')
    d = Aggregate('ID', 'count')

    t = (True, False)
    for i_n,i_f,i_d in product(t,t,t):
        if not i_n and not i_f and not i_d:
            continue
        f = Fraction(n,d, include_numerator=i_n, include_denominator=i_d, include_fraction=i_f)
        a = Aggregator(crime_df, [f])
        df = a.aggregate('District')
        print(df)

def test_fraction(small_df):
    n = Aggregate('arrests', 'sum')
    d = Aggregate('score', 'sum')

    f = Fraction(n,d, include_numerator=True, include_denominator=True, include_fraction=True)

    ag = Aggregator(small_df, [f]).aggregate('name')
    df = pd.DataFrame({'arrests_sum':[3,2,5],
                       'score_sum': [0.3,0.4,1.0],
                       'arrests_sum_per_score_sum': [3/0.3, 2/0.4, 5/1.0]},
                        index=['Anne','Ben','Charlie'], dtype=np.float32)

    df = df.reindex_axis(sorted(df.columns), axis=1)
    ag = ag.reindex_axis(sorted(ag.columns), axis=1)
    df.index.name = 'name'
    assert_frame_equal(ag, df)

def test_proportion(small_df):

    p = Proportion(lambda x: x.score+1, 'arrests', name='mylambda', denom_name='mydenom')

    ag = Aggregator(small_df, [p]).aggregate('name')
    df = pd.DataFrame({'mylambda_per_mydenom':[2.3/3,1.4/2,2.0/5]},
                    index=['Anne','Ben','Charlie'], dtype=np.float32)

    df = df.reindex_axis(sorted(df.columns), axis=1)
    ag = ag.reindex_axis(sorted(ag.columns), axis=1)
    df.index.name = 'name'
    assert_frame_equal(ag, df)

def test_lambda_counts(small_df):
    
    aggregates = [Count([lambda x,c=c: x.score+c for c in range(2)],
                         name=['lambdaname_%d'%c for c in range(2)])]

    ag = Aggregator(small_df, aggregates).aggregate('name')

    df = pd.DataFrame({'lambdaname_0_count':[0.3, 0.4, 1.0],
                       'lambdaname_1_count':[2.3, 1.4, 2.0]},
                        index=['Anne','Ben','Charlie'], dtype=np.float32)

    df = df.reindex_axis(sorted(df.columns), axis=1)
    ag = ag.reindex_axis(sorted(ag.columns), axis=1)
    df.index.name = 'name'
    assert_frame_equal(ag, df)

