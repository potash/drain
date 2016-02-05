import pandas as pd
from drain.aggregate import *
from itertools import product

def test_aggregator(crime_df):
    aggregates = [
        Count(), 
        Count('Arrest'), 
        Count([lambda c: c['Primary Type'] == 'THEFT',
               lambda c: c['Primary Type'] == 'ASSAULT'], 
               ['theft', 'assault'], prop=True),
    ]

    a = Aggregator(crime_df, aggregates)
    print a.aggregate('District')

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

