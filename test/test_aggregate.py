import pandas as pd
from drain.aggregate import *

def test_aggregator(crime_df):
    print crime_df

    aggregates = [
        Count(name='ID'), 
        Count(name='Arrest'), 
        Count(lambda c: c['Primary Type'] == 'THEFT', 'theft', prop=True),
    ]

    a = Aggregator(crime_df, aggregates)
    print a.aggregate('District')
