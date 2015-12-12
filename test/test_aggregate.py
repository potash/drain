import pandas as pd
from drain.aggregate import *

def test_aggregator():
    df = pd.DataFrame({'type':['urban']*5+['rural']*5, 'state':['MI', 'IL']*5, 
                      'population':range(100,600,100) + range(1000,6000,1000)})

    df['income'] = df['population'] * (10+(df['type'] == 'urban')*5)

    aggregates = [
        Count(name='cities'), 
        Count(lambda c: c.type == 'rural', 'rural', prop=True),
        MultiAggregate('population', ['sum', 'mean', 'median'])]

    a = Aggregator(df, aggregates)
    return a.aggregate('state')
