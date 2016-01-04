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

class TestAggregator(SpacetimeAggregator):
    def __init__(self, basedir):
        SpacetimeAggregator.__init__(self, 
                {'state': Spacedeltas('state_id', ['all', '2y']),
                 'city': Spacedeltas('city_id', ['all','5y'])}, 
                [date(2000+y,1,1) for y in range(10)],
                'test', basedir)
                                     
    def get_data(self, d):
        return pd.DataFrame({'type':['urban']*5+['rural']*5, 'state':['MI', 'IL']*5, 'date':[date(2014,1,1)]*10,
                      'population':range(100,600,100) + range(1000,6000,1000)})

        
    def get_aggregates(self, d):
        return [Aggregate('population', ['sum', 'mean']), Aggregate('income', ['sum','mean'])]

#def test_spacetime_aggregator():
# TODO
