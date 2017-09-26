from drain.aggregation import SimpleAggregation, SpacetimeAggregation, AggregationJoin, SpacetimeAggregationJoin
from drain.aggregate import Count
from drain import step
from datetime import date
import pandas as pd
import numpy as np

class SimpleCrimeAggregation(SimpleAggregation):
    @property
    def aggregates(self):
        return [
       	    Count(),
            Count('Arrest'),
      	    Count(lambda c: c['Primary Type'] == 'THEFT', 
                    'theft', prop=True),
	]

def test_simple_aggregation_parallel(drain_setup, crime_step):
    s = SimpleCrimeAggregation(inputs=[crime_step], 
	indexes=['District', 'Community Area'], parallel=True)
    s.execute()
    print(s.result)

def test_simple_aggregation(drain_setup, crime_step):
    s = SimpleCrimeAggregation(inputs=[crime_step], 
	indexes=['District', 'Community Area'], parallel=False)
    s.execute()
    print(s.result)

def test_simple_join(drain_setup, crime_step):
    s = SimpleCrimeAggregation(inputs=[crime_step],
        indexes=['District', 'Community Area'], parallel=True)
    s.execute()

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,2]})
    print(s.join(left))

def test_simple_join_fillna(drain_setup, crime_step):
    s = SimpleCrimeAggregation(inputs=[crime_step],
        indexes=['District', 'Community Area'], parallel=True)
    s.execute()

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,100]})
    print(s.join(left))

def test_spacetime_aggregation(drain_setup, spacetime_crime_agg):
    spacetime_crime_agg.execute()
    print(spacetime_crime_agg.result)

def test_spacetime_join(drain_setup, spacetime_crime_agg):
    spacetime_crime_agg.execute()

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,2], 
        'date':[np.datetime64(date(2015,12,30)), np.datetime64(date(2015,12,31))]})
    print(spacetime_crime_agg.join(left))

def test_spacetime_join_step(spacetime_crime_agg, spacetime_crime_left):
    join = AggregationJoin(inputs=[spacetime_crime_agg, spacetime_crime_left])
    result = join.execute()
    print(result)

def test_spacetime_join_lag(spacetime_crime_agg, spacetime_crime_left):
    join = SpacetimeAggregationJoin(lag='1d', 
            inputs=[spacetime_crime_agg, spacetime_crime_left])
    result = join.execute()
    print(result)

def test_spacetime_join_select(drain_setup, spacetime_crime_agg):
    spacetime_crime_agg.execute()

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,2], 
        'date':[np.datetime64(date(2015,12,30)), np.datetime64(date(2015,12,31))]})
    df = spacetime_crime_agg.join(left)
    print(spacetime_crime_agg.select(df, {'district': ['12h']}))

def test_spacetime_join_fillna(drain_setup, spacetime_crime_agg):
    spacetime_crime_agg.execute()

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,100], 
        'date':[np.datetime64(date(2015,12,30)), np.datetime64(date(2015,12,31))]})
    print(spacetime_crime_agg.join(left))

