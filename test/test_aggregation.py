from drain.aggregation import SimpleAggregation, SpacetimeAggregation
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
class SpacetimeCrimeAggregation(SpacetimeAggregation):
    def __init__(self, inputs, spacedeltas, dates, **kwargs):
        SpacetimeAggregation.__init__(self,
                inputs=inputs, spacedeltas=spacedeltas, dates=dates,
                astype=np.float32,
                date_column='Date', prefix='crimes', **kwargs)

    def get_aggregates(self, date, delta):
        return [
            Count(),
            Count('Arrest'),
            Count(lambda c: c['Primary Type'] == 'THEFT', 
                    'theft', prop=True)
        ]

def test_simple_aggregation_parallel(crime_step):
    s = SimpleCrimeAggregation(inputs=[crime_step], 
	indexes=['District', 'Community Area'], parallel=True)
    s.inputs=[crime_step]
    step.run(s)
    print s.get_result()

def test_simple_aggregation(crime_step):
    s = SimpleCrimeAggregation(inputs=[crime_step], 
	indexes=['District', 'Community Area'], parallel=False)
    s.inputs=[crime_step]
    step.run(s)
    print s.get_result()

def test_simple_join(crime_step):
    s = SimpleCrimeAggregation(inputs=[crime_step],
        indexes=['District', 'Community Area'], parallel=True)
    step.run(s)

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,2]})
    print s.join(left)

def test_simple_join_fillna(crime_step):
    s = SimpleCrimeAggregation(inputs=[crime_step],
        indexes=['District', 'Community Area'], parallel=True)
    step.run(s)

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,100]})
    print s.join(left)

def test_spacetime_aggregation(spacetime_crime_agg):
    step.run(spacetime_crime_agg)
    print spacetime_crime_agg.get_result()

def test_spacetime_join(spacetime_crime_agg):
    step.run(spacetime_crime_agg)

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,2], 
        'date':[np.datetime64(date(2015,12,30)), np.datetime64(date(2015,12,31))]})
    print spacetime_crime_agg.join(left)

def test_spacetime_join_select(spacetime_crime_agg):
    step.run(spacetime_crime_agg)

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,2], 
        'date':[np.datetime64(date(2015,12,30)), np.datetime64(date(2015,12,31))]})
    df = spacetime_crime_agg.join(left)
    print spacetime_crime_agg.select(df, {'district': ['12h']})

def test_spacetime_join_fillna(spacetime_crime_agg):
    step.run(spacetime_crime_agg)

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,100], 
        'date':[np.datetime64(date(2015,12,30)), np.datetime64(date(2015,12,31))]})
    print spacetime_crime_agg.join(left)

