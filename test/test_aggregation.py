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
      	    Count(lambda c: c['Primary Type'] == 'THEFT', 'theft', prop=True),
	]

def test_simple_aggregation(crime_step):
    s = SimpleCrimeAggregation( 
	indexes=['District', 'Community Area'], parallel=True)
    s.inputs=[crime_step]
    step.run(s)
    print s.get_result()

class SpacetimeCrimeAggregation(SpacetimeAggregation):
    def __init__(self, inputs, spacedeltas, dates, **kwargs):
        SpacetimeAggregation.__init__(self,
                inputs=inputs, spacedeltas=spacedeltas, dates=dates,
                date_column='Date', **kwargs)

    def get_aggregates(self, date, delta):
        return [
            Count(name='ID'),
            Count(name='Arrest'),
            Count(lambda c: c['Primary Type'] == 'THEFT', 'theft', prop=True)
        ]


def test_spacetime_aggregation(crime_step):
    s = SpacetimeCrimeAggregation(inputs=[crime_step], 
            spacedeltas={'district': ('District', ['12h', '24h']),
                         'community':('Community Area', ['1d', '2d'])}, parallel=True,
            dates=[date(2015,12,30), date(2015,12,31)])

    step.run(s)
    print s.get_result()

def test_simple_join(crime_step):
    s = SimpleCrimeAggregation(inputs=[crime_step],
        indexes=['District', 'Community Area'], parallel=True)
    step.run(s)

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,2]})
    print s.join(left)

def test_spacetime_join(crime_step):
    s = SpacetimeCrimeAggregation(inputs=[crime_step], 
            spacedeltas={'district': ('District', ['12h', '24h']),
                         'community':('Community Area', ['1d', '2d'])},
            dates=[date(2015,12,30), date(2015,12,31)], parallel=True)
    step.run(s)

    left = pd.DataFrame({'District':[1,2], 'Community Area':[1,2], 'date':[np.datetime64(date(2015,12,30)), np.datetime64(date(2015,12,31))]})
    print s.join(left)

