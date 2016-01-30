from drain.aggregation import SimpleAggregation, SpacetimeAggregation
from drain.aggregate import Count
from drain import step
import numpy as np

class SimpleCrimeAggregation(SimpleAggregation):
    @property
    def aggregates(self):
        return [
       	    Count(name='ID'),
            Count(name='Arrest'),
      	    Count(lambda c: c['Primary Type'] == 'THEFT', 'theft', prop=True),
	]

def test_simple_aggregation(crime_step):
    s = SimpleCrimeAggregation(inputs=[crime_step], 
	indexes=['District', 'Community Area'], parallel=True)
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
            spacedeltas={'district': ('District', ['1m', '2m']),
                         'community':('Community Area', ['1m', '6m'])},
            dates=[np.datetime64('2013-01-01'), np.datetime64('2013-06-01')])

    step.run(s)
    print s.get_result()
