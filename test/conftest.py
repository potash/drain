import pytest
import pandas as pd
import os

from drain.step import Step
from drain.aggregation import SpacetimeAggregation
from drain.aggregate import Count
from datetime import date

@pytest.fixture
def crime_df():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'crimes.csv'),
            parse_dates=['Date'])

class CrimeDataStep(Step):
    def run(self):
        return crime_df()

@pytest.fixture
def crime_step():
    return CrimeDataStep()

class SpacetimeCrimeAggregation(SpacetimeAggregation):
    def __init__(self, inputs, spacedeltas, dates, **kwargs):
        SpacetimeAggregation.__init__(self,
                inputs=inputs, spacedeltas=spacedeltas, dates=dates,
                date_column='Date', prefix='crimes', **kwargs)

    def get_aggregates(self, date, delta):
        return [
            Count(),
            Count('Arrest'),
            Count(lambda c: c['Primary Type'] == 'THEFT',
                    'theft', prop=True)
        ]
@pytest.fixture
def spacetime_crime_agg(crime_step):
    return SpacetimeCrimeAggregation(inputs=[crime_step],
        spacedeltas={'district': ('District', ['12h', '24h']),
                     'community':('Community Area', ['1d', '2d'])}, parallel=True,
        dates=[date(2015,12,30), date(2015,12,31)])