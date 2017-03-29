import pytest
import pandas as pd
import numpy as np
import os
import tempfile
from datetime import date

import drain
from drain import step
from drain.step import Step
from drain.aggregation import SpacetimeAggregation
from drain.aggregate import Count

# this fixture sets up drain for testing
@pytest.fixture(scope="session")
def drain_setup(request):
    # use a temporary dir
    tmpdir = tempfile.mkdtemp()
    drain.PATH = tmpdir
    # configure for yaml dumping/serialization
    def fin():
        print("\nDoing teardown")
    request.addfinalizer(fin)

@pytest.fixture
def crime_df():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), 'crimes.csv'),
            parse_dates=['Date'])

@pytest.fixture
def small_df():
    return pd.DataFrame({
        'name': ['Anne', 'Ben', 'Anne', 'Charlie'],
        'arrests': [1, 2, 2, 5],
        'stop': [True, False, True, True],
        'score': [0.2, 0.4, 0.1, 1.0]
        })

class CrimeDataStep(Step):
    def run(self):
        return crime_df()

@pytest.fixture
def crime_step():
    return CrimeDataStep()

class SpacetimeCrimeAggregation(SpacetimeAggregation):
    def __init__(self, inputs, spacedeltas, dates, parallel=False):
        self.inputs = inputs

        SpacetimeAggregation.__init__(self,
                spacedeltas=spacedeltas, dates=dates,
                date_column='Date', prefix='crimes', parallel=parallel)

    def get_aggregates(self, date, delta):
        return [
            Count(),
            Count('Arrest'),
            Count(lambda c: c['Primary Type'] == 'THEFT',
                    'theft', prop=True)
        ]

class SpacetimeCrimeLeft(Step):
    def run(self):
        return pd.DataFrame({'District':[1,2], 'Community Area':[1,2],
        'date':[np.datetime64(date(2015,12,30)), np.datetime64(date(2015,12,31))]})

@pytest.fixture
def spacetime_crime_agg(crime_step):
    return SpacetimeCrimeAggregation(inputs=[crime_step],
        spacedeltas={'district': ('District', ['12h', '24h']),
                     'community':('Community Area', ['1d', '2d'])}, 
        dates=[date(2015,12,30), date(2015,12,31)])

class SpacetimeCrimeLeft(Step):
    def run(self):
        return pd.DataFrame({'District':[1,2], 'Community Area':[1,2],
        'date':[np.datetime64(date(2015,12,30)), np.datetime64(date(2015,12,31))]})

@pytest.fixture
def spacetime_crime_left():
    return SpacetimeCrimeLeft()
