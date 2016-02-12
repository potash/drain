import pytest
import pandas as pd
import os
import sys
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

