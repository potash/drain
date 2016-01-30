import pytest
import pandas as pd
import os
from drain.step import Step

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
