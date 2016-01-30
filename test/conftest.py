import pytest
import pandas as pd
from drain.step import Step

@pytest.fixture
def df():
    df = pd.DataFrame({'type':['urban']*5+['rural']*5, 'state':['MI', 'IL']*5,
            'population':range(100,600,100) + range(1000,6000,1000)})
    df['income'] = df['population'] * (10+(df['type'] == 'urban')*5)

    return df

class TestDataStep(Step):
    def run(self):
        return df()

@pytest.fixture
def df_step():
    return TestDataStep()
