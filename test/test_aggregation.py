from drain.aggregation import SimpleAggregation
from drain.aggregate import Count
from drain import step

class TestAggregation(SimpleAggregation):
    @property
    def aggregates(self):
        return [Count()]

def test_aggregation(df_step):
    s = TestAggregation(inputs=[df_step], indexes=['state', 'type'], parallel=True)
    step.run(s)
    print s.get_result()
