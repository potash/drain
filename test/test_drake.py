from drain.drake import *
from drain.step import Step

def test_input_targets(drain_setup):
    assert get_input_targets(Step(value=1, inputs=[Step(value=2)])) == set()

def test_input_targets2(drain_setup):
    assert get_input_targets(Step(value=1, target=True, inputs=[Step(value=2, target=True), Step(value=3)])) == set([Step(value=2)])

# no output step
def test_drake_data(drain_setup):
    step = Step(a=1)
    data = get_drake_data([step])
    assert data == {Step(a=1) : set()}

# no output step with no target input
def test_drake_data2(drain_setup):
    step = Step(a=1, inputs=[Step(b=1)])
    data = get_drake_data([step])
    assert data == {step : set()}

# no output step with target
def test_drake_data3(drain_setup):
    step = Step(a=1, inputs=[Step(b=1, target=True)])
    data = get_drake_data([step])
    assert data == { Step(b=1) : set(), step : {Step(b=1)} }

# multiple no output steps on single target
def test_drake_data4(drain_setup):
    steps = [Step(a=1, inputs=[Step(b=1, target=True)]),
             Step(a=2, inputs=[Step(b=1, target=True)])]
    data = get_drake_data(steps)
    assert data == {Step(b=1) : set(), steps[0] : {Step(b=1)}, steps[1] : {Step(b=1)} }

def test_drakefile(drain_setup):
    steps = [Step(a=1, inputs=[Step(b=1, target=True)]),
             Step(a=2, inputs=[Step(b=1, target=True)])]
    print to_drakefile(steps, preview=True)
