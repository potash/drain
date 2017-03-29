from drain.drake import *
from drain.step import Step

def test_inputs_target(drain_setup):
    assert get_inputs(Step(value=1, inputs=[Step(value=2)]), target=True) == set()

def test_inputs_target2(drain_setup):
    inputs = [Step(value=2), Step(value=3)]
    inputs[0].target = True

    step = Step(value=1)
    step.inputs = inputs
    step.target = True

    assert get_inputs(step, target=True) == set([Step(value=2)])

def test_inputs_no_target(drain_setup):
    inputs = [Step(value=2), Step(value=3, inputs=[Step(value=4)])]
    inputs[0].target = True

    step = Step(value=1, inputs=inputs)
    step.target = True

    assert get_inputs(step, target=False) ==\
            set([Step(value=3, inputs=[Step(value=4)]), Step(value=4)])

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
    inputs = [Step(b=1)]
    inputs[0].target = True
    step = Step(a=1, inputs=inputs)
    data = get_drake_data([step])
    assert data == { Step(b=1) : set(), step : {Step(b=1)} }

# multiple no output steps on single target
def test_drake_data4(drain_setup):
    inputs = [Step(b=1)]
    inputs[0].target = True

    steps = [Step(a=1, inputs=inputs),
             Step(a=2, inputs=inputs)]

    data = get_drake_data(steps)
    assert data == {Step(b=1) : set(), steps[0] : {Step(b=1)}, steps[1] : {Step(b=1)} }

def test_drakefile(drain_setup):
    inputs = [Step(b=1)]
    inputs[0].target = True
    steps = [Step(a=1, inputs=inputs),
             Step(a=2, inputs=inputs)]
    print(to_drakefile(steps, preview=True))
