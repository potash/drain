from drain.step import *
import tempfile

def setup_module(module):
    tmpdir = tempfile.mkdtemp()
    initialize(tmpdir)

def test_argument_product():
    assert argument_product({'a' : ArgumentCollection([1,2])}) == [{'a': 1}, {'a': 2}]

def test_argument_product_deep():
    assert argument_product({'a' : {'b': ArgumentCollection([1,2])} }) == [{'a': {'b': 1}}, {'a': {'b': 2}}]

def test_step_product():
    assert step_product(Step._template(a={'b': ArgumentCollection([1,2])})) == [Step._template(a={'b': 1}), Step._template(a={'b': 2})]

def test_parallel():
    assert parallel(Step._template(a=1), Step._template(b=1)) == [[Step._template(a=1)], [Step._template(b=1)]]

def test_parallel_multiple():
    assert parallel([Step._template(a=1), Step._template(b=1)], [Step._template(c=1)]) == [[Step._template(a=1)], [Step._template(b=1)], [Step._template(c=1)]]

def test_search():
    assert search(Step._template(a=ArgumentCollection([1,2]))) == [Step._template(a=1), Step._template(a=2)]

def test_search_multiple():
    assert search(Step._template(a=ArgumentCollection([1,2])), Step._template(b=ArgumentCollection([1,2]))) == [Step._template(a=1), Step._template(a=2), Step._template(b=1), Step._template(b=2)]

def test_product():
    assert product([Step._template(a=1), Step._template(a=2)], Step._template(b=1)) == [(Step._template(a=1), Step._template(b=1)), (Step._template(a=2), Step._template(b=1))]
    
def test_serial():
    assert serial([Step._template(a=1), Step._template(a=2)], [Step._template(b=1), Step._template(b=2)]) == [Step._template(inputs=[Step._template(a=1), Step._template(a=2)],b=1), Step._template(inputs=[Step._template(a=1), Step._template(a=2)],b=2)]

def test_serial2():
    assert serial([[Step._template(a=1), Step._template(a=2)], [Step._template(b=1), Step._template(b=2)]], [Step._template(c=1), Step._template(c=2)]) == [Step._template(c=1, inputs=[Step._template(a=1), Step._template(a=2)]), Step._template(c=2, inputs=[Step._template(a=1), Step._template(a=2)]), Step._template(c=1, inputs=[Step._template(b=1), Step._template(b=2)]), Step._template(c=2, inputs=[Step._template(b=1), Step._template(b=2)])]
    
def test_run():
    step = yaml.load("""
    !serial 
      - !search
        - !step:drain.step.Scalar 
            value : !range [1,10]
      - !step:drain.step.Add {target : True}
    """)[0]
    print step
    step._template_construct()
    print step
    assert run(step) == 45

def test_run2():
    step = yaml.load("""
    !serial 
      - !product
        - !step:drain.step.Scalar {value : 1.}
        - !step:drain.step.Scalar {value : 2.}
      - !step:drain.step.Divide { inputs_mapping : [denominator, numerator] }
    """)[0]
    step._template_construct()
    assert run(step) == 2


def test_input_targets():
    assert get_input_targets(Step(value=1, inputs=[Step(value=2)])) == set()

def test_input_targets2():
    assert get_input_targets(Step(value=1, target=True, inputs=[Step(value=2, target=True), Step(value=3)])) == set([Step(value=2)])

def test_target_output():
    step = Step(value=1, target=True, inputs=[Step(value=2), Step(value=3,target=True)])
    assert get_output_targets(step) == set([step])

def test_target_output_no_output():
    step = Step(value=1, inputs=[Step(value=2, target=True)])
    assert get_output_targets(step) == set([Step(value=2)])

def test_target_output_output_multi():
    step = Step(value=1, inputs=[Step(value=2, target=True), Step(value=3, target=True)])
    assert get_output_targets(step) == set([Step(value=2), Step(value=3)])

def test_drake_data_helper():
    steps = [Step(value=1)]
    assert get_drake_data_helper(steps) == ({},{Step(value=1): set()})

def test_drake_data_helper2():
    steps = [Step(value=1, inputs=[Step(value=2, target=True)])]
    assert get_drake_data_helper(steps) == ({Step(value=2): set()}, {steps[0] : {Step(value=2)}})

def test_drake_data_helper3():
    steps = [Step(value=1, target=True)]
    assert get_drake_data_helper(steps) == ({Step(value=1): set()}, {})

def test_drake_data_helper4():
    steps = [Step(value=1, inputs=[Step(value=2, target=True), Step(value=3, target=True)])]
    assert get_drake_data_helper(steps) == (
            {Step(value=2):set(), Step(value=3):set()},
            {steps[0]: set([Step(value=2), Step(value=3)])}
    )

# when "same" step with and without target, runs both...
# should it?
def test_drake_data_helper5():
    steps = [Step(value=1), Step(value=1, target=True)]
    assert get_drake_data_helper(steps) == ({Step(value=1): set()}, {Step(value=1):set()})

def test_drake_data_helper6():
    steps = [Step(value=1, inputs=[Step(value=3, target=True)]), Step(value=2,  inputs=[Step(value=3, target=True)])]
    assert get_drake_data_helper(steps) == (
            {Step(value=3) : set()},
            {steps[0] : {Step(value=3)}, steps[1] : {Step(value=3)}},
    )

# no output step
def test_drake_data():
    step = Step(a=1)
    data = get_drake_data([step])
    assert data == {Step(a=1) : set()}

# no output step with no target input
def test_drake_data2():
    step = Step(a=1, inputs=[Step(b=1)])
    data = get_drake_data([step])
    assert data == {step : set()}

# no output step with target
def test_drake_data3():
    step = Step(a=1, inputs=[Step(b=1, target=True)])
    data = get_drake_data([step])
    assert data == { Step(b=1) : set(), step : {Step(b=1)} }

# multiple no output steps on single target
def test_drake_data4():
    steps = [Step(a=1, inputs=[Step(b=1, target=True)]),
             Step(a=2, inputs=[Step(b=1, target=True)])]
    data = get_drake_data(steps)
    assert data == {Step(b=1) : set(), steps[0] : {Step(b=1)}, steps[1] : {Step(b=1)} }

def test_drakefile():
    steps = [Step(a=1, inputs=[Step(b=1, target=True)]),
             Step(a=2, inputs=[Step(b=1, target=True)])]
    print to_drakefile(steps, preview=True)

def test_get_named_inputs():
    step1 = Step(a=1, name='Step1')
    step2 = Step(b=1, inputs=[Step(c=1, inputs=[step1, Step(d=1)])])
    assert step2.get_named_steps() == {'Step1': step1}

def test_get_named_inputs2():
    step1 = Step(a=1, name='Step1')
    step2 = Step(b=1, name='Step2', inputs=[Step(c=1, inputs=[step1, Step(d=1)])])
    assert step2.get_named_steps() == {'Step2': step2, 'Step1': step1}

def test_get_named_arguments():
    step1 = Step(a=1, name='Step1')
    step2 = Step(b=1, inputs=[Step(c=1, inputs=[step1, Step(d=1)])])
    assert step2.get_named_arguments() == {('Step1', 'a'): 1}
