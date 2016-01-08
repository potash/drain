from drain.drain import *
import tempfile

def setup_module(module):
    tmpdir = tempfile.mkdtemp()
    initialize(tmpdir)

def test_argument_product():
    assert argument_product({'a' : ArgumentCollection([1,2])}) == [{'a': 1}, {'a': 2}]

def test_argument_product_deep():
    assert argument_product({'a' : {'b': ArgumentCollection([1,2])} }) == [{'a': {'b': 1}}, {'a': {'b': 2}}]

def test_step_product():
    assert step_product(StepTemplate(a={'b': ArgumentCollection([1,2])})) == [StepTemplate(a={'b': 1}), StepTemplate(a={'b': 2})]

def test_parallel():
    assert parallel(StepTemplate(a=1), StepTemplate(b=1)) == [StepTemplate(a=1), StepTemplate(b=1)]

def test_parallel_multiple():
    assert parallel([StepTemplate(a=1), StepTemplate(b=1)], [StepTemplate(c=1)]) == [StepTemplate(a=1), StepTemplate(b=1), StepTemplate(c=1)]

def test_search():
    assert search(StepTemplate(a=ArgumentCollection([1,2]))) == [StepTemplate(a=1), StepTemplate(a=2)]

def test_search_multiple():
    assert search(StepTemplate(a=ArgumentCollection([1,2])), StepTemplate(b=ArgumentCollection([1,2]))) == [StepTemplate(a=1), StepTemplate(a=2), StepTemplate(b=1), StepTemplate(b=2)]

def test_product():
    assert product([StepTemplate(a=1), StepTemplate(a=2)], StepTemplate(b=1)) == [(StepTemplate(a=1), StepTemplate(b=1)), (StepTemplate(a=2), StepTemplate(b=1))]
    
def test_serial():
    assert serial([StepTemplate(a=1), StepTemplate(a=2)], [StepTemplate(b=1), StepTemplate(b=2)]) == [StepTemplate(inputs=[StepTemplate(a=1), StepTemplate(a=2)],b=1), StepTemplate(inputs=[StepTemplate(a=1), StepTemplate(a=2)],b=2)]
    
def test_run():
    step = yaml.load("""
    !serial 
      - !search
        - !step:drain.drain..Add 
            value : !range [1,10]
      - !step:drain.drain.Add {value : 3, target : True}
    """)
    step = step[0].construct()
    run(step)
    assert step.result == 48

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
    assert get_drake_data_helper(steps) == ({},{},{Step(value=1): set()})

def test_drake_data_helper2():
    steps = [Step(value=1, inputs=[Step(value=2, target=True)])]
    assert get_drake_data_helper(steps) == ({Step(value=2): set()}, {Step(value=2): {steps[0]}}, {})

def test_drake_data_helper3():
    steps = [Step(value=1, target=True)]
    assert get_drake_data_helper(steps) == ({Step(value=1): set()}, {}, {})

def test_drake_data_helper4():
    steps = [Step(value=1, inputs=[Step(value=2, target=True), Step(value=3, target=True)])]
    assert get_drake_data_helper(steps) == (
            {Step(value=2):set(), Step(value=3):set()},
            {},
            {steps[0]: set([Step(value=2), Step(value=3)])}
    )

# when "same" step with and without target, runs both...
# should it?
def test_drake_data_helper5():
    steps = [Step(value=1), Step(value=1, target=True)]
    assert get_drake_data_helper(steps) == ({Step(value=1): set()}, {}, {Step(value=1):set()})

def test_drake_data_helper6():
    steps = [Step(value=1, inputs=[Step(value=3, target=True)]), Step(value=2,  inputs=[Step(value=3, target=True)])]
    assert get_drake_data_helper(steps) == (
            {Step(value=3) : set()},
            {Step(value=3): set(steps)},
            {}
    )

# no output step
def test_drake_data():
    step = Step(a=1)
    data = get_drake_data([step])
    assert data == [(set(), None, set([Step(a=1)]))]

# no output step with no target input
def test_drake_data2():
    step = Step(a=1, inputs=[Step(b=1)])
    data = get_drake_data([step])
    assert data == [(set([]), None, set([Step(a=1,inputs=[Step(b=1)])]))]

# no output step with target
def test_drake_data3():
    step = Step(a=1, inputs=[Step(b=1, target=True)])
    data = get_drake_data([step])
    assert data == [(set([]), Step(b=1), set([Step(a=1,inputs=[Step(b=1)])]))]

# multiple no output steps on single target
def test_drake_data4():
    steps = [Step(a=1, inputs=[Step(b=1, target=True)]),
             Step(a=2, inputs=[Step(b=1, target=True)])]
    data = get_drake_data(steps)
    assert data == [(set([]), Step(b=1), set([Step(a=1,inputs=[Step(b=1)]), Step(a=2,inputs=[Step(b=1)])]))]

def test_drakefile():
    steps = [Step(a=1, inputs=[Step(b=1, target=True)]),
             Step(a=2, inputs=[Step(b=1, target=True)])]
    print to_drakefile(steps, preview=True)
