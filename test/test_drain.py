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
    assert step_product(StepTemplate(a={'b': ArgumentCollection([1,2])})) == [Step(a={'b': 1}), Step(a={'b': 2})]

def test_parallel():
    assert parallel(Step(a=1), Step(b=1)) == [Step(a=1), Step(b=1)]

def test_parallel_multiple():
    assert parallel([Step(a=1), Step(b=1)], [Step(c=1)]) == [Step(a=1), Step(b=1), Step(c=1)]

def test_search():
    assert search(StepTemplate(a=ArgumentCollection([1,2]))) == [Step(a=1), Step(a=2)]

def test_search_multiple():
    assert search(StepTemplate(a=ArgumentCollection([1,2])), StepTemplate(b=ArgumentCollection([1,2]))) == [Step(a=1), Step(a=2), Step(b=1), Step(b=2)]

def test_product():
    assert product([Step(a=1), Step(a=2)], Step(b=1)) == [(Step(a=1), Step(b=1)), (Step(a=2), Step(b=1))]
    
def test_serial():
    assert serial([StepTemplate(a=1), StepTemplate(a=2)], [StepTemplate(b=1), StepTemplate(b=2)]) == [Step(inputs=[Step(a=1), Step(a=2)],b=1), Step(inputs=[Step(a=1), Step(a=2)],b=2)]
    
def test_run():
    step = yaml.load("""
    !serial 
      - !search
        - !step:drain.drain..Add 
            value : !range [1,10]
      - !step:drain.drain.Add {value : 3, target : True}
    """)[0]
    run(step)
    assert step.output == 48

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

def test_drake_helper():
    steps = [Step(value=1)]
    assert to_drake_helper(steps) == ({},{},{Step(value=1): set()})

def test_drake_helper2():
    steps = [Step(value=1, inputs=[Step(value=2, target=True)])]
    assert to_drake_helper(steps) == ({Step(value=2): set()}, {Step(value=2): {steps[0]}}, {})

def test_drake_helper3():
    steps = [Step(value=1, target=True)]
    assert to_drake_helper(steps) == ({Step(value=1): set()}, {}, {})

def test_drake_helper4():
    steps = [Step(value=1, inputs=[Step(value=2, target=True), Step(value=3, target=True)])]
    assert to_drake_helper(steps) == (
            {Step(value=2):set(), Step(value=3):set()},
            {},
            {steps[0]: set([Step(value=2), Step(value=3)])}
    )

# when "same" step with and without target, runs both...
# should it?
def test_drake_helper5():
    steps = [Step(value=1), Step(value=1, target=True)]
    assert to_drake_helper(steps) == ({Step(value=1): set()}, {}, {Step(value=1):set()})

