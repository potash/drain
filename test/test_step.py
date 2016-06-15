from drain.step import *
from drain import step
import tempfile

def test_run(drain_setup):
    s = Add(inputs = [Scalar(value=value) for value in range(1,10)])
    assert run(s) == 45

def test_run_inputs_mapping():
    s = Divide(inputs = [Scalar(value=1), Scalar(value=2)], 
            inputs_mapping=['denominator', 'numerator'])
    assert run(s) == 2


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
    assert step2.named_steps == {'Step1': step1}

def test_get_named_inputs2():
    step1 = Step(a=1, name='Step1')
    step2 = Step(b=1, name='Step2', inputs=[Step(c=1, inputs=[step1, Step(d=1)])])
    assert step2.named_steps == {'Step2': step2, 'Step1': step1}

def test_get_named_arguments():
    step1 = Step(a=1, name='Step1')
    step2 = Step(b=1, inputs=[Step(c=1, inputs=[step1, Step(d=1)])])
    assert step2.named_arguments == {('Step1', 'a'): 1}

class DumpStep(Step):
    def __init__(self, n, n_df, return_list, **kwargs):
        # number of objects to return and number of them to be dataframes
        # and whether to use a list or dict
        if n_df == None:
            n_df = n
        Step.__init__(self, n=n, n_df=n_df, return_list=return_list, **kwargs)

    def run(self):
        l = ['a']*self.n + [pd.DataFrame(range(5))]*self.n_df
        if len(l) == 1:
            return l[0]

        if self.return_list:
            return l
        else:
            d = {'k'+str(k):v for k,v in zip(range(len(l)), l)}
            return d

def test_dump_joblib():
    t = DumpStep(n=10, n_df=0, return_list=False, target=True)
    run(t)
    t.dump()
    t.load()
    print t.get_result()

def test_dump_hdf_single():
    t = DumpStep(n=0, n_df=1, return_list=False, target=True)
    run(t)
    t.dump()
    t.load()
    print t.get_result()


def test_dump_hdf_list():
    t = DumpStep(n=0, n_df=5, return_list=True, target=True)
    run(t)
    t.dump()
    t.load()
    print t.get_result()

def test_dump_hdf_dict():
    t = DumpStep(n=0, n_df=5, return_list=False, target=True)
    run(t)
    t.dump()
    t.load()
    print t.get_result()
