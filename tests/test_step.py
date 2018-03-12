from drain.step import *
from drain import step
import numpy as np
import tempfile

class Scalar(Step):
    def __init__(self, value):
        Step.__init__(self, value=value)

    def run(self):
        return self.value


class Add(Step):
    def run(self, *values):
        return sum(values)


class Divide(Step):
    def run(self, numerator, denominator):
        return numerator / denominator


def test_run(drain_setup):
    s = Add(inputs = [Scalar(value=value) for value in range(1,10)])
    s.execute()
    assert s.result == 45

def test_run_map_results():
    s = Divide(inputs=[MapResults(
            inputs=[Scalar(value=1), Scalar(value=2)], 
            mapping=['denominator', 'numerator'])])
    s.execute()
    assert s.result == 2

def test_map_results():
    a = Scalar(1)
    b = Scalar(2)

    a.execute()
    b.execute()

    c = MapResults(inputs=[a,b], mapping=['a','b'])
    c.execute()

    assert c.result == {'a':1, 'b':2}

def test_map_results_dict():
    a = Scalar(1)
    b = Scalar(2)

    a.execute()
    b.execute()

    c = MapResults(inputs=[a,b], mapping=['a','b'])
    c.execute()

    assert c.result == {'a':1, 'b':2}

def test_map_results_list():
    a = Scalar([1,2])
    a.execute()

    c = MapResults(inputs=[a], mapping=[['a','b']])
    c.execute()
    assert c.result == {'a':1, 'b':2}

def test_map_results_default():
    a = Scalar([1,2])
    a.execute()

    c = MapResults(inputs=[a], mapping=[MapResults.DEFAULT])
    c.execute()
    assert c.result == [1,2]


class DumpStep(Step):
    def __init__(self, n, n_df, return_list):
        # number of objects to return and number of them to be dataframes
        # and whether to use a list or dict
        if n_df == None:
            n_df = n

        Step.__init__(self, n=n, n_df=n_df, return_list=return_list)
        self.target = True

    def run(self):
        l = ['a']*self.n + [pd.DataFrame(np.arange(5))]*self.n_df
        if len(l) == 1:
            return l[0]

        if self.return_list:
            return l
        else:
            d = {'k'+str(k):v for k,v in zip(range(len(l)), l)}
            return d

def test_dump_joblib():
    t = DumpStep(n=10, n_df=0, return_list=False)

    t.execute()
    r = t.result
    t.dump()
    t.load()
    assert r == t.result

def test_dump_hdf_single():
    t = DumpStep(n=0, n_df=1, return_list=False)

    t.execute()
    r = t.result
    t.dump()
    t.load()
    assert r.equals(t.result)

def test_dump_hdf_list():
    t = DumpStep(n=0, n_df=5, return_list=True)

    t.execute()
    r = t.result
    t.dump()
    t.load()

    for a,b in zip(r,t.result):
        assert a.equals(b)

def test_dump_hdf_dict():
    t = DumpStep(n=0, n_df=5, return_list=False)

    t.execute()
    r = t.result
    t.dump()
    t.load()

    for k in r:
        assert r[k].equals(t.result[k])

def test_expand_inputs():
    s = Step(a=1, b={'c':Step(c=2)})
    assert step._expand_inputs(s) == {s, Step(c=2)}

def test_collect_kwargs():
    s = Step(a=1, b={'c':Step(c=2)})
    s.name = 'Step2'
    assert step._collect_kwargs(s) == {
            'Step2': {'a':1},
            'Step': {'c':2}
    }

def test_call():
    s = Call(pd.DataFrame, data=[0,10])
    t = Call(s, 'mean')
    t.execute()

    assert t.result.values[0] == 5

def test_get_reslt():
    c = Call(dict, a=1)
    d = GetItem(c, 'a')
    d.execute()

    assert d.result == 1
