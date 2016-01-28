from drain import data,step
import tempfile
import os

def setup_module(module):
    tmpdir = tempfile.mkdtemp()
    step.BASEDIR=tmpdir
    step.configure_yaml()

def test_to_hdf():
   d = data.ClassificationData()
   h = data.ToHDF(inputs=[d], target=True)

   h.setup_dump()
   step.run(h)

   r0, r1 = h.get_result(), d.get_result()

   for key in r1.keys():
       assert r0[key].equals(r1[key])
       
