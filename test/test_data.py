from drain import data
import os

def test_read_write(tmpdir):
    filename = os.path.join(tmpdir.dirname,'cd.pkl')
    cd = data.ClassificationData()
    cd.read()
    cd.write(filename)

    cd2 = data.ClassificationData()
    cd2.read(filename)

    assert(cd.df.equals(cd2.df))
