from drain import data
import os

def test_read_write(tmpdir):
    cd = data.ClassificationData()
    cd.read()
    cd.write(tmpdir.dirname)

    cd2 = data.ClassificationData()
    cd2.read(tmpdir.dirname)

    assert(cd.df.equals(cd2.df))

def test_prefix_column():
    assert data.prefix_column('level_id', 'column') == 'st_level_column'
    assert data.prefix_column('level_id', 'column', prefix='prefix') == 'st_level_prefix_column'
    assert data.prefix_column('level_id', 'column', prefix='prefix', delta=1) == 'st_level_1y_prefix_column'
    assert data.prefix_column('level_id', 'column', prefix='prefix', delta=-1) == 'st_level_all_prefix_column'
