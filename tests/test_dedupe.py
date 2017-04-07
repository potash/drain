from drain.dedupe import *

edges = pd.DataFrame([[1,2],[4,5], [3,2]])

def test_follow():
    assert follow(1, edges, directed=False) == {1,2,3}

def test_follow_directed():
    assert follow(1, edges, directed=True) == {1,2}

def test_get_components():
    assert get_components(edges) == [{1,2,3}, {4,5}]

def test_components_to_df():
    assert components_to_df([{1,2,3},{4,5}]).equals(pd.DataFrame([[1,1],[1,2],[1,3],[4,4],[4,5]], columns=['id1','id2']))
