from drain.util import *

def test_dict_product():
    assert dict_product({1:[2,3], 4:[5,6]}) == [{1: 2, 4: 5}, {1: 2, 4: 6}, {1: 3, 4: 5}, {1: 3, 4: 6}]

def test_dict_product_empty():
    assert dict_product({}) == [{}]

def test_dict_product_kwargs():
    assert dict_product({'a':1},b=[2,3]) == [{'a':1, 'b':2}, {'a':1, 'b':3}]

def test_list_expand_basecase():
    assert list(list_expand([(1,2)])) == [(1,2)]

def test_list_expand_single():
    assert list(list_expand({1:[2,3], 4:[4,5]})) == [(1,2), (1,3), (4,4), (4,5)]

def test_list_expand_multiple():
    assert list(list_expand({1:[2,3,4], 5:{7:[0]}})) == [(1, 2), (1,3), (1,4), (5, 7, 0)]

def test_dict_expand():
    assert dict_expand({1:2, 3:4}) == {1:2, 3:4}

def test_dict_expand_deep():
    assert dict_expand({1:2, 3:{4:{5:6}}}) == {1:2, (3,4,5):6}

def test_dict_diff_empty():
    assert dict_diff([{}]) == [{}]

def test_dict_diff_single():
    assert dict_diff([{0:1}, {0:1, 2:3}]) == [{}, {2:3}]

def test_dict_diff_double():
    assert dict_diff([{}, {1:2, 3:4}, {3:4}]) == [{}, {1:2, 3:4}, {3:4}]

def test_is_instance_collection_empty():
    assert not is_instance_collection([], int)

def test_is_instance_collection_list():
    assert is_instance_collection([pd.DataFrame()], pd.DataFrame)

def test_is_instance_collection_dict():
    assert is_instance_collection({'a':pd.DataFrame()}, pd.DataFrame)

def test_is_instance_collection_multiple():
    assert is_instance_collection([pd.DataFrame(), pd.Series()], [pd.DataFrame, pd.Series])

def test_is_instance_collection_false():
    assert not is_instance_collection([pd.DataFrame(), 1], pd.DataFrame)


