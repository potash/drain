from drain.util import *

def test_dict_product():
    assert dict_product({1:[2,3], 4:[5,6]}) == [{1: 2, 4: 5}, {1: 2, 4: 6}, {1: 3, 4: 5}, {1: 3, 4: 6}]
def test_dict_product_empty():
    assert dict_product({}) == [{}]

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

def test_dict_diff():
    assert dict_diff([{}, {}]) == [{}, {}]

def test_dict_diff():
    assert dict_diff([{}, {1:2}]) == [{}, {1:2}]

