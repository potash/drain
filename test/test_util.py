from drain.util import *

def test_dict_product():
    assert dict_product({1:[2,3], 4:[5,6]}) == [{1: 2, 4: 5}, {1: 2, 4: 6}, {1: 3, 4: 5}, {1: 3, 4: 6}]
def test_dict_product_empty():
    assert dict_product({}) == [{}]
