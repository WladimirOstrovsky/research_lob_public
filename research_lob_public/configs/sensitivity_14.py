"""
MM: inventory skew
"""
ID = 14
NAME = 'Inventory Skew'
PARAM_NUM = 2
KEYS = ['inventory_skew_bid', 'inventory_skew_ask']
SYMMETRIC = True  # TODO can also be not symmetric
FUNC_1 = lambda x: abs(x).mean(axis=1)
DIM = 2


DICT_SENSITIVITY = {
    0: {
        'inventory_skew_bid': -0.2,
        'inventory_skew_ask': -0.2
    },
    1: {
        'inventory_skew_bid': -0.4,
        'inventory_skew_ask': -0.4
    },
    2: {
        'inventory_skew_bid': -0.8,
        'inventory_skew_ask': -0.8
    },
    3: {
        'inventory_skew_bid': -1.5,
        'inventory_skew_ask': -1.5
    },
    4: {
        'inventory_skew_bid': -2.,
        'inventory_skew_ask': -2.
    },
    5: {
        'inventory_skew_bid': -2.5,
        'inventory_skew_ask': -2.5
    },
    6: {
        'inventory_skew_bid': -3.5,
        'inventory_skew_ask': -3.5
    },
    7: {
        'inventory_skew_bid': -4.5,
        'inventory_skew_ask': -4.5
    },
    8: {
        'inventory_skew_bid': -5.5,
        'inventory_skew_ask': -5.5
    }
}
