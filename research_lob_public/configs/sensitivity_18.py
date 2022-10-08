"""
MM: maximum order size
"""
ID = 18
NAME = 'Maximum Order Size (asym.)'
PARAM_NUM = 2
KEYS = ['order_size_bid_max', 'order_size_ask_max']
SYMMETRIC = False
FUNC_1 = lambda x: x.mean(axis=1)
DIM = 2

_MULT = 100

DICT_SENSITIVITY = {

    0: {
        'order_size_bid_max': 25 * _MULT,
        'order_size_ask_max': 25 * _MULT
    },
    1: {
        'order_size_bid_max': 25 * _MULT,
        'order_size_ask_max': 50 * _MULT
    },
    2: {
        'order_size_bid_max': 25 * _MULT,
        'order_size_ask_max': 75 * _MULT
    },
    3: {
        'order_size_bid_max': 25 * _MULT,
        'order_size_ask_max': 100 * _MULT
    },
    4: {
        'order_size_bid_max': 25 * _MULT,
        'order_size_ask_max': 125 * _MULT
    },
    #
    5: {
        'order_size_bid_max': 50 * _MULT,
        'order_size_ask_max': 25 * _MULT
    },
    6: {
        'order_size_bid_max': 50 * _MULT,
        'order_size_ask_max': 50 * _MULT
    },
    7: {
        'order_size_bid_max': 50 * _MULT,
        'order_size_ask_max': 75 * _MULT
    },
    8: {
        'order_size_bid_max': 50 * _MULT,
        'order_size_ask_max': 100 * _MULT
    },
    9: {
        'order_size_bid_max': 50 * _MULT,
        'order_size_ask_max': 125 * _MULT
    },
    #
    10: {
        'order_size_bid_max': 75 * _MULT,
        'order_size_ask_max': 25 * _MULT
    },
    11: {
        'order_size_bid_max': 75 * _MULT,
        'order_size_ask_max': 50 * _MULT
    },
    12: {
        'order_size_bid_max': 75 * _MULT,
        'order_size_ask_max': 75 * _MULT
    },
    13: {
        'order_size_bid_max': 75 * _MULT,
        'order_size_ask_max': 100 * _MULT
    },
    14: {
        'order_size_bid_max': 75 * _MULT,
        'order_size_ask_max': 125 * _MULT
    },
    #
    15: {
        'order_size_bid_max': 100 * _MULT,
        'order_size_ask_max': 25 * _MULT
    },
    16: {
        'order_size_bid_max': 100 * _MULT,
        'order_size_ask_max': 50 * _MULT
    },
    17: {
        'order_size_bid_max': 100 * _MULT,
        'order_size_ask_max': 75 * _MULT
    },
    18: {
        'order_size_bid_max': 100 * _MULT,
        'order_size_ask_max': 100 * _MULT
    },
    19: {
        'order_size_bid_max': 100 * _MULT,
        'order_size_ask_max': 125 * _MULT
    },
    #
    20: {
        'order_size_bid_max': 125 * _MULT,
        'order_size_ask_max': 25 * _MULT
    },
    21: {
        'order_size_bid_max': 125 * _MULT,
        'order_size_ask_max': 50 * _MULT
    },
    22: {
        'order_size_bid_max': 125 * _MULT,
        'order_size_ask_max': 75 * _MULT
    },
    23: {
        'order_size_bid_max': 125 * _MULT,
        'order_size_ask_max': 100 * _MULT
    },
    24: {
        'order_size_bid_max': 125 * _MULT,
        'order_size_ask_max': 125 * _MULT
    },

}
