"""
MM: inventory skew
"""
import numpy as np
import itertools

ID = 17
NAME = 'Inventory Skew (asym.)'
PARAM_NUM = 2
KEYS = ['inventory_skew_bid', 'inventory_skew_ask']
SYMMETRIC = False
FUNC_1 = lambda x: abs(x).mean(axis=1)
DIM = 2


# DICT_SENSITIVITY = {
#     0: {
#         'inventory_skew_bid': -0.10,
#         'inventory_skew_ask': -0.10
#     },
#     1: {
#         'inventory_skew_bid': -0.10,
#         'inventory_skew_ask': -0.25
#     },
#     2: {
#         'inventory_skew_bid': -0.10,
#         'inventory_skew_ask': -0.5
#     },
#     3: {
#         'inventory_skew_bid': -0.10,
#         'inventory_skew_ask': -0.75
#     },
#     4: {
#         'inventory_skew_bid': -0.10,
#         'inventory_skew_ask': -1.0
#     },
#     5: {
#         'inventory_skew_bid': -0.10,
#         'inventory_skew_ask': -1.25
#     },
#     #
#     6: {
#         'inventory_skew_bid': -0.25,
#         'inventory_skew_ask': -0.10
#     },
#     7: {
#         'inventory_skew_bid': -0.25,
#         'inventory_skew_ask': -0.25
#     },
#     8: {
#         'inventory_skew_bid': -0.25,
#         'inventory_skew_ask': -0.5
#     },
#     9: {
#         'inventory_skew_bid': -0.25,
#         'inventory_skew_ask': -0.75
#     },
#     10: {
#         'inventory_skew_bid': -0.25,
#         'inventory_skew_ask': -1.0
#     },
#     11: {
#         'inventory_skew_bid': -0.25,
#         'inventory_skew_ask': -1.25
#     },
# 
#     #
#     12: {
#         'inventory_skew_bid': -0.50,
#         'inventory_skew_ask': -0.10
#     },
#     13: {
#         'inventory_skew_bid': -0.50,
#         'inventory_skew_ask': -0.25
#     },
#     14: {
#         'inventory_skew_bid': -0.50,
#         'inventory_skew_ask': -0.5
#     },
#     15: {
#         'inventory_skew_bid': -0.50,
#         'inventory_skew_ask': -0.75
#     },
#     16: {
#         'inventory_skew_bid': -0.50,
#         'inventory_skew_ask': -1.0
#     },
#     17: {
#         'inventory_skew_bid': -0.50,
#         'inventory_skew_ask': -1.25
#     },
#     #
#     18: {
#         'inventory_skew_bid': -0.75,
#         'inventory_skew_ask': -0.10
#     },
#     19: {
#         'inventory_skew_bid': -0.75,
#         'inventory_skew_ask': -0.25
#     },
#     20: {
#         'inventory_skew_bid': -0.75,
#         'inventory_skew_ask': -0.5
#     },
#     21: {
#         'inventory_skew_bid': -0.75,
#         'inventory_skew_ask': -0.75
#     },
#     22: {
#         'inventory_skew_bid': -0.75,
#         'inventory_skew_ask': -1.0
#     },
#     23: {
#         'inventory_skew_bid': -0.75,
#         'inventory_skew_ask': -1.25
#     },
#     #
#     24: {
#         'inventory_skew_bid': -1.0,
#         'inventory_skew_ask': -0.10
#     },
#     25: {
#         'inventory_skew_bid': -1.0,
#         'inventory_skew_ask': -0.25
#     },
#     26: {
#         'inventory_skew_bid': -1.0,
#         'inventory_skew_ask': -0.5
#     },
#     27: {
#         'inventory_skew_bid': -1.0,
#         'inventory_skew_ask': -0.75
#     },
#     28: {
#         'inventory_skew_bid': -1.0,
#         'inventory_skew_ask': -1.0
#     },
#     29: {
#         'inventory_skew_bid': -1.0,
#         'inventory_skew_ask': -1.25
#     },
#     #
#     30: {
#         'inventory_skew_bid': -1.25,
#         'inventory_skew_ask': -0.10
#     },
#     31: {
#         'inventory_skew_bid': -1.25,
#         'inventory_skew_ask': -0.25
#     },
#     31: {
#         'inventory_skew_bid': -1.25,
#         'inventory_skew_ask': -0.5
#     },
#     32: {
#         'inventory_skew_bid': -1.25,
#         'inventory_skew_ask': -0.75
#     },
#     33: {
#         'inventory_skew_bid': -1.25,
#         'inventory_skew_ask': -1.0
#     },
#     34: {
#         'inventory_skew_bid': -1.25,
#         'inventory_skew_ask': -1.25
#     },
# }

_MAX_ORDER_SIZE = 500

# DICT_SENSITIVITY = {
#     0: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5
#     },
#     1: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1
#     },
#     2: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2
#     },
#     3: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3
#     },
#     4: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4
#     },
#     5: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5
#     },
#     #
#     6: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5
#     },
#     7: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1
#     },
#     8: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2
#     },
#     9: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3
#     },
#     10: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4
#     },
#     11: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5
#     },

#     #
#     12: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5
#     },
#     13: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1
#     },
#     14: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2
#     },
#     15: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3
#     },
#     16: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4
#     },
#     17: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5
#     },
#     #
#     18: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5
#     },
#     19: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1
#     },
#     20: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2
#     },
#     21: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3
#     },
#     22: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4
#     },
#     23: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5
#     },
#     #
#     24: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5
#     },
#     25: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1
#     },
#     26: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2
#     },
#     27: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3
#     },
#     28: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4
#     },
#     29: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5
#     },
#     #
#     30: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 0.5
#     },
#     31: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 1
#     },
#     32: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 2
#     },
#     33: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 3
#     },
#     34: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 4
#     },
#     35: {
#         'inventory_skew_bid': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5,
#         'inventory_skew_ask': np.round(np.log(1 / _MAX_ORDER_SIZE) / _MAX_ORDER_SIZE, 6) / 5
#     },
# }

l = np.linspace(-0.005, -0.05, 5)
l_list = list(itertools.product(l, l))

DICT_SENSITIVITY = {}
for idx, i in enumerate(l_list):
    DICT_SENSITIVITY.update({idx: {
        'inventory_skew_ask': i[0],
        'inventory_skew_bid': i[1]
    }})
