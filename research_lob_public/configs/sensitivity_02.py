"""
Share allocation
"""
import numpy as np
ID = 2
NAME = 'Share Initiation'
PARAM_NUM = 2
KEYS = ['Smin', 'Smax']
SYMMETRIC = True
FUNC_1 = lambda x: abs(x).mean(axis=1)
DIM = 2


DICT_SENSITIVITY = {
    0: {
        'Smin': -10,
        'Smax': 10
    },
    1: {
        'Smin': -20,
        'Smax': 20
    },
    2: {
        'Smin': -30,
        'Smax': 30
    },
    3: {
        'Smin': -40,
        'Smax': 40
    },
    4: {
        'Smin': -50,
        'Smax': 50
    },
    5: {
        'Smin': -75,
        'Smax': 75
    },
    6: {
        'Smin': -100,
        'Smax': 100
    },
    7: {
        'Smin': -125,
        'Smax': 125
    },
    8: {
        'Smin': -150,
        'Smax': 150
    },
    9: {
        'Smin': -175,
        'Smax': 175
    },
    10: {
        'Smin': -200,
        'Smax': 200
    }
}

