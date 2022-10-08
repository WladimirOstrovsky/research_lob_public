"""
Heterogeneity (variance)
"""
ID = 5
NAME = 'Heterogeneity'
PARAM_NUM = 1
KEYS = ['sigma']
SYMMETRIC = True
FUNC_1 = None
DIM = 1


DICT_SENSITIVITY = {
    0: {
        'sigma': 0.00010,
    },
    1: {
        'sigma': 0.00025,
    },
    2: {
        'sigma': 0.00050,
    },
    3: {
        'sigma': 0.00100,
    },
    4: {
        'sigma': 0.00150,
    },
    5: {
        'sigma': 0.00250,
    },
    6: {
        'sigma': 0.00500,
    },
    7: {
        'sigma': 0.00750,
    },
    8: {
        'sigma': 0.01000,
    }
}
