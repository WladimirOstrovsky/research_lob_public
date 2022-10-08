"""
Order cancellation threshold
"""
ID = 6
NAME = 'Tau'
PARAM_NUM = 1
KEYS = ['tau']
SYMMETRIC = True
FUNC_1 = None
DIM = 1


DICT_SENSITIVITY = {
    0: {
        'tau': 25,
    },
    1: {
        'tau': 50,
    },
    2: {
        'tau': 100,
    },
    3: {
        'tau': 150,
    },
    4: {
        'tau': 250,
    },
    5: {
        'tau': 350,
    },
    6: {
        'tau': 500,
    }
}
