"""
Cash allocation
"""
ID = 3
NAME = 'Cash Initiation'
PARAM_NUM = 2
KEYS = ['Cmin', 'Cmax']
SYMMETRIC = True
FUNC_1 = lambda x: x.mean(axis=1)
DIM = 2


DICT_SENSITIVITY = {
    0: {
        'Cmin': 0,
        'Cmax': 50
    },
    1: {
        'Cmin': 20,
        'Cmax': 100
    },
    2: {
        'Cmin': 40,
        'Cmax': 100
    },
    3: {
        'Cmin': 80,
        'Cmax': 100
    },
    4: {
        'Cmin': 40,
        'Cmax': 200
    },
    5: {
        'Cmin': 100,
        'Cmax': 200
    },
    6: {
        'Cmin': 150,
        'Cmax': 200
    },
    7: {
        'Cmin': 200,
        'Cmax': 300
    },
    8: {
        'Cmin': 300,
        'Cmax': 400
    },
    9: {
        'Cmin': 400,
        'Cmax': 500
    }
}
