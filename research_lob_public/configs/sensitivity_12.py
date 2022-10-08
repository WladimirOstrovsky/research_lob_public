"""
MM: risk aversion
"""
ID = 12
NAME = 'Risk Aversion MM'
PARAM_NUM = 1
KEYS = ['risk_aversion_mm']
SYMMETRIC = True
FUNC_1 = None
DIM = 1


DICT_SENSITIVITY = {
    0: {
        'risk_aversion_mm': 1.,
    },
    1: {
        'risk_aversion_mm': 2.,
    },
    2: {
        'risk_aversion_mm': 3.,
    },
    3: {
        'risk_aversion_mm': 5.,
    },
    4: {
        'risk_aversion_mm': 7.,
    },
    5: {
        'risk_aversion_mm': 10.,
    },
    6: {
        'risk_aversion_mm': 15.,
    },
    7: {
        'risk_aversion_mm': 20.,
    }
}
