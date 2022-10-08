"""
MM: arrival intensity
"""
ID = 13
NAME = 'Arrival Intensity'
PARAM_NUM = 1
KEYS = ['arrival_intensity_mm']
SYMMETRIC = True
FUNC_1 = None
DIM = 1


DICT_SENSITIVITY = {
    0: {
        'arrival_intensity_mm': 0.1,
    },
    1: {
        'arrival_intensity_mm': 0.2,
    },
    2: {
        'arrival_intensity_mm': 0.4,
    },
    3: {
        'arrival_intensity_mm': 0.6,
    },
    4: {
        'arrival_intensity_mm': 1.,
    },
    5: {
        'arrival_intensity_mm': 1.4,
    },
    6: {
        'arrival_intensity_mm': 1.8,
    },
    7: {
        'arrival_intensity_mm': 2.5,
    },
    8: {
        'arrival_intensity_mm': 3.5,
    },
    9: {
        'arrival_intensity_mm': 5.,
    }
}
