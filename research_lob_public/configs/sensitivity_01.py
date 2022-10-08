"""
Allocation agent-types
"""
ID = 1
NAME = 'Agent Allocation'
PARAM_NUM = 2
KEYS = ['p_fundamentalist', 'p_chartist']
SYMMETRIC = False
FUNC_1 = None
DIM = 2


DICT_SENSITIVITY = {
    # 0: {
    #     'p_fundamentalist': 0.5,
    #     'p_chartist': 0.5
    # },

    0: {
        'p_fundamentalist': 0.1,
        'p_chartist': 0.8
    },
    1: {
        'p_fundamentalist': 0.2,
        'p_chartist': 0.7
    },
    2: {
        'p_fundamentalist': 0.3,
        'p_chartist': 0.6
    },
    3: {
        'p_fundamentalist': 0.4,
        'p_chartist': 0.5
    },
    4: {
        'p_fundamentalist': 0.5,
        'p_chartist': 0.4
    },
    5: {
        'p_fundamentalist': 0.6,
        'p_chartist': 0.3
    },
    6: {
        'p_fundamentalist': 0.7,
        'p_chartist': 0.2
    },
    7: {
        'p_fundamentalist': 0.8,
        'p_chartist': 0.1
    },

    #
    8: {
        'p_fundamentalist': 0.5,
        'p_chartist': 0.5
    },
    9: {
        'p_fundamentalist': 0.4,
        'p_chartist': 0.4
    },
    10: {
        'p_fundamentalist': 0.3,
        'p_chartist': 0.3
    },
    11: {
        'p_fundamentalist': 0.2,
        'p_chartist': 0.2
    },
    12: {
        'p_fundamentalist': 0.1,
        'p_chartist': 0.1
    },

    #
    13: {
        'p_fundamentalist': 0.1,
        'p_chartist': 0.2
    },
    14: {
        'p_fundamentalist': 0.1,
        'p_chartist': 0.3
    },
    15: {
        'p_fundamentalist': 0.1,
        'p_chartist': 0.4
    },
    16: {
        'p_fundamentalist': 0.1,
        'p_chartist': 0.5
    },

    #
    17: {
        'p_fundamentalist': 0.2,
        'p_chartist': 0.1
    },
    18: {
        'p_fundamentalist': 0.3,
        'p_chartist': 0.1
    },
    19: {
        'p_fundamentalist': 0.4,
        'p_chartist': 0.1
    },
    20: {
        'p_fundamentalist': 0.5,
        'p_chartist': 0.1
    }
}
