"""
Risk preferences - agents
"""
ID = 11
NAME = 'Risk Aversion'
PARAM_NUM = 1
KEYS = ['risk_aversion_agents']
SYMMETRIC = True
FUNC_1 = None
DIM = 1


DICT_SENSITIVITY = {
    # 0: {
    #     'risk_aversion_agents': 20.,
    # },
    0: {
        'risk_aversion_agents': 1.,
    },
    1: {
        'risk_aversion_agents': 2.,
    },
    2: {
        'risk_aversion_agents': 3.,
    },
    3: {
        'risk_aversion_agents': 5.,
    },
    4: {
        'risk_aversion_agents': 7.,
    },
    5: {
        'risk_aversion_agents': 10.,
    },
    6: {
        'risk_aversion_agents': 15.,
    },
    7: {
        'risk_aversion_agents': 20.,
    }
}
