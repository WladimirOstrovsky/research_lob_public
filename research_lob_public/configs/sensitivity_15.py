"""
MM: maximum order size
"""
ID = 15
NAME = 'Maximum Order Size'
PARAM_NUM = 2
KEYS = ['order_size_bid_max', 'order_size_ask_max']
SYMMETRIC = True  # TODO can also be not symmetric
FUNC_1 = lambda x: x.mean(axis=1)
DIM = 2


DICT_SENSITIVITY = {
    0: {
        'order_size_bid_max': 25,
        'order_size_ask_max': 25
    },
    # 1: {
    #     'order_size_bid_max': 50,
    #     'order_size_ask_max': 50
    # },
    # 2: {
    #     'order_size_bid_max': 100,
    #     'order_size_ask_max': 100
    # },
    # 3: {
    #     'order_size_bid_max': 150,
    #     'order_size_ask_max': 150
    # },
    # 4: {
    #     'order_size_bid_max': 200,
    #     'order_size_ask_max': 200
    # },
    # 5: {
    #     'order_size_bid_max': 300,
    #     'order_size_ask_max': 300
    # },
    # 6: {
    #     'order_size_bid_max': 400,
    #     'order_size_ask_max': 400
    # },
    # 7: {
    #     'order_size_bid_max': 500,
    #     'order_size_ask_max': 500
    # }
}
