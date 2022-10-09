"""
base config for simulations
"""
import numpy as np
import plotly.io as pio
pio.renderers.default = 'browser'
np.set_printoptions(suppress=True)


class S:

    # cutoff defines period that are ignored for any figures/metrics/etc.
    cutoff = 1500
    cutoff_bool = True

    # %% setup parameters - risk averse agent
    kMax = 0.

    # uniform distribution parameters for chartist lookback
    Lmin = 5
    Lmax = 250

    # uniform distribution parameters for initial stock allocation
    Smin = -20 * 100  # upper bound of uniform distribution, amount of stock assigned
    Smax = 20 * 100  # lower bound of uniform distribution

    # uniform distribution parameters for initial cash allocation
    Cmin = 20 * 100  # upper bound of uniform distribution, amount of cash assigned
    Cmax = 100 * 100  # lower bound of uniform distribution

    # agent parameters
    alpha = 0.25
    alpha_mm = 0.25
    risk_aversion_agents = 10.
    risk_aversion_mm = 12.5
    arrival_intensity_mm = 0.6
    order_size_bid_max = 50 * 100
    order_size_ask_max = 50 * 100
    calc_skew_bool = True
    if calc_skew_bool:
        # note / 2 is same as ln((order_size_bid_max / 20) / order_size_bid_max) / order_size_bid_max
        inventory_skew_bid = -np.round(np.log(1 / order_size_bid_max) / order_size_bid_max, 8) / 2
        inventory_skew_ask = -np.round(np.log(1 / order_size_bid_max) / order_size_bid_max, 8) / 2
    else:
        inventory_skew_bid = 0.0001
        inventory_skew_ask = 0.0001
    sigma = 0.0005

    # distribution of individual parameter within simulation
    distribution_param = 'risk_aversion'
    distribution_bool = False
    distribution = [1., 3., 5., 10., 15.]

    # %% setup parameters - market
    price_underlying = 1000.
    price_fundamental_new = price_underlying

    p = 0.5
    tau = 50  # cut off period to clear old trades, trades older than 50 timestamps will be removed from orderbook
    cancel_threshold = 0.1
    price_ub = 1800.
    price_lb = 200.
    random_orders = 50
    round_decimals = 1
    information_threshold = 0.0010
    jump_adjustment = 0.0075 #0.0010

    number_agents = 1000
    # number_agents_market_maker = 1

    p_fundamentalist = 0.45  # 0.35
    p_fundamentalist_3 = 0.0

    p_chartist = 0.45  # 0.35
    p_chartist_3 = 0.0

    period_init = 1000
    period_max = 5000  # 40000
    rebate = 0.0024# * 100
    fee = 0.0029# * 100

    def allocate(self):
        # %% initiate agents
        p_noise = round(1 - self.p_fundamentalist - self.p_chartist, 2)
        number_fundamentalist = self.number_agents * self.p_fundamentalist
        number_chartist = self.number_agents * self.p_chartist
        number_noise = self.number_agents * p_noise

        range_fundamentalist = range(2, int(number_fundamentalist) + 1)
        range_chartist = range(range_fundamentalist[-1] + 1,
                               int(range_fundamentalist[-1] + number_chartist) + 1)
        range_noise = range(range_chartist[-1] + 1,
                            int(range_chartist[-1] + number_noise) + 1)

        setattr(self, 'p_noise', p_noise)
        setattr(self, 'number_fundamentalist', number_fundamentalist)
        setattr(self, 'number_chartist', number_chartist)
        setattr(self, 'number_noise', number_noise)
        setattr(self, 'range_fundamentalist', range_fundamentalist)
        setattr(self, 'range_chartist', range_chartist)
        setattr(self, 'range_noise', range_noise)

        agent_types = ['MarketMaker4']

        if self.number_fundamentalist > 0:
            agent_types.append('AgentCombinationFundamentalist2')
        if self.number_chartist > 0:
            agent_types.append('AgentCombinationChartist2')
        if self.number_noise > 0:
            agent_types.append('AgentCombinationNoise2')

        setattr(self, 'agent_types', agent_types)

        # add price history to order book
        price = self.price_underlying * np.ones(self.period_init)
        ret = np.zeros(self.period_init)
        price[0:self.period_init] = self.price_underlying * (1. + 0.001 * np.random.randn(self.period_init))
        ret[0:self.period_init] = 0.001 * np.random.randn(self.period_init)

        setattr(self, 'ret', ret)
        setattr(self, 'price', price)

        initial_data_fundamentalist = {
            'type': 'AgentCombinationFundamentalist2',
            'sigma_fundamentalist': self.sigma,  # sigma - fundamentalists
            'risk_aversion': self.risk_aversion_agents,
            'price_fundamental': self.price_underlying,
            'Lmax': self.Lmax,
            # upper bound of uniform distribution, time scale factor to calculate average return for chartists
            'Lmin': self.Lmin,  # lower bound of uniform distribution
            'Smax': self.Smax,  # upper bound of uniform distribution, amount of stock assigned
            'Smin': self.Smin,  # lower bound of uniform distribution
            'Cmax': self.Cmax,  # upper bound of uniform distribution, amount of cash assigned
            'Cmin': self.Cmin,  # lower bound of uniform distribution
            'round_decimals': self.round_decimals,  # round decimals to, price discreteness
            'price_ub': self.price_ub,  # enter np.inf for no upper bound
            'price_lb': self.price_lb,  # enter 0 for no lower bound
            'var_scale': 1 * 252,  # scaling variance of CRRA allocation
            'alpha': self.alpha,
            'fee': self.fee,  # per 1 share, liquidity provision
            'rebate': None  # per 1 share, liquidity taking
        }

        initial_data_chartist = {
            'type': 'AgentCombinationChartist2',
            'sigma_chartist': self.sigma * 2,  # sigma - chartists
            'risk_aversion': self.risk_aversion_agents,
            'price_fundamental': self.price_underlying,
            'Lmax': self.Lmax,
            # upper bound of uniform distribution, time scale factor to calculate average return for chartists
            'Lmin': self.Lmin,  # lower bound of uniform distribution
            'Smax': self.Smax,  # upper bound of uniform distribution, amount of stock assigned
            'Smin': self.Smin,  # lower bound of uniform distribution
            'Cmax': self.Cmax,  # upper bound of uniform distribution, amount of cash assigned
            'Cmin': self.Cmin,  # lower bound of uniform distribution
            'round_decimals': self.round_decimals,  # round decimals to, price discreteness
            'price_ub': self.price_ub,  # enter np.inf for no upper bound
            'price_lb': self.price_lb,  # enter 0 for no lower bound
            'var_scale': 1 * 252,  # scaling variance of CRRA allocation
            'ret_scale': 1 * 1,  # scaling variance of CRRA allocation
            'alpha': self.alpha,
            'fee': self.fee,  # per 1 share, liquidity provision
            'rebate': None  # per 1 share, liquidity taking
        }

        initial_data_noise = {
            'type': 'AgentCombinationNoise2',
            'sigma_noise': self.sigma,  # sigma - noise trader
            'risk_aversion': self.risk_aversion_agents,
            'price_fundamental': self.price_underlying,
            'Lmax': self.Lmax,
            # upper bound of uniform distribution, time scale factor to calculate average return for chartists
            'Lmin': self.Lmin,  # lower bound of uniform distribution
            'Smax': self.Smax,  # upper bound of uniform distribution, amount of stock assigned
            'Smin': self.Smin,  # lower bound of uniform distribution
            'Cmax': self.Cmax,  # upper bound of uniform distribution, amount of cash assigned
            'Cmin': self.Cmin,  # lower bound of uniform distribution
            'round_decimals': self.round_decimals,  # round decimals to, price discreteness
            'price_ub': self.price_ub,  # enter np.inf for no upper bound
            'price_lb': self.price_lb,  # enter 0 for no lower bound
            'var_scale': 1 * 252,  # scaling variance of CRRA allocation
            'fee': self.fee,  # per 1 share, liquidity provision
            'rebate': None  # per 1 share, liquidity taking
        }

        initial_data_marketmaker = {
            'id': 1,
            'type': 'MarketMaker4',
            'risk_aversion': self.risk_aversion_mm,
            'arrival_intensity': self.arrival_intensity_mm,
            'inventory_skew_bid': self.inventory_skew_bid,
            'inventory_skew_ask': self.inventory_skew_ask,
            'order_size_bid_max': self.order_size_bid_max,
            'order_size_ask_max': self.order_size_ask_max,
            'Smax': 0,  # upper bound of uniform distribution, amount of stock assigned
            'Smin': 0,  # lower bound of uniform distribution
            'Cmax': 50 * 10,  # upper bound of uniform distribution, amount of cash assigned
            'Cmin': 50 * 10,  # lower bound of uniform distribution
            'round_decimals': 1,  # round decimals to, price discreteness
            'price_ub': self.price_ub,  # enter np.inf for no upper bound
            'price_lb': self.price_lb,  # enter 0 for no lower bound
            # initialize shares and cash
            'shares_init': 0,
            'cash_init': self.price_underlying * 50 * 100 * 1,  # np.random.randint(Cmin, Cmax),
            # 'price_share_init': orderbook.price[period_init - 1, 1],
            'order_size_standard': 50,
            'var_scale': 252 * 1 / 252,
            'depth': 5,
            'lookback': 250,
            'alpha': self.alpha_mm,
            'base_spread': 2,
            'intensity_position': 0.00005,
            'intensity_asymmetry': 0.00003,
            'spread_factor': 1000,
            'fee': self.fee,  # per 1 share, liquidity provision
            'rebate': self.rebate  # per 1 share, liquidity taking
        }

        setattr(self, 'initial_data_fundamentalist', initial_data_fundamentalist)
        setattr(self, 'initial_data_chartist', initial_data_chartist)
        setattr(self, 'initial_data_noise', initial_data_noise)
        setattr(self, 'initial_data_marketmaker', initial_data_marketmaker)