import numpy as np
from scipy.stats import lognorm


class Agent:
    """
    Parent class of all agent specifications that takes the input variables and method.
    """

    def __new__(cls, initial_data, portfolio):
        """
        Creates new class, either by directly using method class or an umbrella class depending on
        portfolio input
        :param initial_data: dict, dictionary with parameters
        :param portfolio: boolean, True = portfolio of options; False = only one option
        :return:
        """
        cls.initial_data = initial_data
        cls.portfolio = portfolio

        def return_subclass(type_name):
            """
            Dictionary of pricing methods. Add method here.
            :return: class, returns class matching to string/name
            """
            switcher = {
                'AgentCombinationFundamentalist2': AgentCombinationFundamentalist2,
                'AgentCombinationChartist2': AgentCombinationChartist2,
                'AgentCombinationNoise2': AgentCombinationNoise2,
                'MarketMaker4': MarketMaker4
            }
            subclass_ = switcher.get(type_name, 'Invalid subclass')
            return subclass_

        # portfolio = False
        if not portfolio:
            new_instance = super(Agent, cls).__new__(return_subclass(initial_data['type']))
            new_instance.__init__(initial_data, portfolio)
            return new_instance
        # portfolio = True
        elif portfolio:
            new_instance = super().__new__(Agent)
            # new_instance.__init__(**initial_data)
            return new_instance
        else:
            raise KeyError('Entered wrong argument for portfolio.')

    def __init__(self, initial_data=None, portfolio=True):
        """
        Initialize attributes only if portfolio=False
        """
        # check whether method and initial_data are not empty + portfolio is True
        if initial_data and not portfolio:
            for key in initial_data:
                setattr(self, key, initial_data[key])

            # will be overwritten by individual agent class if needed
            self.orders_active = None
            self.orders_executed = None
            self.shares = None
            self.cash = None

    def add_agent(self, initial_data):
        """
        Add option as attribute.
        """
        # check if attribute exists
        if initial_data['id'] in self.__dict__:
            raise Warning('You are overwriting an existing class attribute: ' + initial_data['id'])
        new_instance = Agent(initial_data, portfolio=False)
        setattr(self, initial_data['id'] if isinstance(initial_data['id'], str) else str(initial_data['id']),
                new_instance)

    def select_random_agent(self, agent_type=None):
        """
        Selects agent from portfolio randomly. If type is not selected, function will select an agent from all types.
        :param agent_type: string or list
        :return: Agent object
        """
        agents = self.__dict__
        if agent_type:
            # retrieve all available agents given agent_type
            if isinstance(agent_type, list):
                agents_select = [x[1] for x in agents.items() if x[1].type in agent_type]
            else:
                agents_select = [x[1] for x in agents.items() if x[1].type == agent_type]

            # select particular agent
            if len(agents_select) == 1:
                # if only one agent of that type exist
                random_agent = agents_select[0]
            else:
                random_agent = agents_select[np.random.randint(1, len(agents_select))]
        else:
            agents_list = list(agents.values())
            random_agent = agents_list[np.random.randint(1, len(agents_list))]
        return random_agent

    def add_to_executed(self, timestamp_execution, lots, type_, price, timestamp, update_wealth=True,
                        update_costs=True):
        # TODO: this function only applies when an individual agent can place
        """
        Stores executed resting orders and evaluate wealth.
        :param timestamp_execution: int, timestamp of order execution = timestamp of counterparty order
        :param lots: int, traded size (can be different from agent's order)
        :param type_: string, market or limit of the counterparty
        :param price: float, traded price
        :param timestamp: int, timestamp of agent's executed order
        :param update_wealth: boolean, True=evaluate wealth
        :param update_costs: boolean, True=evaluate costs
        :return: None
        """
        # store executed order
        if self.orders_executed.size > 0:
            self.orders_executed = np.insert(self.orders_executed, len(self.orders_executed),
                                             np.array([[timestamp_execution, lots, type_, price, timestamp]],
                                                      dtype=object), axis=0)
        else:
            self.orders_executed = np.array([[timestamp_execution, lots, type_, price, timestamp]], dtype=object)

        # modify self.orders_active and add self.orders
        mask = self.orders_active[:, 0] == timestamp
        order_hit = self.orders_active[mask]
        if order_hit[0, 2] > lots:
            # fill (lots) lower than submitted order
            lots_remaining = order_hit[0, 2] - lots

            # update active order with new lot size
            self.orders_active[mask, 2] = lots_remaining

            # TODO: timestamp and timestamp of executed can be the same!
            # update orders - after active orders adjusted to have remaining lots
            self.add_to_attribute(self, 'orders',
                                  np.concatenate((self.orders_active[mask], np.array([[3]]),
                                                  np.array([[timestamp]])), axis=1))

        elif order_hit[0, 2] <= lots:

            # update orders - before active orders adjusted to have details of completely filled order,
            #   but need to adjusted lot size to zero before
            self.orders_active[mask, 2] = 0

            self.add_to_attribute(self, 'orders',
                                  np.concatenate((self.orders_active[mask], np.array([[3]]),
                                                  np.array([[timestamp]])), axis=1))

            # submitted order filled completely
            self.orders_active = self.orders_active[~mask]

        else:
            # cannot happen
            pass

        # differentiate based on boolean indicator
        if update_costs:
            # find out who was the aggressor, i.e. liquidity provider or taker
            if timestamp == timestamp_execution:
                # order submitted and immediately executed: self.agent is liquidity taker
                self.add_to_attribute(self, 'orders_cost', np.array([[timestamp, -self.fee * lots]], dtype=object))
            elif timestamp != timestamp_execution:
                # resting order executed: self.agent is liquidity provider

                # check whether rebate exists, usually only for market makers
                if self.rebate:
                    self.add_to_attribute(self, 'orders_cost', np.array([[timestamp, self.rebate * lots]],
                                                                        dtype=object))
                else:
                    pass

            else:
                pass

        # differentiate based on boolean indicator
        if update_wealth:
            # calc notional paid/received + include transaction costs
            # Note that transaction costs/rebate has to be included in cash position as cash is evaluated when entering
            # a position. To separate cash from transaction costs it, add transaction costs/deduct rebates to cash.

            # distinguish between transaction costs or no costs (update_costs) & if orders_cost is empty
            if update_costs & self.orders_cost.size > 0:
                # check whether orders_cost is not empty
                mask = self.orders_cost[:, 0] == timestamp
                notional = lots * price + self.orders_cost[mask, 1][0] if mask.sum() > 0 else 0
            else:
                notional = lots * price

            # update shares
            if order_hit[0, 1] == 'buy':
                shares_remaining = self.shares[-1, 1] + lots
                cash_remaining = self.cash[-1, 1] - notional
            elif order_hit[0, 1] == 'sell':
                shares_remaining = self.shares[-1, 1] - lots
                cash_remaining = self.cash[-1, 1] + notional
            else:
                raise Exception('Order side does not exist. Abort.')

            # update shares and cash levels
            self.shares = np.insert(self.shares, len(self.shares),
                                    np.array([[timestamp_execution, shares_remaining]],
                                             dtype=object), axis=0)

            self.cash = np.insert(self.cash, len(self.cash),
                                  np.array([[timestamp_execution, cash_remaining]],
                                           dtype=object), axis=0)

            self.cost = 1
        else:
            pass

    def check_modify(self, timestamp, side, lots, type_, order_price):
        """
        Checks whether modify and/or submit.
        :param timestamp: int
        :param side: str
        :param lots: int
        :param order_price: float
        :return: boolean, int, boolean
        """
        # check whether lots traded are zero
        if lots == 0:
            # agent does not want to trade

            # check whether active order exist
            if self.orders_active.size > 0:
                # active order exist - cancel

                # mask order to be cancelled
                mask = self.orders_active[:, 1] == side
                order_cancelled = self.orders_active[mask]

                # check whether cancellable order is active (or executed already)
                if order_cancelled.size > 0:
                    # store orders cancelled
                    self.add_to_attribute(self, 'orders_cancelled',
                                          np.concatenate((np.array([[timestamp]]), order_cancelled), axis=1))

                    # adjust active orders
                    timestamp_modify = order_cancelled[0, 0]
                    self.orders_active[self.orders_active[:, 0] == timestamp_modify] = \
                        np.array([[timestamp,
                                   side, lots, type_, order_price]], dtype=object)

                    # store orders modified
                    self.add_to_attribute(self, 'orders_modified',
                                          np.array([[timestamp, side, lots, type_, order_price,
                                                     timestamp_modify]], dtype=object))

                    # store order
                    self.add_to_attribute(self, 'orders',
                                          np.array([[timestamp, side, lots, type_, order_price, 4,
                                                     order_cancelled[0, 0]]], dtype=object))

                    modify = False
                    submit = True  # submit because need to cancel the order in the market!
                    cancel = True

                    return modify, timestamp_modify, submit, cancel

                else:
                    # no orders to be cancelled; order was executed already
                    modify = False
                    submit = False  # submit because need to cancel the order in the market!
                    cancel = False

                    return modify, None, submit, cancel

            else:
                modify = False
                timestamp_modify = None
                submit = False
                cancel = False
                return modify, timestamp_modify, submit, cancel
        else:
            pass

        # check whether active orders exist
        if self.orders_active.size > 0:
            # check whether order price is the same as already active orders - do not submit
            if order_price in self.orders_active[:, 4]:
                modify = False
                timestamp_modify = None
                submit = False
                cancel = False
                # no need to alter any attribute
                return modify, timestamp_modify, submit, cancel
            else:
                pass

            # check whether active orders exist
            if side in self.orders_active[:, 1]:
                # active order exist - modify active order
                modify = True
                timestamp_modify = self.orders_active[self.orders_active[:, 1] == side][0, 0]
                submit = True
                cancel = False

                # store orders modified
                self.add_to_attribute(self, 'orders_modified',
                                      np.array([[timestamp, side, lots, type_, order_price,
                                                 timestamp_modify]], dtype=object))

                # store orders
                self.add_to_attribute(self, 'orders',
                                      np.array([[timestamp, side, lots, type_, order_price, 2,
                                                 timestamp_modify]], dtype=object))

                # store orders_active
                self.orders_active[self.orders_active[:, 0] == timestamp_modify] = \
                    np.array([[timestamp,
                               side, lots, type_, order_price]], dtype=object)

            # elif (timestamp in self.orders_active[:, 0]) & (side not in self.orders_active[:, 1]):
            elif side not in self.orders_active[:, 1]:
                # active order exist, but was generated by market maker in same timestamp - add order
                modify = False
                timestamp_modify = None
                submit = True
                cancel = False

                # store orders
                self.add_to_attribute(self, 'orders',
                                      np.array([[timestamp, side, lots, 'limit', order_price, 1, None]],
                                               dtype=object))

                # store orders_active
                self.add_to_attribute(self, 'orders_active',
                                      np.array([[timestamp, side, lots, type_, order_price]], dtype=object))

            else:
                pass

            return modify, timestamp_modify, submit, cancel

        else:
            # no active order - add new order
            modify = False
            timestamp_modify = None
            submit = True
            cancel = False

            # store orders
            self.add_to_attribute(self, 'orders',
                                  np.array([[timestamp, side, lots, 'limit', order_price, 1, None]],
                                           dtype=object))

            # store orders_active
            self.orders_active = np.array([[timestamp, side, lots, type_, order_price]], dtype=object)

            return modify, timestamp_modify, submit, cancel

        return modify, timestamp_modify, submit, cancel

    def add_to_cancelled(self, timestamp):
        # TODO: allow multiple orders being cancelled
        """
        Adds cancelled order and other attributes
        :param timestamp: int
        :return: None
        """
        mask = self.orders_active[:, 0] == timestamp
        if mask.sum() > 0:
            # cancelled order was/is active

            # store orders and remove active order
            # self.add_to_attribute(self, 'orders_cancelled',
            #                       np.concatenate((np.array([[timestamp]]), self.orders_active), axis=1))
            # self.add_to_attribute(self, 'orders_modified',
            #                       np.concatenate((self.orders_active, np.array([[timestamp]])), axis=1))
            # self.add_to_attribute(self, 'orders',
            #                       np.concatenate((self.orders_active, np.array([[4]]), np.array([[timestamp]])),
            #                                      axis=1))
            self.add_to_attribute(self, 'orders_cancelled',
                                  np.concatenate((np.array([[timestamp]]), self.orders_active[mask]), axis=1))
            self.add_to_attribute(self, 'orders_modified',
                                  np.concatenate((self.orders_active[mask], np.array([[timestamp]])), axis=1))
            self.add_to_attribute(self, 'orders',
                                  np.concatenate((self.orders_active[mask], np.array([[4]]), np.array([[timestamp]])),
                                                 axis=1))

            self.orders_active = self.orders_active[~mask]

        else:
            raise Exception('Order is not active, check code.')

    @staticmethod
    def ewm_std(ret, lookback, alpha):

        return_window = np.lib.stride_tricks.sliding_window_view(np.expand_dims(ret, axis=1),
                                                                 window_shape=(lookback, 1),
                                                                 axis=None, subok=False, writeable=False)
        return_window = np.squeeze(return_window, 1)
        std_window = np.std(return_window, axis=0)
        weights = (1 - alpha) ** np.arange(lookback)
        weights /= weights.sum()
        var = np.convolve(weights, std_window[:, 0], 'valid')
        return var

    @staticmethod
    def ewm_mean(ret, lookback, alpha):

        return_window = np.lib.stride_tricks.sliding_window_view(np.expand_dims(ret, axis=1),
                                                                 window_shape=(lookback, 1),
                                                                 axis=None, subok=False, writeable=False)
        return_window = np.squeeze(return_window, 1)
        std_window = np.mean(return_window, axis=0)
        weights = (1 - alpha) ** np.arange(lookback)
        weights /= weights.sum()
        mean = np.convolve(weights, std_window[:, 0], 'valid')
        return mean

    @staticmethod
    def calc_wealth(shares, price, cash):
        wealth = shares * price + cash
        return wealth

    @staticmethod
    def add_to_orders(object_, timestamp, side, lots, type, price):
        """
        Adds order to order store array.
        """
        side_int = int(1 if side == 'buy' else -1)
        type_int = int(1 if 'limit' else 0)
        if object_.orders.size > 0:
            object_.orders = np.insert(object_.orders, len(object_.orders),
                                       np.array([[timestamp, side_int, lots, type_int, price]]), axis=0)
        else:
            object_.orders = np.array([[timestamp, side_int, lots, type_int, price]])

    @staticmethod
    def add_to_forecasts_price(object_, timestamp, forecast_price):
        """
        Adds forecast to forecast store array.
        """
        if object_.forecasts_price.size > 0:
            object_.forecasts_price = np.insert(object_.forecasts_price, len(object_.forecasts_price),
                                                np.array([[timestamp, forecast_price]]), axis=0)
        else:
            object_.forecasts_price = np.array([[timestamp, forecast_price]])

    @staticmethod
    def add_to_attribute(object_, attribute_name, array_):
        """
        Adds forecast to attribute store array.
        """
        attribute = getattr(object_, attribute_name)

        if attribute.size > 0:
            attribute = np.insert(attribute, len(attribute), array_, axis=0)
        else:
            attribute = array_

        setattr(object_, attribute_name, attribute)

    @staticmethod
    def add_to_executed2(object_, timestamp, array_):
        """
        Adds order array [timestamp, lots, price] to executed store array.
        """
        if object_.executed.size > 0:
            object_.executed = np.insert(object_.executed, len(object_.executed),
                                         np.array([np.concatenate((np.array([timestamp]), array_), axis=0)]), axis=0)
        else:
            object_.executed = np.array([np.concatenate((np.array([timestamp]), array_), axis=0)])

    @staticmethod
    def get_lognorm_params(mu, sigma):
        """
        Calculates parameters that can be plugged in into np.random.lognormal from real valued inputs.
        """
        normal_std = np.sqrt(np.log(1 + (sigma / mu) ** 2))
        normal_mean = np.log(mu) - normal_std ** 2 / 2
        return normal_std, normal_mean

    @staticmethod
    def draw_order_price(s, loc, scale, side):
        """
        Draws order price as described in Bartolozzi 2010. Difference to Bartolozzi is that best bid/ask is
        replaced by market price.
        :param s: shape, here 0.5
        :param loc: location, here: market price
        :param scale: variance, here: 10
        :param side: string
        :return:
        """
        dist = lognorm(s=s, loc=loc, scale=scale)
        draw = dist.rvs(size=1)
        quantile_ = dist.ppf(0.5)

        if side == 'buy':
            order_price = (loc + -2) - (draw - quantile_)
        elif side == 'sell':
            order_price = (loc - -2) + (draw - quantile_)
        else:
            raise Exception('Side does not exist.')

        return order_price[0]

    def __str__(cls):
        return cls.__repr__()

    def __getinitargs__(self):
        # we need this function to load the saved pickle file
        return self.args

    def __getnewargs__(self):
        return self.initial_data, self.portfolio

    def __reduce__(self):
        return _new, (self.__class__, self.initial_data, self.portfolio)

    def __getstate__(self):
        # this function saves the state of the initialized
        state = self.__dict__.copy()
        return state


def _new(cls, initial_data, portfolio):
    return cls.__new__(cls, initial_data, portfolio)


class AgentCombinationFundamentalist2(Agent):
    """
    Fundamental part of AgentCombination
    """

    def __init__(self, initial_data, portfolio):
        """
        :param initial_data: dict, assert statements below check all inputs
        :param portfolio: boolean
        """
        super().__init__()
        self.sigma_fundamentalist = None
        self.risk_aversion = None
        self.sigma_wn = None
        self.price_lb = None
        self.price_ub = None
        self.Lmin = None
        self.Lmax = None
        self.Smin = None
        self.Smax = None
        self.Cmin = None
        self.Cmax = None
        self.lookback = None
        self.price_factor = 0.003
        self.round_decimals = None
        self.price_fundamental = None
        self.shares_init = None
        self.cash_init = None
        self.price_share_init = None
        self.var_scale = None
        self.fee = None
        self.rebate = None

        Agent.__init__(self, initial_data, portfolio)

        self.forecasts_return = np.array([])
        self.forecasts_price = np.array([])
        # [timestamp, side, lots, type, price,
        #   flag (new=1, modified=2, modified by execution/partial fill=3, ...),
        #   timestamp linked order]
        self.orders = np.array([], dtype=object)
        # [timestamp, side, lots, type, price]
        self.orders_active = np.array([[]], dtype=object)
        # [timestamp, side, lots, type, price, timestamp modified]
        self.orders_modified = np.array([], dtype=object)
        # [timestamp of cancel, timestamp cancelled, side, lots, type, price]
        self.orders_cancelled = np.array([], dtype=object)
        # [timestamp, lots, type_, price, timestamp resting order that was hit]
        self.orders_executed = np.array([], dtype=object)
        # [timestamp, total fee/rebate]
        self.orders_cost = np.array([], dtype=object)

        # set initial shares and cash allocated
        self.shares = np.array([[0, initial_data['shares_init']]])  # [timestamp, cash]
        self.cash = np.array([[0, initial_data['cash_init']]])  # [timestamp, cash]
        self.price = np.array([[0, initial_data['price_share_init']]])  # [timestamp, cash]

    def update_forecast(self, price, ret, timestamp):
        """
        Calculates forecasted return and price.
        :param price: np.array / price
        :param ret: np.array
        :param timestamp: int
        :return: None
        """
        # forecasts
        forecast_fundamentalist = np.log(self.price_fundamental / price)

        # combine forecasts
        forecast_return = forecast_fundamentalist + self.sigma_fundamentalist * np.random.randn()

        # bound the forecast
        forecast_return = min(forecast_return, 0.5)
        forecast_return = max(forecast_return, -0.5)

        # exponentiate the forecast to get future price forecast
        # forecast could have variance adjustment
        forecast_price = price * np.exp(forecast_return)

        # store return forecast
        if self.forecasts_return.size > 0:
            self.forecasts_return = np.insert(self.forecasts_return, len(self.forecasts_return),
                                              np.array([[timestamp, forecast_return]]), axis=0)
        else:
            self.forecasts_return = np.array([[timestamp, forecast_return]])

        # store price forecast
        if self.forecasts_price.size > 0:
            self.forecasts_price = np.insert(self.forecasts_price, len(self.forecasts_price),
                                             np.array([[timestamp, forecast_price]]), axis=0)
        else:
            self.forecasts_price = np.array([[timestamp, forecast_price]])

    def place_order(self, price, ret, timestamp, **kwargs):
        """
        Evaluates whether to place buy or sell order and returns required order parameters.
        :param price: np.array / float
        :param timestamp: int
        :return: string, int, string, float
        """

        # determine portfolio allocation to risky asset
        ret_last = (self.forecasts_price[-1, 1] - price) / price

        # determine direction
        if ret_last > 0:
            side = 'buy'
        elif ret_last < 0:
            side = 'sell'
        else:
            # TODO
            side = 'buy'

        # draw order price from lognormal
        if self.forecasts_price[-1, 1] > price:
            # bid
            if kwargs['bids'].size > 0:
                order_price = min(max(round((1. - self.price_factor) * self.draw_order_price(s=0.5,
                                                                                             loc=kwargs['bids'][0][2],
                                                                                             scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_lb), self.price_ub)
            else:
                order_price = min(max(round((1. - self.price_factor) * self.draw_order_price(s=0.5, loc=price, scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_lb), self.price_ub)
        else:
            # ask
            if kwargs['asks'].size > 0:
                order_price = max(min(round((1. + self.price_factor) * self.draw_order_price(s=0.5,
                                                                                             loc=kwargs['asks'][0][2],
                                                                                             scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_ub), self.price_lb)
            else:
                order_price = max(min(round((1. + self.price_factor) * self.draw_order_price(s=0.5, loc=price, scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_ub), self.price_lb)

        # calculate desired allocation
        # var = np.var(np.append(ret, ret_last))
        var = self.ewm_std(np.append(ret[:self.Lmax], ret_last), self.lookback, self.alpha)[0] ** 2
        allocation_raw = ret_last / (self.risk_aversion * var * self.var_scale)

        # bound allocation to (-1, 1) - short sale allowed, i.e. one can short sale his whole wealth
        allocation = min(1., allocation_raw)
        allocation = max(-1., allocation)
        # print('Allocation {}'.format(allocation))

        # calculate current wealth
        wealth = self.shares[-1][1] * price + self.cash[-1][1]

        # calculate desired number of shares based on order price
        if wealth < 0:
            # liquidate stocks
            lots_t1 = -self.shares[-1][1]
            # print('liquidate')
        else:
            lots_t1 = np.round(wealth * allocation / order_price, 0)

        # determine how many shares agent has to bay
        if side == 'buy':
            if lots_t1 < self.shares[-1][1]:
                # sell part of shares to have target allocation, sell at order_price
                side = 'sell'
                lots = self.shares[-1][1] - lots_t1
            elif lots_t1 > self.shares[-1][1]:
                # buy shares
                lots = min(abs(lots_t1 - self.shares[-1][1]), np.floor(self.cash[-1][1] / order_price))
            else:
                # target lot size equals current shares
                lots = 0
        elif side == 'sell':
            if lots_t1 < self.shares[-1][1]:
                # sell part of shares to have target allocation, sell at order_price
                lots = self.shares[-1][1] - lots_t1
            elif lots_t1 > self.shares[-1][1]:
                # buy shares to get to target allocation
                side = 'buy'
                lots = min(abs(lots_t1 - self.shares[-1][1]), np.floor(self.cash[-1][1] / order_price))
            else:
                # target lot size equals current shares
                lots = 0
        else:
            raise KeyError('Side \"{}\" does not exist. Abort.'.format(side))

        # print('Current shares {}'.format(self.shares[-1][1]))
        # print('Target shares {}'.format(lots_t1))
        # print('Lots {}'.format(lots))
        # print('Cash/price {}'.format(np.floor(self.cash[-1][1] / order_price)))

        # check whether orders exist
        modify, timestamp_modify, submit, cancel = self.check_modify(timestamp=timestamp + 1, side=side, lots=lots,
                                                                     type_='limit', order_price=order_price)

        # post order
        order_dict = {
            'id': self.id,
            # True = modify existing order, False = Submit new order
            'modify': modify,
            # Timestamp of old order that needs to be modified, None if no modification
            'timestamp_modify': timestamp_modify,
            'side': side,
            'lots': lots,
            'type': 'limit',
            'order_price': order_price,
            'submit': submit,
            'cancel': cancel
        }

        return order_dict,

    def __repr__(self):
        return str(self.id)

    def __str__(self):
        return self.__repr__()

    def __getinitargs__(self):
        # we need this function to load the saved pickle file
        return self.args

    def __getnewargs__(self):
        return self.initial_data, self.portfolio

    def __getstate__(self):
        # this function saves the state of the initialized
        state = self.__dict__.copy()
        return state


class AgentCombinationChartist2(Agent):
    """
    Chartist part of AgentCombination.
    """

    def __init__(self, initial_data, portfolio):
        """
        :param initial_data: dict, assert statements below check all inputs
        :param portfolio: boolean
        """
        super().__init__()
        self.sigma_chartist = None
        self.risk_aversion = None
        self.sigma_wn = None
        self.price_lb = None
        self.price_ub = None
        self.Lmin = None
        self.Lmax = None
        self.Smin = None
        self.Smax = None
        self.Cmin = None
        self.Cmax = None
        self.lookback = None
        self.price_factor = None
        self.round_decimals = None
        self.price_fundamental = None
        self.shares_init = None
        self.cash_init = None
        self.price_share_init = None
        self.var_scale = None
        self.fee = None
        self.rebate = None

        Agent.__init__(self, initial_data, portfolio)

        self.forecasts_return = np.array([])
        self.forecasts_price = np.array([])
        # [timestamp, side, lots, type, price,
        #   flag (new=1, modified=2, modified by execution/partial fill=3, ...),
        #   timestamp linked order]
        self.orders = np.array([], dtype=object)
        # [timestamp, side, lots, type, price]
        self.orders_active = np.array([[]], dtype=object)
        # [timestamp, side, lots, type, price, timestamp modified]
        self.orders_modified = np.array([], dtype=object)
        # [timestamp of cancel, timestamp cancelled, side, lots, type, price]
        self.orders_cancelled = np.array([], dtype=object)
        # [timestamp, lots, type_, price, timestamp resting order that was hit]
        self.orders_executed = np.array([], dtype=object)
        # [timestamp, total fee/rebate]
        self.orders_cost = np.array([], dtype=object)

        # set initial shares and cash allocated
        self.shares = np.array([[0, initial_data['shares_init']]])  # [timestamp, cash]
        self.cash = np.array([[0, initial_data['cash_init']]])  # [timestamp, cash]
        self.price = np.array([[0, initial_data['price_share_init']]])  # [timestamp, cash]

    def update_forecast(self, price, ret, timestamp):
        """
        Calculates forecasted return and price.
        :param price: np.array / price
        :param ret: np.array
        :param timestamp: int
        :return: None
        """

        # forecasts
        forecast_chartist_temp = np.cumsum(ret[:self.Lmax][::-1]) / np.arange(1., float(self.Lmax + 1))
        forecast_chartist = forecast_chartist_temp[self.lookback]
        # forecast_chartist_temp = self.ewm_mean(ret, self.lookback, self.alpha)[0] * self.ret_scale
        # forecast_chartist = forecast_chartist_temp

        # combine forecasts
        forecast_return = forecast_chartist + self.sigma_chartist * np.random.randn()

        # bound the forecast
        forecast_return = min(forecast_return, 0.5)
        forecast_return = max(forecast_return, -0.5)

        # exponentiate the forecast to get future price forecast
        # forecast could have variance adjustment
        forecast_price = price * np.exp(forecast_return)

        # store return forecast
        if self.forecasts_return.size > 0:
            self.forecasts_return = np.insert(self.forecasts_return, len(self.forecasts_return),
                                              np.array([[timestamp, forecast_return]]), axis=0)
        else:
            self.forecasts_return = np.array([[timestamp, forecast_return]])

        # store price forecast
        if self.forecasts_price.size > 0:
            self.forecasts_price = np.insert(self.forecasts_price, len(self.forecasts_price),
                                             np.array([[timestamp, forecast_price]]), axis=0)
        else:
            self.forecasts_price = np.array([[timestamp, forecast_price]])

    def place_order(self, price, ret, timestamp, **kwargs):
        """
        Evaluates whether to place buy or sell order and returns required order parameters.
        :param price: np.array / float
        :param timestamp: int
        :return: string, int, string, float
        """

        # determine portfolio allocation to risky asset
        ret_last = (self.forecasts_price[-1, 1] - price) / price

        # determine direction
        if ret_last > 0:
            side = 'buy'
        elif ret_last < 0:
            side = 'sell'
        else:
            # TODO
            side = 'buy'

        # draw order price from lognormal
        if self.forecasts_price[-1, 1] > price:
            # bid
            if kwargs['bids'].size > 0:
                order_price = min(max(round((1. - self.price_factor) * self.draw_order_price(s=0.5,
                                                                                             loc=kwargs['bids'][0][2],
                                                                                             scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_lb), self.price_ub)
            else:
                order_price = min(max(round((1. - self.price_factor) * self.draw_order_price(s=0.5, loc=price, scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_lb), self.price_ub)
        else:
            # ask
            if kwargs['asks'].size > 0:
                order_price = max(min(round((1. + self.price_factor) * self.draw_order_price(s=0.5,
                                                                                             loc=kwargs['asks'][0][2],
                                                                                             scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_ub), self.price_lb)
            else:
                order_price = max(min(round((1. + self.price_factor) * self.draw_order_price(s=0.5, loc=price, scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_ub), self.price_lb)

        # calculate desired allocation
        # var = np.var(np.append(ret, ret_last))
        var = self.ewm_std(np.append(ret[:self.Lmax], ret_last), self.lookback, self.alpha)[0] ** 2
        allocation_raw = ret_last / (self.risk_aversion * var * self.var_scale)

        # bound allocation to (-1, 1) - short sale allowed, i.e. one can short sale his whole wealth
        allocation = min(1., allocation_raw)
        allocation = max(-1., allocation)
        # print('Allocation {}'.format(allocation))

        # calculate current wealth
        wealth = self.shares[-1][1] * price + self.cash[-1][1]

        # calculate desired number of shares based on order price
        if wealth < 0:
            # liquidate stocks
            lots_t1 = -self.shares[-1][1]
            # print('liquidate')
        else:
            lots_t1 = np.round(wealth * allocation / order_price, 0)

        # determine how many shares agent has to bay
        if side == 'buy':
            if lots_t1 < self.shares[-1][1]:
                # sell part of shares to have target allocation, sell at order_price
                side = 'sell'
                lots = self.shares[-1][1] - lots_t1
            elif lots_t1 > self.shares[-1][1]:
                # buy shares
                lots = min(abs(lots_t1 - self.shares[-1][1]), np.floor(self.cash[-1][1] / order_price))
            else:
                # target lot size equals current shares
                lots = 0
        elif side == 'sell':
            if lots_t1 < self.shares[-1][1]:
                # sell part of shares to have target allocation, sell at order_price
                lots = self.shares[-1][1] - lots_t1
            elif lots_t1 > self.shares[-1][1]:
                # buy shares to get to target allocation
                side = 'buy'
                lots = min(abs(lots_t1 - self.shares[-1][1]), np.floor(self.cash[-1][1] / order_price))
            else:
                # target lot size equals current shares
                lots = 0
        else:
            raise KeyError('Side \"{}\" does not exist. Abort.'.format(side))

        # print('Current shares {}'.format(self.shares[-1][1]))
        # print('Target shares {}'.format(lots_t1))
        # print('Lots {}'.format(lots))
        # print('Cash/price {}'.format(np.floor(self.cash[-1][1] / order_price)))

        # check whether orders exist
        modify, timestamp_modify, submit, cancel = self.check_modify(timestamp=timestamp + 1, side=side, lots=lots,
                                                                     type_='limit', order_price=order_price)

        # post order
        order_dict = {
            'id': self.id,
            # True = modify existing order, False = Submit new order
            'modify': modify,
            # Timestamp of old order that needs to be modified, None if no modification
            'timestamp_modify': timestamp_modify,
            'side': side,
            'lots': lots,
            'type': 'limit',
            'order_price': order_price,
            'submit': submit,
            'cancel': cancel
        }

        return order_dict,

    def __repr__(self):
        return str(self.id)

    def __str__(self):
        return self.__repr__()

    def __getinitargs__(self):
        # we need this function to load the saved pickle file
        return self.args

    def __getnewargs__(self):
        return self.initial_data, self.portfolio

    def __getstate__(self):
        # this function saves the state of the initialized
        state = self.__dict__.copy()
        return state


class AgentCombinationNoise2(Agent):
    """
    Noise part of AgentCombination. Can take take allocation betwenn (-1, 1) and cash based allocation (-1, 1), i.e.
    -/+ price * shares.
    """

    def __init__(self, initial_data, portfolio):
        """
        :param initial_data: dict, assert statements below check all inputs
        :param portfolio: boolean
        """
        super().__init__()
        self.sigma_noise = None
        self.risk_aversion = None
        self.sigma_wn = None
        self.price_lb = None
        self.price_ub = None
        self.Lmin = None
        self.Lmax = None
        self.Smin = None
        self.Smax = None
        self.Cmin = None
        self.Cmax = None
        self.lookback = None
        self.price_factor = None
        self.round_decimals = None
        self.price_fundamental = None
        self.shares_init = None
        self.cash_init = None
        self.price_share_init = None
        self.var_scale = None
        self.fee = None
        self.rebate = None

        Agent.__init__(self, initial_data, portfolio)

        self.forecasts_return = np.array([])
        self.forecasts_price = np.array([])
        # [timestamp, side, lots, type, price,
        #   flag (new=1, modified=2, modified by execution/partial fill=3, ...),
        #   timestamp linked order]
        self.orders = np.array([], dtype=object)
        # [timestamp, side, lots, type, price]
        self.orders_active = np.array([[]], dtype=object)
        # [timestamp, side, lots, type, price, timestamp modified]
        self.orders_modified = np.array([], dtype=object)
        # [timestamp of cancel, timestamp cancelled, side, lots, type, price]
        self.orders_cancelled = np.array([], dtype=object)
        # [timestamp, lots, type_, price, timestamp resting order that was hit]
        self.orders_executed = np.array([], dtype=object)
        # [timestamp, total fee/rebate]
        self.orders_cost = np.array([], dtype=object)

        # set initial shares and cash allocated
        self.shares = np.array([[0, initial_data['shares_init']]])  # [timestamp, cash]
        self.cash = np.array([[0, initial_data['cash_init']]])  # [timestamp, cash]
        self.price = np.array([[0, initial_data['price_share_init']]])  # [timestamp, cash]

    def update_forecast(self, price, ret, timestamp):
        """
        Calculates forecasted return and price.
        :param price: np.array / price
        :param ret: np.array
        :param timestamp: int
        :return: None
        """

        # forecasts
        forecast_noise = self.sigma_noise * np.random.randn()

        # combine forecasts
        forecast_return = forecast_noise

        # bound the forecast
        forecast_return = min(forecast_return, 0.5)
        forecast_return = max(forecast_return, -0.5)

        # exponentiate the forecast to get future price forecast
        # forecast could have variance adjustment
        forecast_price = price * np.exp(forecast_return)

        # store return forecast
        if self.forecasts_return.size > 0:
            self.forecasts_return = np.insert(self.forecasts_return, len(self.forecasts_return),
                                              np.array([[timestamp, forecast_return]]), axis=0)
        else:
            self.forecasts_return = np.array([[timestamp, forecast_return]])

        # store price forecast
        if self.forecasts_price.size > 0:
            self.forecasts_price = np.insert(self.forecasts_price, len(self.forecasts_price),
                                             np.array([[timestamp, forecast_price]]), axis=0)
        else:
            self.forecasts_price = np.array([[timestamp, forecast_price]])

    def place_order(self, price, ret, timestamp, **kwargs):
        """
        Evaluates whether to place buy or sell order and returns required order parameters.
        :param price: np.array / float
        :param timestamp: int
        :return: string, int, string, float
        """

        # determine portfolio allocation to risky asset
        ret_last = (self.forecasts_price[-1, 1] - price) / price

        # determine direction
        if ret_last > 0:
            side = 'buy'
        elif ret_last < 0:
            side = 'sell'
        else:
            # TODO
            side = 'buy'

        # draw order price from lognormal
        if self.forecasts_price[-1, 1] > price:
            # bid
            if kwargs['bids'].size > 0:
                order_price = min(max(round((1. - self.price_factor) * self.draw_order_price(s=0.5,
                                                                                             loc=kwargs['bids'][0][2],
                                                                                             scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_lb), self.price_ub)
            else:
                order_price = min(max(round((1. - self.price_factor) * self.draw_order_price(s=0.5, loc=price, scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_lb), self.price_ub)
        else:
            # ask
            if kwargs['asks'].size > 0:
                order_price = max(min(round((1. + self.price_factor) * self.draw_order_price(s=0.5,
                                                                                             loc=kwargs['asks'][0][2],
                                                                                             scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_ub), self.price_lb)
            else:
                order_price = max(min(round((1. + self.price_factor) * self.draw_order_price(s=0.5, loc=price, scale=10,
                                                                                             side=side),
                                            self.round_decimals), self.price_ub), self.price_lb)

        # calculate desired allocation
        allocation_raw = ret_last / (self.risk_aversion * np.var(np.append(ret[:self.Lmax], ret_last)) * self.var_scale)

        # bound allocation to (-1, 1) - short sale allowed, i.e. one can short sale his whole wealth
        allocation = min(1, allocation_raw)
        allocation = max(-1, allocation)
        # print('Allocation {}'.format(allocation))

        # calculate current wealth
        wealth = self.shares[-1][1] * price + self.cash[-1][1]

        # calculate desired number of shares based on order price
        if wealth < 0:
            # liquidate stocks
            lots_t1 = -self.shares[-1][1]
            # print('liquidate')
        else:
            lots_t1 = np.round(wealth * allocation / order_price, 0)

        # determine how many shares agent has to bay
        if side == 'buy':
            if lots_t1 < self.shares[-1][1]:
                # sell part of shares to have target allocation, sell at order_price
                side = 'sell'
                lots = self.shares[-1][1] - lots_t1
            elif lots_t1 > self.shares[-1][1]:
                # buy shares
                lots = min(abs(lots_t1 - self.shares[-1][1]), np.floor(self.cash[-1][1] / order_price))
            else:
                # target lot size equals current shares
                lots = 0
        elif side == 'sell':
            if lots_t1 < self.shares[-1][1]:
                # sell part of shares to have target allocation, sell at order_price
                lots = self.shares[-1][1] - lots_t1
            elif lots_t1 > self.shares[-1][1]:
                # buy shares to get to target allocation
                side = 'buy'
                lots = min(abs(lots_t1 - self.shares[-1][1]), np.floor(self.cash[-1][1] / order_price))
            else:
                # target lot size equals current shares
                lots = 0
        else:
            raise KeyError('Side \"{}\" does not exist. Abort.'.format(side))

        # print('Current shares {}'.format(self.shares[-1][1]))
        # print('Target shares {}'.format(lots_t1))
        # print('Lots {}'.format(lots))
        # print('Cash/price {}'.format(np.floor(self.cash[-1][1] / order_price)))

        # check whether orders exist
        modify, timestamp_modify, submit, cancel = self.check_modify(timestamp=timestamp + 1, side=side, lots=lots,
                                                                     type_='limit', order_price=order_price)

        # post order
        order_dict = {
            'id': self.id,
            # True = modify existing order, False = Submit new order
            'modify': modify,
            # Timestamp of old order that needs to be modified, None if no modification
            'timestamp_modify': timestamp_modify,
            'side': side,
            'lots': lots,
            'type': 'limit',
            'order_price': order_price,
            'submit': submit,
            'cancel': cancel
        }

        return order_dict,

    def __repr__(self):
        return str(self.id)

    def __str__(self):
        return self.__repr__()

    def __getinitargs__(self):
        # we need this function to load the saved pickle file
        return self.args

    def __getnewargs__(self):
        return self.initial_data, self.portfolio

    def __getstate__(self):
        # this function saves the state of the initialized
        state = self.__dict__.copy()
        return state


class MarketMaker4(Agent):
    """
    Market maker a la Avellaneda and Stoikov (2008).
    """

    def __init__(self, initial_data, portfolio):
        """
        :param initial_data: dict, assert statements below check all inputs
        :param portfolio: boolean
        """
        super().__init__()
        self.price_lb = None
        self.price_ub = None
        self.Smin = None
        self.Smax = None
        self.Cmin = None
        self.Cmax = None
        self.risk_aversion = None
        self.arrival_intensity = None
        self.order_size_bid_max = None
        self.order_size_ask_max = None
        self.round_decimals = None
        self.shares_init = None
        self.cash_init = None
        self.price_share_init = None
        self.order_size_standard = None
        self.fee = None
        self.rebate = None

        Agent.__init__(self, initial_data, portfolio)

        self.forecasts_return = np.array([])
        self.forecasts_price_bid = np.array([])
        self.forecasts_price_ask = np.array([])
        # [timestamp, side, lots, type, price,
        #   flag (new=1, modified=2, modified by execution/partial fill=3, ...),
        #   timestamp linked order]
        self.orders = np.array([], dtype=object)
        # [timestamp, side, lots, type, price]
        self.orders_active = np.array([[]], dtype=object)
        # [timestamp, side, lots, type, price, timestamp modified]
        self.orders_modified = np.array([], dtype=object)
        # [timestamp of cancel, timestamp cancelled, side, lots, type, price]
        self.orders_cancelled = np.array([], dtype=object)
        # [timestamp, lots, type_, price, timestamp resting order that was hit]
        self.orders_executed = np.array([], dtype=object)
        # [timestamp, total fee/rebate]
        self.orders_cost = np.array([[]])

        # set initial shares and cash allocated
        self.shares = np.array([[0, initial_data['shares_init']]])  # [timestamp, cash]
        self.cash = np.array([[0, initial_data['cash_init']]])  # [timestamp, cash]
        self.price = np.array([[0, initial_data['price_share_init']]])  # [timestamp, cash]

    def update_forecast(self, price, ret, bids, asks, timestamp):
        """
        Calculates forecasted return and price.
        :param price: np.array / price
        :param ret: np.array
        :param timestamp: int
        :return: None
        """
        # 1. calculation reservation price
        T = 1.
        t = 0.
        # t = timestamp / 12000
        var = self.ewm_std(ret, self.lookback, self.alpha)[0] ** 2
        # var = np.var(ret)
        reservation_price = price - self.shares[-1][1] * self.risk_aversion * var * (T - t) * self.var_scale

        # 2. calculate bid and ask prices
        spread = self.risk_aversion * var * (T - t) * self.var_scale \
                 + np.log(1 + self.risk_aversion / self.arrival_intensity)
        half_spread = spread / 2

        price_forecast_bid = reservation_price - half_spread
        price_forecast_ask = reservation_price + half_spread

        # 2. calculate spread based on Nakajimi (2006)
        # if self.shares[-1][1] == 0:
        #     upper_spread = self.base_spread
        #     lower_spread = self.base_spread
        # else:
        #     upper_spread = -(self.intensity_position - self.intensity_asymmetry * (
        #             self.shares[-1][1] / abs(self.shares[-1][1]))) * (self.shares[-1][1] ** 3) + self.base_spread
        #     lower_spread = (self.intensity_position + self.intensity_asymmetry * (
        #             self.shares[-1][1] / abs(self.shares[-1][1]))) * (self.shares[-1][1] ** 3) + self.base_spread

        # price_forecast_bid = reservation_price * (1 - lower_spread / self.spread_factor)
        # price_forecast_ask = reservation_price * (1 + upper_spread / self.spread_factor)

        if asks.size > 0:
            if price_forecast_bid > asks[0][2]:
                # print('Adjust bid to avoid hitting own')
                # adjust price_forecast_bid to avoid aggressive order
                price_forecast_bid_adj = asks[0][2] - self.round_decimals / 10
            else:
                price_forecast_bid_adj = price_forecast_bid
        else:
            price_forecast_bid_adj = price_forecast_bid

        if bids.size > 0:
            if price_forecast_ask < bids[0][2]:
                # print('Adjust ask to avoid hitting own')
                # adjust price_forecast_ask to avoid aggressive order
                price_forecast_ask_adj = bids[0][2] + self.round_decimals / 10
            else:
                price_forecast_ask_adj = price_forecast_ask
        else:
            price_forecast_ask_adj = price_forecast_ask

        # make sure own bid does not cross own ask or vice versa
        if price_forecast_bid_adj > price_forecast_ask_adj:
            # print('Do not adjust original prices otherwise market maker hits himself')
            # leave price forecasts as they were without adjustment to avoid aggressive orders
            pass
        else:
            # adjust original price forecasts
            price_forecast_bid = price_forecast_bid_adj
            price_forecast_ask = price_forecast_ask_adj

        # print('Reservation price {}'.format(reservation_price))
        # print('Spread {}'.format(spread))

        # adjust forecasts for lower and upper bound
        if (price_forecast_bid >= self.price_ub) | (price_forecast_ask >= self.price_ub):
            # calculate distance to upper bound
            abs_deviation = max(price_forecast_bid - self.price_ub, price_forecast_ask - self.price_ub)
            price_forecast_bid = price_forecast_bid - abs_deviation
            price_forecast_ask = price_forecast_ask - abs_deviation
        elif (price_forecast_bid <= self.price_lb) | (price_forecast_ask <= self.price_lb):
            # calculate distance to lower bound
            abs_deviation = min(price_forecast_bid - self.price_lb, price_forecast_ask - self.price_lb)
            price_forecast_bid = price_forecast_bid + abs_deviation
            price_forecast_ask = price_forecast_ask + abs_deviation
        else:
            pass

        self.add_to_attribute(self, 'forecasts_price_bid', np.array([[timestamp, price_forecast_bid]]))
        self.add_to_attribute(self, 'forecasts_price_ask', np.array([[timestamp, price_forecast_ask]]))

    def place_order(self, price, bids, asks, timestamp):
        """
        Evaluates whether to place buy or sell order and returns required order parameters.
        :param price: np.array / float
        :param timestamp: int
        :return: string, int, string, float
        """

        # adjust to keep prices in line with boundaries
        order_price_bid = min(max(round(self.forecasts_price_bid[-1, 1], self.round_decimals),
                                  self.price_lb), self.price_ub)
        order_price_ask = max(min(round(self.forecasts_price_ask[-1, 1], self.round_decimals),
                                  self.price_ub), self.price_lb)

        side_bid = 'buy'
        # lots_bid = self.order_size_standard
        if self.shares[-1][1] <= 0:
            # lots_bid = self.order_size_bid_max
            lots_bid = self.order_size_bid_max
        else:
            # lots_bid = int(round(self.order_size_bid_max * np.exp(self.inventory_skew_bid * self.shares[-1][1]), 0))
            lots_bid = int(round(self.order_size_bid_max * np.exp(-self.inventory_skew_bid * self.shares[-1][1]), 0))
            int(round(self.order_size_bid_max * np.exp(0 * -50), 0))
        type_bid = 'limit'

        modify_bid, timestamp_modify_bid, submit_bid, cancel_bid = self.check_modify(
            timestamp=timestamp + 1, side=side_bid,
            lots=lots_bid, type_=type_bid,
            order_price=order_price_bid)

        side_ask = 'sell'
        # lots_ask = self.order_size_standard
        if self.shares[-1][1] >= 0:
            # lots_ask = self.order_size_ask_max
            lots_ask = self.order_size_ask_max
        else:
            # lots_ask = int(round(self.order_size_ask_max * np.exp(self.inventory_skew_ask * self.shares[-1][1]), 0))
            lots_ask = int(round(self.order_size_ask_max * np.exp(self.inventory_skew_ask * self.shares[-1][1]), 0))
        type_ask = 'limit'

        # note: add unit to timestamp if bid is submitted
        modify_ask, timestamp_modify_ask, submit_ask, cancel_ask = self.check_modify(
            timestamp=timestamp + 2 if submit_bid else timestamp + 1, side=side_ask,
            lots=lots_ask, type_=type_ask,
            order_price=order_price_ask)

        # aggregate orders to dict
        order_dict_bid = {
            'id': self.id,
            # True = modify existing order, False = Submit new order
            'modify': modify_bid,
            # Timestamp of old order that needs to be modified, None if no modification
            'timestamp_modify': timestamp_modify_bid,
            'side': side_bid,
            'lots': lots_bid,
            'type': type_bid,
            'order_price': order_price_bid,
            'submit': submit_bid,
            'cancel': cancel_bid
        }

        order_dict_ask = {
            'id': self.id,
            # True = modify existing order, False = Submit new order
            'modify': modify_ask,
            # Timestamp of old order that needs to be modified, None if no modification
            'timestamp_modify': timestamp_modify_ask,
            'side': side_ask,
            'lots': lots_ask,
            'type': type_ask,
            'order_price': order_price_ask,
            'submit': submit_ask,
            'cancel': cancel_ask
        }

        # sort order submission: 1. cancel, 2. modify, 3. new

        # A. modification + order crossing
        if submit_bid and submit_ask and not modify_bid and not modify_ask:
            # 1. bid and ask are new orders - ordering does not matter

            # check whether ask is cancelled
            if order_dict_ask['cancel']:
                # place ask first
                return_ = (order_dict_ask, order_dict_bid)

            else:
                # place bid first
                return_ = (order_dict_bid, order_dict_ask)

            # return_ = (order_dict_bid, order_dict_ask)

        elif submit_bid and submit_ask and modify_bid and modify_ask:
            # 2. bid and ask are modified - evaluate if crossing of both orders to
            #   left or right from own modified orders

            order_modified_ask = self.orders[self.orders[:, 0] == timestamp_modify_ask]
            order_modified_bid = self.orders[self.orders[:, 0] == timestamp_modify_bid]
            if order_price_bid > order_modified_ask[0, 4]:
                # bid and ask cross to the right

                # adjust orders and orders_active by changing timestamp
                timestamp_new_bid = self.orders_active[self.orders_active[:, 1] == 'sell'][0, 0]
                timestamp_new_ask = self.orders_active[self.orders_active[:, 1] == 'buy'][0, 0]

                # change orders
                self.orders[(np.isin(self.orders[:, 0], self.orders_active[:, 0])) &
                            (self.orders[:, 1] == 'buy'), 0] = timestamp_new_bid
                self.orders[(np.isin(self.orders[:, 0], self.orders_active[:, 0])) &
                            (self.orders[:, 1] == 'sell'), 0] = timestamp_new_ask

                # change orders_active
                self.orders_active[self.orders_active[:, 1] == 'buy', 0] = timestamp_new_bid
                self.orders_active[self.orders_active[:, 1] == 'sell', 0] = timestamp_new_ask

                # place ask first
                return_ = (order_dict_ask, order_dict_bid)

            elif order_price_ask < order_modified_bid[0, 4]:
                # bid and ask cross to the left
                return_ = (order_dict_bid, order_dict_ask)

            else:
                # no crossing - ordering does not matter
                return_ = (order_dict_bid, order_dict_ask)

        elif submit_bid and submit_ask and modify_bid and not modify_ask:
            # 3. bid is modified - submit modified always first

            # check whether ask is cancelled
            if order_dict_ask['cancel']:
                # place ask first
                return_ = (order_dict_ask, order_dict_bid)

            else:
                # place bid first
                return_ = (order_dict_bid, order_dict_ask)

        elif submit_bid and submit_ask and not modify_bid and modify_ask:
            # 4. ask is modified - submit modified always first

            # check whether bid is cancelled
            if order_dict_bid['cancel']:
                # place bid first; submit cancel first even though ask is modified
                return_ = (order_dict_bid, order_dict_ask)

            else:
                # bid is new submit and ask is modified: change order
                timestamp_new_bid = self.orders_active[self.orders_active[:, 1] == 'sell'][0, 0]
                timestamp_new_ask = self.orders_active[self.orders_active[:, 1] == 'buy'][0, 0]

                # change orders
                self.orders[(np.isin(self.orders[:, 0], self.orders_active[:, 0])) &
                            (self.orders[:, 1] == 'buy'), 0] = timestamp_new_bid
                self.orders[(np.isin(self.orders[:, 0], self.orders_active[:, 0])) &
                            (self.orders[:, 1] == 'sell'), 0] = timestamp_new_ask

                # change orders_active
                self.orders_active[self.orders_active[:, 1] == 'buy', 0] = timestamp_new_bid
                self.orders_active[self.orders_active[:, 1] == 'sell', 0] = timestamp_new_ask

                # place ask first
                return_ = (order_dict_ask, order_dict_bid)

        else:
            # 5. no submission of bid or ask or both orders
            return_ = (order_dict_bid, order_dict_ask)

        # B. cancellation + modification/no modification
        # 1. bid
        if order_dict_bid['cancel']:
            # bid cancelled

            if order_dict_ask['modify'] & order_dict_ask['submit']:
                # 1a. ask modified + submitted: no change of order
                pass

            elif (not order_dict_ask['modify']) & order_dict_ask['submit']:
                # 1b. ask submitted: no change of order
                pass

            else:
                # 1c. ask not modified/submitted:
                pass

        # 2. ask
        if order_dict_ask['cancel']:
            # ask cancelled
            if order_dict_bid['modify'] & order_dict_bid['submit']:
                # 2a. bid modified + submitted: change of order
                timestamp_new_ask = self.orders_active[self.orders_active[:, 1] == 'buy'][0, 0]
                timestamp_new_bid = self.orders_active[self.orders_active[:, 1] == 'sell'][0, 0]

                # change orders
                self.orders[(np.isin(self.orders[:, 0], self.orders_active[:, 0])) &
                            (self.orders[:, 1] == 'sell'), 0] = timestamp_new_ask
                self.orders[(np.isin(self.orders[:, 0], self.orders_active[:, 0])) &
                            (self.orders[:, 1] == 'buy'), 0] = timestamp_new_bid

                # change orders_active
                self.orders_active[self.orders_active[:, 1] == 'buy', 0] = timestamp_new_bid
                self.orders_active[self.orders_active[:, 1] == 'sell', 0] = timestamp_new_ask

            elif (not order_dict_bid['modify']) & order_dict_bid['submit']:
                # 2b. bid submitted: change of order
                timestamp_new_ask = self.orders_active[self.orders_active[:, 1] == 'buy'][0, 0]
                timestamp_new_bid = self.orders_active[self.orders_active[:, 1] == 'sell'][0, 0]

                # change orders
                self.orders[(np.isin(self.orders[:, 0], self.orders_active[:, 0])) &
                            (self.orders[:, 1] == 'sell'), 0] = timestamp_new_ask
                self.orders[(np.isin(self.orders[:, 0], self.orders_active[:, 0])) &
                            (self.orders[:, 1] == 'buy'), 0] = timestamp_new_bid

                # change orders_active
                self.orders_active[self.orders_active[:, 1] == 'buy', 0] = timestamp_new_bid
                self.orders_active[self.orders_active[:, 1] == 'sell', 0] = timestamp_new_ask

            else:
                # 1c. bid not modified/submitted: no change of order
                pass

        # D. remove cancellations from orders_active
        if order_dict_bid['cancel']:
            # cancel bid
            mask = self.orders_active[:, 1] == 'buy'
            self.orders_active = self.orders_active[~mask]

        elif order_dict_ask['cancel']:
            # cancel ask
            mask = self.orders_active[:, 1] == 'sell'
            self.orders_active = self.orders_active[~mask]

        else:
            # nothing to cancel
            pass

        # print(self.orders_active)

        return return_

    def __repr__(self):
        return str(self.id)

    def __str__(self):
        return self.__repr__()

    def __getinitargs__(self):
        # we need this function to load the saved pickle file
        return self.args

    def __getnewargs__(self):
        return self.initial_data, self.portfolio

    def __getstate__(self):
        # this function saves the state of the initialized
        state = self.__dict__.copy()
        return