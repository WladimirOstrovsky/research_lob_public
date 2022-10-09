"""
Initiates and runs orderbook
"""

from AgentClass import Agent
from OrderbookClass import Orderbook
import time
import numpy as np
import numpy.random as rnd
import plotly.io as pio
pio.renderers.default = 'browser'
np.set_printoptions(suppress=True)


def main(config):

    # initiate agents
    orderbook = Orderbook(periods=config.period_max * 2, period_init=config.period_init, period_max=config.period_max)
    orderbook.add_price_history(config.price)
    orderbook.add_random_orders(tau=config.random_orders, lot_size=(config.Smin, config.Smax)
                                if config.Smin >= 0 else (abs(config.Smin), config.Smax * 2),
                                mult=4, interval=(900, 1100))

    config.initial_data_marketmaker.update({'price_share_init': orderbook.price[config.period_init - 1, 1]})

    agent_portfolio = Agent(initial_data=None, portfolio=True)
    agent_portfolio.add_agent(initial_data=config.initial_data_marketmaker)

    # initiate agents - fundamentalists
    for i in config.range_fundamentalist:
        config.initial_data_fundamentalist.update({'id': i})  # identifier
        # uniform draw for every agent: forecast price to order factor
        config.initial_data_fundamentalist.update({'price_factor': config.kMax * np.random.rand()})
        # uniform draw for every agent: chartist lookback period
        config.initial_data_fundamentalist.update({'lookback': np.random.randint(config.Lmin, config.Lmax)})
        # uniform draw for every agent: amount of stocks
        config.initial_data_fundamentalist.update({'shares_init': np.random.randint(config.Smin, config.Smax)})
        # uniform draw for every agent: amount of cash
        config.initial_data_fundamentalist.update({'cash_init':
                                                       config.price_underlying * np.random.randint(config.Cmin,
                                                                                                   config.Cmax)})
        # initial price of the stock to calculate the agent's wealth
        config.initial_data_fundamentalist.update({'price_share_init': orderbook.price[config.period_init - 1, 1]})

        # TODO: does not work for uneven values -
        if config.distribution_bool:
            config.initial_data_fundamentalist.update({
                config.distribution_param:
                    (config.distribution *
                     int(len(config.range_fundamentalist) /
                         len(config.distribution)))[i - min(config.range_fundamentalist)]
            })

        agent_portfolio.add_agent(initial_data=config.initial_data_fundamentalist)  # initialize agent

    # initiate agents - chartist
    for i in config.range_chartist:
        config.initial_data_chartist.update({'id': i})  # identifier
        # uniform draw for every agent: forecast price to order factor
        config.initial_data_chartist.update({'price_factor': config.kMax * np.random.rand()})
        # uniform draw for every agent: chartist lookback period
        config.initial_data_chartist.update({'lookback': np.random.randint(config.Lmin, config.Lmax)})
        # uniform draw for every agent: amount of stocks
        config.initial_data_chartist.update({'shares_init': np.random.randint(config.Smin, config.Smax)})
        # uniform draw for every agent: amount of cash
        config.initial_data_chartist.update({'cash_init': config.price_underlying * np.random.randint(config.Cmin,
                                                                                                      config.Cmax)})
        # initial price of the stock to calculate the agent's wealth
        config.initial_data_chartist.update({'price_share_init': orderbook.price[config.period_init - 1, 1]})

        if config.distribution_bool:
            config.initial_data_chartist.update({
                config.distribution_param:
                    (config.distribution *
                     int(len(config.range_chartist) / len(config.distribution)))[i - min(config.range_chartist)]
            })

        agent_portfolio.add_agent(initial_data=config.initial_data_chartist)  # initialize agent

    # initiate agents - noise
    for i in config.range_noise:
        config.initial_data_noise.update({'id': i})  # identifier
        # uniform draw for every agent: forecast price to order factor
        config.initial_data_noise.update({'price_factor': config.kMax * np.random.rand()})
        # uniform draw for every agent: chartist lookback period
        config.initial_data_noise.update({'lookback': np.random.randint(config.Lmin, config.Lmax)})
        # uniform draw for every agent: amount of stocks
        config.initial_data_noise.update({'shares_init': np.random.randint(config.Smin, config.Smax)})
        # uniform draw for every agent: amount of cash
        config.initial_data_noise.update({'cash_init': config.price_underlying * np.random.randint(config.Cmin,
                                                                                                   config.Cmax)})
        # initial price of the stock to calculate the agent's wealth
        config.initial_data_noise.update({'price_share_init': orderbook.price[config.period_init - 1, 1]})

        if config.distribution_bool:
            config.initial_data_noise.update({
                config.distribution_param:
                    (config.distribution *
                     int(len(config.range_noise) /
                         len(config.distribution)))[i - min(config.range_noise)]
            })

        agent_portfolio.add_agent(initial_data=config.initial_data_noise)  # initialize agent

    # %% simulate LOB
    start = time.time()
    agent_type_previous = None

    # loop through periods
    for timestamp in range(config.period_init, config.period_max):
        timestamp_last = orderbook.timestamp
        trade = False
        # calculate new fundamental price
        if (rnd.uniform(0, 1) * rnd.uniform(0, 1) < config.information_threshold) & (timestamp > 2000):
            config.price_fundamental_new = config.price_fundamental_new * (1 +
                                                                           config.jump_adjustment * np.random.randn())
            # print('Price fundamental adjusted to {}'.format(config.price_fundamental_new))
        #
        # if timestamp in (2500, 5000, 7500):
        #     price_fundamental_new = price_fundamental_new * (1 - (0.01 if np.random.random() < 0.5 else - 0.01))

        # select agent until found one who is willing to trade
        while not trade:

            # select until MM follows another agent and another agent follows MM
            # while not choice:
            agent_type = np.random.choice(np.array(['MarketMaker4', config.agent_types[1:]], dtype=object),
                                          p=[config.p, 1 - config.p])

            if agent_type_previous:
                if (agent_type_previous == 'MarketMaker4') & (agent_type == 'MarketMaker4'):
                    # choose another agent_type
                    continue
                else:
                    # any other combination is fine
                    agent_type_previous = agent_type
            else:
                # first iteration
                agent_type_previous = agent_type

            random_agent = agent_portfolio.select_random_agent(agent_type)
            random_agent.price_fundamental = config.price_fundamental_new

            # print(random_agent.type)
            if agent_type == 'MarketMaker4':
                random_agent.update_forecast(price=orderbook.price[timestamp_last - 1, 1],
                                             ret=orderbook.ret[
                                                 timestamp_last -
                                                 config.initial_data_marketmaker['lookback'] * 2 - 1:timestamp_last - 1,
                                                 1],
                                             bids=orderbook.bids,
                                             asks=orderbook.asks,
                                             timestamp=timestamp_last)
                order_dicts_raw = random_agent.place_order(
                    price=orderbook.price[timestamp_last - 1, 1],
                    bids=orderbook.bids,
                    asks=orderbook.asks,
                    timestamp=timestamp_last)

            elif sorted(agent_type) == sorted(list(set(config.agent_types) - set(['MarketMaker4']))):
                random_agent.update_forecast(price=orderbook.price[timestamp_last - 1, 1],
                                             ret=orderbook.ret[timestamp_last - config.Lmax - 1:timestamp_last - 1, 1],
                                             timestamp=timestamp_last)
                order_dicts_raw = random_agent.place_order(
                    price=orderbook.price[timestamp_last - 1, 1],
                    ret=orderbook.ret[0:timestamp_last - 1, 1],
                    timestamp=timestamp_last,
                    bids=orderbook.bids,
                    asks=orderbook.asks,
                )

            else:
                # cannot happen
                raise Exception('Should not happen. Check code.')

            # check whether any submission is in order_dict
            if any([i['submit'] for i in order_dicts_raw]):
                trade = True

                # remove order_dicts with submit = False
                order_dicts = tuple([od for od in order_dicts_raw if od['submit']])
                # print(order_dicts)
                # for order_print in order_dicts:
                #     print(order_print)
            else:
                # cannot happen
                # print('No order during this tick.')
                continue

        # loop through orders
        for i in range(0, len(order_dicts)):
            # check whether order in tuple should be submitted
            if order_dicts[i]['submit']:
                # try:
                    # print('Orders message submitted: {}'.format(order_dicts[i]))
                #     print('State of orderbook:')
                    # print('{:<30}  {:^30}  {:>50}'.format('BIDs', 'MID', 'ASKs'))
                    # print('{:<30}  {:^30}  {:>50}'.format(np.array2string(orderbook.bids[0:1]),
                    #                                       np.array2string(orderbook.price[timestamp_last-1]),
                    #                                       np.array2string(orderbook.asks[0:1])))
                # except:
                    # print('Either bids or asks are empty.')
                # differentiate between combinations of modify and cancel
                if (not order_dicts[i]['modify']) & (not order_dicts[i]['cancel']):
                    # modify=False, cancel=False
                    # add order
                    # print('Add order')
                    fills = orderbook.add_order(side=order_dicts[i]['side'], lots=order_dicts[i]['lots'],
                                                type_=order_dicts[i]['type'], price=order_dicts[i]['order_price'],
                                                id=random_agent.id)

                elif (not order_dicts[i]['modify']) & (order_dicts[i]['cancel']):
                    # modify=False, cancel=True
                    # cancel order
                    # print('Cancel order')
                    fills = orderbook.cancel_order(order_dicts[i]['timestamp_modify'], modify=False)

                elif (order_dicts[i]['modify']) & (not order_dicts[i]['cancel']):
                    # modify=True, cancel=False
                    # modify order
                    # print('Modify order')
                    fills = orderbook.modify_order(timestamp=order_dicts[i]['timestamp_modify'],
                                                   side=order_dicts[i]['side'], lots=order_dicts[i]['lots'],
                                                   type_=order_dicts[i]['type'], price=order_dicts[i]['order_price'],
                                                   id=random_agent.id)

                else:
                    raise Exception('Case not evaluated.Check code.')

                # adjust agents holding after order was executed
                if fills['fills'].size > 0:
                    # fills exist
                    # print(fills)
                    match_ids = np.unique(fills['fills_orders'][:, 0])
                    # loop through match_ids
                    for idx, match_id in enumerate(match_ids):
                        # select fills_orders witch given match_id
                        fills_orders = fills['fills_orders'][fills['fills_orders'][:, 0] == match_id]

                        if len(fills_orders[fills_orders[:, 6] == 1]) > 1:
                            raise Exception('Market maker executes himself. Check code.')

                        # loop through fills within a match_id
                        for k in range(0, len(fills_orders)):
                            # order executed against order placed by agent

                            if fills_orders[k][6]:
                                # agent order got filled, adjust agent's wealth/orders
                                getattr(agent_portfolio, str(int(fills_orders[k][6]))).add_to_executed(
                                    timestamp_execution=fills['fills'][idx][0], lots=fills['fills'][idx][1], type_=None,
                                    price=fills['fills'][idx][2], timestamp=fills_orders[k][1])
                            else:
                                # order executed against initially set orders
                                pass

                        else:
                            # initial order was executed
                            pass

                else:
                    # no fills
                    pass
            else:
                # submit = False
                raise Exception('Not evaluated. Check code.')

        # removing if statement will result in every timestamp that orders, older than tau, will be removed
        if (rnd.uniform(0, 1) < config.cancel_threshold):# & (timestamp > 2000):
            # print('Cancel oldest orders')
            cleared_orders = orderbook.clean_oldest_orders(config.tau)
            # cleared_orders = orderbook.clean_last_orders(tau)

            if cleared_orders.size > 0:
                # select orders cleared against other agents
                cleared_agent_orders = cleared_orders[~np.isnan(cleared_orders[:, 4].astype(float))]

                if cleared_agent_orders.size > 0:
                    # orders cleared against other agents

                    # loop through cleared orders
                    for j in range(0, len(cleared_agent_orders)):
                        getattr(agent_portfolio, str(int(cleared_agent_orders[j][4]))).add_to_cancelled(
                            timestamp=cleared_agent_orders[j, 1])

                else:
                    # no agent orders cleared
                    pass
            else:
                # when no orders to clear
                pass

    # print('Run execution time: {}'.format(time.time() - start))

    return orderbook, agent_portfolio, config.agent_types
