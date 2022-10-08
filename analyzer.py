"""
Functions to help analysing AgentClass and OrderbookClass
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import re


def clean_price_data(orderbook):
    """
    Cleans orderbook.
    Removes zeros from orderbook price and return.
    :param orderbook: object
    :return: object
    """
    # remove last zeros created at the beginning
    price_temp = np.trim_zeros(orderbook.price[:, 1], 'b')
    orderbook.price = orderbook.price[0:len(price_temp)]

    ret_temp = np.trim_zeros(orderbook.ret[:, 1], 'b')
    # add zeros because it will trim zeros even though they should not be removed
    ret_temp = np.concatenate((ret_temp, np.zeros(len(price_temp) - len(ret_temp))))
    orderbook.ret = orderbook.ret[0:len(ret_temp)]

    return orderbook


def calc_shares(agent_portfolio, agent_types, orderbook, cutoff=2000):
    """
    Retrieves and aggregates amount of shares of individual agent types.
    :param agent_portfolio: object
    :param agent_types: list
    :param orderbook: object
    :param cutoff: int
    :return: dict
    """
    # pre-define dicts
    shares_temp = {}
    shares = {}

    # loop through agent types
    for agent_type in agent_types:
        # retrieve attribute from agent objects
        shares_temp.update({agent_type: [getattr(agent_, 'shares')
                                         for id_, agent_ in agent_portfolio.__dict__.items() if
                                         agent_.type == agent_type]})
        shares.update({agent_type: np.ones((len(orderbook.price) + 1, len(shares_temp[agent_type]))) * np.nan})

        # loop through agents and store in pre-defined array
        for i in range(0, len(shares_temp[agent_type])):
            shares[agent_type][shares_temp[agent_type][i][:, 0], i] = shares_temp[agent_type][i][:, 1]

        # forward fill missing values
        shares[agent_type] = pd.DataFrame(shares[agent_type]).replace(to_replace=np.nan, method='ffill').values
        # cut off
        shares[agent_type] = shares[agent_type][cutoff:, :]

    return shares


def calc_shares_dist(agent_portfolio, agent_types, orderbook, cutoff=2000, distribution_param=None, distribution=None):
    """
    Retrieves and aggregates amount of shares of individual agent types.
    :param agent_portfolio: object
    :param agent_types: list
    :param orderbook: object
    :param cutoff: int
    :param distribution_param: string
    :param distribution: list
    :return: dict
    """
    # filter out MarketMaker agent
    r = re.compile('MarketMaker*')
    agent_mm = list(filter(r.match, agent_types))[0]
    agent_types_wo_mm = list(set(agent_types) - set([agent_mm]))

    # pre-define dicts
    shares_temp = {agent: {} for agent in agent_types_wo_mm}
    shares = {agent: {} for agent in agent_types_wo_mm}

    # loop through agent types
    for agent_type in agent_types_wo_mm:

        # agent_type = agent_types[1]  # TODO
        for dist in distribution:
            # dist = distribution[0]  # TODO
            # retrieve attribute from agent objects
            shares_temp[agent_type].update(
                {dist:
                     [getattr(agent_, 'shares')
                      for id_, agent_ in agent_portfolio.__dict__.items()
                      if ((agent_.type == agent_type) & (getattr(agent_, distribution_param) == dist))]
                 }
            )
            shares[agent_type].update(
                {dist:
                     np.ones((len(orderbook.price) + 1, len(shares_temp[agent_type][dist]))) * np.nan
                 }
            )

            # loop through agents and store in pre-defined array
            for i in range(0, len(shares_temp[agent_type][dist])):
                shares[agent_type][dist][shares_temp[agent_type][dist][i][:, 0], i] = shares_temp[
                                                                                          agent_type][dist][i][:, 1]

            # forward fill missing values
            shares[agent_type][dist] = pd.DataFrame(shares[agent_type][dist]).replace(to_replace=np.nan, method='ffill').values
            # cut off
            shares[agent_type][dist] = shares[agent_type][dist][cutoff:, :]

    return shares


def calc_cash(agent_portfolio, agent_types, orderbook, cutoff=2000):
    """
    Retrieves and aggregates amount of cash of individual agent types.
    :param agent_portfolio: object
    :param agent_types: list
    :param orderbook: object
    :param cutoff: int
    :return: dict
    """
    # pre-define dicts
    cash_temp = {}
    cash = {}

    costs_temp = {}
    costs = {}

    # loop through agent types
    for agent_type in agent_types:
        # retrieve attribute from agent objects
        cash_temp.update({agent_type: [getattr(agent_, 'cash')
                                       for id_, agent_ in agent_portfolio.__dict__.items() if
                                       agent_.type == agent_type]})
        cash.update({agent_type: np.ones((len(orderbook.price) + 1, len(cash_temp[agent_type]))) * np.nan})

        try:
            costs_temp.update({agent_type: [getattr(agent_, 'orders_cost')
                                            for id_, agent_ in agent_portfolio.__dict__.items() if
                                            agent_.type == agent_type]})
            costs.update({agent_type: np.ones((len(orderbook.price) + 1, len(cash_temp[agent_type]))) * np.nan})
        except AttributeError:
            pass

        # loop through agents and store in pre-defined array
        for i in range(0, len(cash_temp[agent_type])):
            cash[agent_type][cash_temp[agent_type][i][:, 0].astype(int), i] = cash_temp[agent_type][i][:, 1]

            if costs:
                try:
                    # cumulate here otherwise forward filled values will be comulated
                    costs[agent_type][costs_temp[agent_type][i][:, 0].astype(int), i] = np.cumsum(
                        costs_temp[agent_type][i][:, 1])
                except IndexError:
                    # if no transactions for a given agent, orders_costs will be empty
                    pass
            else:
                pass

        # forward fill missing values
        cash[agent_type] = pd.DataFrame(cash[agent_type]).replace(to_replace=np.nan, method='ffill').values
        # cut off
        cash[agent_type] = cash[agent_type][cutoff:, :]

        if costs:
            # forward fill missing values
            costs[agent_type] = pd.DataFrame(costs[agent_type]).replace(to_replace=np.nan, method='ffill').values
            # fill nans with zero, leading zeros
            costs[agent_type] = pd.DataFrame(costs[agent_type]).replace(np.nan, 0).values
            # cutoff
            costs[agent_type] = costs[agent_type][cutoff:, :]

            # cumulate costs and add to cash for separation
            cash[agent_type] = cash[agent_type] - costs[agent_type]

        else:
            pass

    return cash, costs


def calc_cash_dist(agent_portfolio, agent_types, orderbook, cutoff=2000, distribution_param=None, distribution=None):
    """
    Retrieves and aggregates amount of cash of individual agent types.
    :param agent_portfolio: object
    :param agent_types: list
    :param orderbook: object
    :param cutoff: int
    :param distribution_param: string
    :param distribution: list
    :return: dict
    """
    # filter out MarketMaker agent
    r = re.compile('MarketMaker*')
    agent_mm = list(filter(r.match, agent_types))[0]
    agent_types_wo_mm = list(set(agent_types) - set([agent_mm]))

    # pre-define dicts
    cash_temp = {agent: {} for agent in agent_types_wo_mm}
    cash = {agent: {} for agent in agent_types_wo_mm}

    costs_temp = {agent: {} for agent in agent_types_wo_mm}
    costs = {agent: {} for agent in agent_types_wo_mm}

    # loop through agent types
    for agent_type in agent_types_wo_mm:

        # agent_type = agent_types_wo_mm[1]  # TODO
        for dist in distribution:
            # dist = distribution[0]  # TODO
            # retrieve attribute from agent objects
            cash_temp[agent_type].update(
                {dist:
                     [getattr(agent_, 'cash')
                      for id_, agent_ in agent_portfolio.__dict__.items()
                      if ((agent_.type == agent_type) & (
                             getattr(agent_, distribution_param) == dist))]
                 }
            )
            cash[agent_type].update(
                {dist:
                     np.ones((len(orderbook.price) + 1, len(cash_temp[agent_type][dist]))) * np.nan
                 }
            )

            try:
                costs_temp[agent_type].update(
                    {dist:
                         [getattr(agent_, 'orders_cost')
                          for id_, agent_ in agent_portfolio.__dict__.items()
                          if ((agent_.type == agent_type) & (
                                 getattr(agent_, distribution_param) == dist))]
                     }
                )
                costs[agent_type].update(
                    {dist:
                         np.ones(
                             (len(orderbook.price) + 1, len(cash_temp[agent_type][dist]))) * np.nan
                     }
                )
            except AttributeError:
                pass

            # loop through agents and store in pre-defined array
            for i in range(0, len(cash_temp[agent_type][dist])):
                cash[agent_type][dist][cash_temp[agent_type][dist][i][:, 0].astype(int), i] = cash_temp[
                                                                                          agent_type][dist][
                                                                                          i][:, 1]

                if costs:
                    try:
                        # cumulate here otherwise forward filled values will be comulated
                        costs[agent_type][dist][costs_temp[agent_type][dist][i][:, 0].astype(int), i] = np.cumsum(
                            costs_temp[agent_type][dist][i][:, 1])
                    except IndexError:
                        # if no transactions for a given agent, orders_costs will be empty
                        pass
                else:
                    pass

            # forward fill missing values
            cash[agent_type][dist] = pd.DataFrame(cash[agent_type][dist]).replace(to_replace=np.nan,
                                                                                      method='ffill').values
            # cut off
            cash[agent_type][dist] = cash[agent_type][dist][cutoff:, :]

            if costs:
                # forward fill missing values
                costs[agent_type][dist] = pd.DataFrame(costs[agent_type][dist]).replace(
                    to_replace=np.nan, method='ffill').values
                # fill nans with zero, leading zeros
                costs[agent_type][dist] = pd.DataFrame(costs[agent_type][dist]).replace(np.nan, 0).values
                # cutoff
                costs[agent_type][dist] = costs[agent_type][dist][cutoff:, :]

                # cumulate costs and add to cash for separation
                cash[agent_type][dist] = cash[agent_type][dist] - costs[agent_type][dist]

            else:
                pass

    return cash, costs


def calc_wealth(shares, cash, costs, agent_types, orderbook, cutoff=2000):
    """
    Aggregates shares and cash to wealth.
    :param shares: np.array
    :param cash: np.array
    :param costs: np.array
    :param agent_types: list
    :param orderbook: object
    :param cutoff: int
    :return: dict
    """
    # pre-define dicts
    wealth = {}
    wealth_evolution = {}

    # TODO: agents which have slightly negative wealth lead to inf wealth evolution

    # loop through agent types
    for agent_type in agent_types:
        # shares are allocated at t=0 while orderbook beings at t=1
        if costs:
            wealth.update({agent_type: np.expand_dims(orderbook.price[cutoff:, 1], 1) * shares[agent_type][1:] +
                                       cash[agent_type][1:] + costs[agent_type][1:]})
        else:
            wealth.update({agent_type: np.expand_dims(orderbook.price[cutoff:, 1], 1) * shares[agent_type][1:] +
                                       cash[agent_type][1:]})

        wealth_evolution.update({agent_type:
                                     np.cumprod(1 + np.diff(wealth[agent_type], axis=0) / wealth[agent_type][1:, :],
                                                axis=0)})

    return wealth, wealth_evolution


def calc_wealth_dist(shares, cash, costs, agent_types, orderbook, cutoff=2000,
                     distribution_param=None, distribution=None):
    """
    Aggregates shares and cash to wealth.
    :param shares: np.array
    :param cash: np.array
    :param costs: np.array
    :param agent_types: list
    :param orderbook: object
    :param cutoff: int
    :param distribution_param: string
    :param distribution: list
    :return: dict
    """
    # filter out MarketMaker agent
    r = re.compile('MarketMaker*')
    agent_mm = list(filter(r.match, agent_types))[0]
    agent_types_wo_mm = list(set(agent_types) - set([agent_mm]))

    # pre-define dicts
    wealth = {agent: {} for agent in agent_types_wo_mm}
    wealth_evolution = {agent: {} for agent in agent_types_wo_mm}
    # TODO: agents which have slightly negative wealth lead to inf wealth evolution

    # loop through agent types
    for agent_type in agent_types_wo_mm:
        # agent_type = agent_types[1]  # TODO
        for dist in distribution:
            # dist = distribution[-1]  # TODO
            # shares are allocated at t=0 while orderbook beings at t=1
            if costs:
                wealth[agent_type].update(
                    {dist:
                         np.expand_dims(orderbook.price[cutoff:, 1], 1) *
                         shares[agent_type][dist][1:] +
                         cash[agent_type][dist][1:] + costs[agent_type][dist][1:]
                     }
                )
            else:
                wealth[agent_type].update(
                    {dist: np.expand_dims(orderbook.price[cutoff:, 1], 1) *
                           shares[agent_type][dist][1:] +
                           cash[agent_type][dist][1:]
                     }
                )

            wealth_evolution[agent_type].update(
                {dist:
                     np.cumprod(1 + np.diff(wealth[agent_type][dist], axis=0) /
                                wealth[agent_type][dist][1:, :],
                                axis=0)
                 }
            )

    return wealth, wealth_evolution


def calc_wealth_share(shares, cash, costs, agent_types, orderbook, cutoff=2000):
    """
    Aggregates shares and cash to wealth.
    :param shares: np.array
    :param cash: np.array
    :param agent_types: list
    :param orderbook: object
    :param cutoff: int
    :return: dict
    """
    # pre-define dicts
    wealth_share_temp = {}
    wealth_share_sum = {}
    wealth_share_simple = {}
    wealth_share_normalized = {}

    # calculate sum of wealth by agent type
    # loop through agent types
    for agent_type in agent_types:
        # agent_type = agent_types[0]
        # shares are allocated at t=0 while orderbook beings at t=1
        wealth_share_temp.update({agent_type: np.expand_dims(orderbook.price[cutoff:, 1], 1) * shares[agent_type][1:] +
                                   cash[agent_type][1:] + costs[agent_type][1:]})
        wealth_share_sum.update({agent_type: wealth_share_temp[agent_type].sum(axis=1)})

    # calculate wealth share neglecting different numbers of agents
    # calculate total wealth across all agents
    wealth_sum = sum(wealth_share_sum[agent_type] for agent_type in agent_types)

    # loop through agent types
    for id_, agent_type in enumerate(agent_types):
        wealth_share_simple.update({agent_type: wealth_share_sum[agent_type] / wealth_sum})

    # calculate wealth share normalized to 1
    # calculate factor that is applied to every data point that normalizes the wealth shares to one and let them
    #   start at equally distributed shares
    wealth_share_factor = [(1/len(agent_types)) / wealth_share_simple[agent_type] for agent_type in agent_types]

    # loop through agent types
    for id_, agent_type in enumerate(agent_types):
        wealth_share_normalized.update({agent_type: wealth_share_simple[agent_type] * wealth_share_factor[id_][0]})

    return wealth_share_simple, wealth_share_normalized


def calc_wealth_share_dist(shares, cash, costs, agent_types, orderbook, cutoff=2000,
                           distribution_param=None, distribution=None):
    """
    Aggregates shares and cash to wealth.
    :param shares: np.array
    :param cash: np.array
    :param agent_types: list
    :param orderbook: object
    :param cutoff: int
    :param distribution_param: string
    :param distribution: list
    :return: dict
    """
    # filter out MarketMaker agent
    r = re.compile('MarketMaker*')
    agent_mm = list(filter(r.match, agent_types))[0]
    agent_types_wo_mm = list(set(agent_types) - set([agent_mm]))

    # pre-define dicts
    wealth_share_temp = {agent: {} for agent in agent_types_wo_mm}
    wealth_share_sum = {agent: {} for agent in agent_types_wo_mm}
    wealth_share_simple = {agent: {} for agent in agent_types_wo_mm}
    wealth_share_normalized = {agent: {} for agent in agent_types_wo_mm}

    # calculate sum of wealth by agent type
    # loop through agent types
    for agent_type in agent_types_wo_mm:
        # agent_type = agent_types[1]  # TODO
        for dist in distribution:
            # dist = distribution[-1]  # TODO
            # agent_type = agent_types[0]
            # shares are allocated at t=0 while orderbook beings at t=1
            wealth_share_temp[agent_type].update(
                {dist:
                     np.expand_dims(orderbook.price[cutoff:, 1], 1) * shares[agent_type][dist][1:] +
                     cash[agent_type][dist][1:] + costs[agent_type][dist][1:]
                 }
            )

            wealth_share_sum[agent_type].update(
                {dist:
                     wealth_share_temp[agent_type][dist].sum(axis=1)
                 }
            )

    # calculate wealth share neglecting different numbers of agents
    # calculate total wealth across all agents
    wealth_sum = sum(wealth_share_sum[agent_type][dist]
                     for dist in distribution
                     for agent_type in agent_types_wo_mm)

    # loop through agent types
    for id_, agent_type in enumerate(agent_types_wo_mm):
        for dist in distribution:
            wealth_share_simple[agent_type].update({dist: wealth_share_sum[agent_type][dist] / wealth_sum})

    # calculate wealth share normalized to 1
    # calculate factor that is applied to every data point that normalizes the wealth shares to one and let them
    #   start at equally distributed shares
    wealth_share_factor = {agent_type:
                               {dist:
                                    (1 / (len(agent_types_wo_mm) * len(distribution))) /
                                    wealth_share_simple[agent_type][dist] for dist in distribution
                                } for agent_type in agent_types_wo_mm
                           }

    # loop through agent types
    for agent_type in agent_types_wo_mm:
        for dist in distribution:
            wealth_share_normalized[agent_type].update(
                {dist:
                     wealth_share_simple[agent_type][dist] * wealth_share_factor[agent_type][dist][0]
                 }
            )

    return wealth_share_simple, wealth_share_normalized


def normalize_wealth_share(wealth_share_simple, agent_types, period_init, period_max, cutoff_bool=False, cutoff=2000):
    """
    Calculates normalized wealth share by first averaging simple wealth shares and then normalizing them by their
    starting values.
    :param wealth_share_simple: dict, keys are indices indicating the run and values are dicts containing simple wealth
    shares of all agent types for each run
    :param agent_types: list
    :param period_init: int
    :param period_max: int
    :param cutoff_bool: boolean
    :param cutoff: int
    :return: dict, dict
    """
    # calculate average over all runs
    # TODO: note that behaviour is not specified if cutoff is zero
    if not cutoff_bool:
        wealth_share_simple_avg = {agent_type: np.array(list(zip(*(
            value[agent_type][period_init:period_max]
            for key, value in wealth_share_simple.items())
                                                                 ))).mean(axis=1)
                                   for agent_type in agent_types}
    else:
        # need to deduct cutoff from period_max as the cutoff was already taken care of before
        wealth_share_simple_avg = {agent_type: np.array(list(zip(*(
            value[agent_type][:period_max-cutoff]
            for key, value in wealth_share_simple.items())
                                                                 ))).mean(axis=1)
                                   for agent_type in agent_types}

    # calculate wealth share factor by normalizing
    # wealth_share_factor = [(1 / len(agent_types)) / wealth_share_simple_avg[agent_type][0] for agent_type in
    #                        agent_types]

    wealth_share_factor = {agent_type: (1 / len(agent_types)) / wealth_share_simple_avg[agent_type] for agent_type in
                           agent_types}

    # loop through agent types
    wealth_share_normalized = {}
    for id_, agent_type in enumerate(agent_types):
        # print(agent_type)
        # wealth_share_normalized.update({agent_type: wealth_share_simple_avg[agent_type] * wealth_share_factor[id_]})
        wealth_share_normalized.update({agent_type: wealth_share_simple_avg[agent_type] * wealth_share_factor[agent_type][0]})


    return wealth_share_normalized, wealth_share_simple_avg


def normalize_wealth_share_dist(wealth_share_simple, agent_types, period_init, period_max, cutoff_bool=False,
                                cutoff=2000, distribution_param=None, distribution=None):
    """
    Calculates normalized wealth share by first averaging simple wealth shares and then normalizing them by their
    starting values.
    :param wealth_share_simple: dict, keys are indices indicating the run and values are dicts containing simple wealth
    shares of all agent types for each run
    :param agent_types: list
    :param period_init: int
    :param period_max: int
    :param cutoff_bool: boolean
    :param cutoff: int
    :param distribution_param: string
    :param distribution: list
    :return: dict, dict
    """
    # calculate average over all runs
    # TODO: note that behaviour is not specified if cutoff is zero

    # filter out MarketMaker agent
    r = re.compile('MarketMaker*')
    agent_mm = list(filter(r.match, agent_types))[0]
    agent_types_wo_mm = list(set(agent_types) - set([agent_mm]))

    # pre-define dicts
    wealth_share_normalized = {agent: {} for agent in agent_types_wo_mm}

    if not cutoff_bool:
        wealth_share_simple_avg = {agent_type:
                                       {dist:
                                            np.array(list(zip(*(value[agent_type][dist][period_init:period_max]
                                                                for key, value in wealth_share_simple.items())
                                                              ))).mean(axis=1) for dist in distribution
                                        } for agent_type in agent_types_wo_mm
                                   }

    else:
        # need to deduct cutoff from period_max as the cutoff was already taken care of before
        wealth_share_simple_avg = {agent_type:
                                       {dist:
                                            np.array(list(zip(*(
                                                value[agent_type][dist][:period_max - cutoff]
                                                for key, value in wealth_share_simple.items())
                                                              ))).mean(axis=1) for dist in distribution
                                        } for agent_type in agent_types_wo_mm
                                   }

    # calculate wealth share factor by normalizing
    wealth_share_factor = {agent_type:
                               {dist:
                                    (1 / (len(agent_types_wo_mm) * len(distribution))) /
                                    wealth_share_simple_avg[agent_type][dist] for dist in distribution
                                } for agent_type in agent_types_wo_mm
                           }

    # loop through agent types
    for agent_type in agent_types_wo_mm:
        for dist in distribution:
            wealth_share_normalized[agent_type].update(
                {dist:
                     wealth_share_simple_avg[agent_type][dist] * wealth_share_factor[agent_type][dist]
                 }
            )

    return wealth_share_normalized, wealth_share_simple_avg


def calc_autocorr(series_, t=100):
    """
    Calculates autocorrlation function.
    :param series_: np.array, 1d
    :param t: int
    :return: np.array
    """
    autocorr = np.array([1] + [np.corrcoef(series_[:-i], series_[i:])[0, 1] for i in range(1, t + 1)])
    # https: // www.statsmodels.org / devel / generated / statsmodels.tsa.stattools.acf.html
    # autocorr2 = smt.tsa.stattools.acf(series_, nlags=t, fft=True)

    return autocorr


def calc_stats_shares(ts, start, end):
    """
    Calculate statistics based on time series of inventory or cash.
    :param ts: dict of arrays, can be shares or cash
    :param start: int
    :param end: int
    :return: dict
    """
    stats_shares = {}
    for agent_type in ts.keys():
        # print(agent_type)
        ts_agent = ts[agent_type][start:end]

        mean_shares = np.mean(np.mean(ts_agent, axis=0))
        std_shares = np.mean(np.std(ts_agent, axis=0))
        var_shares = np.mean(np.var(ts_agent, axis=0))
        median_shares = np.mean(np.median(ts_agent, axis=0))

        diff_ = np.diff(ts_agent, axis=0)
        num_trades_per_agent = np.mean(np.count_nonzero(diff_, axis=0))
        shares_per_trade_avg = np.nanmean(np.array([np.mean(np.abs(diff_[:, j][diff_[:, j] != 0]))
                                                    for j in range(0, ts_agent.shape[1])]))
        shares_per_trade_min = np.min(np.where(np.abs(diff_) > 0)[1])
        shares_per_trade_max = np.max(np.where(np.abs(diff_) > 0)[1])
        total_shares_traded_per_agent = np.mean(np.sum(np.abs(diff_), axis=0))

        # taking only trades in account, i.e. removing non-trading timestamps
        std_trades = np.nanmean(np.array([np.std(diff_[:, j][diff_[:, j] != 0])
                                          for j in range(0, ts_agent.shape[1])]))
        var_trades = np.nanmean(np.array([np.var(diff_[:, j][diff_[:, j] != 0])
                                          for j in range(0, ts_agent.shape[1])]))
        skew_trades = np.nanmean(np.array([skew(diff_[:, j][diff_[:, j] != 0])
                                          for j in range(0, ts_agent.shape[1])]))
        kurt_trades = np.nanmean(np.array([kurtosis(diff_[:, j][diff_[:, j] != 0])
                                          for j in range(0, ts_agent.shape[1])]))

        stats_shares.update({agent_type: {
            'mean_shares': mean_shares,
            'std_shares': std_shares,
            'var_shares': var_shares,
            'median_shares': median_shares,
            'num_trades_per_agent': num_trades_per_agent,
            'shares_per_trade_avg': shares_per_trade_avg,
            'shares_per_trade_min': shares_per_trade_min,
            'shares_per_trade_max': shares_per_trade_max,
            'total_shares_traded_avg': total_shares_traded_per_agent,
            'var_trades': var_trades,
            'std_trades': std_trades,
            'skew_trades': skew_trades,
            'kurt_trades': kurt_trades,
        }})

    return stats_shares


def calc_stats_wealth(ts, start, end):
    """
    Calculate statistics based on time series of inventory or cash.
    :param ts: dict of arrays
    :param start: int
    :param end: int
    :return: dict
    """
    stats_wealth = {}
    for agent_type in ts.keys():
        # print(agent_type)
        # ts_agent = ts[agent_type][orderbook.period_init:orderbook.period_max]
        ts_agent = ts[agent_type][start:end]
        ts_agent = ts_agent[:, ~np.isinf(ts_agent).any(axis=0)]  # remove inf
        ts_agent = ts_agent[:, ~np.isnan(ts_agent).any(axis=0)]  # remove nan
        ts_agent = ts_agent[:, ~(ts_agent <= 0).any(axis=0)]  # remove negative and zeros

        # small outlier detection
        d = np.abs(ts_agent - np.median(ts_agent, axis=0))
        mdev = np.nanmedian(d)
        # s = d / mdev if mdev else 0.
        s = d / mdev if mdev else np.zeros(d.shape)
        ts_agent = ts_agent[:, (s < 10.).any(axis=0)]
        # aaa = ts_agent[:, ~(s < 10.).any(axis=0)]

        mean_ = np.mean(np.mean(ts_agent - 1, axis=0))
        geometric_mean = np.mean((ts_agent[-1, :]) ** (1 / len(ts_agent)) - 1)
        std_ = np.mean(np.std(ts_agent - 1, axis=0))
        var_ = np.mean(np.var(ts_agent - 1, axis=0))
        median_ = np.mean(np.median(ts_agent - 1, axis=0))
        sharpe = geometric_mean / std_  # TODO: not correctly calculated, calculate first sharpe for every agent and then average them
        skew_ = np.mean(skew(ts_agent - 1))
        kurt_ = np.mean(kurtosis(ts_agent - 1))

        stats_wealth.update({agent_type: {
            'mean': mean_,
            'std': std_,
            'var': var_,
            'median': median_,
            'geometric_mean': geometric_mean,
            'sharpe': sharpe,
            'skew': skew_,
            'kurt': kurt_,
        }})

        if (abs(np.array(list(stats_wealth[agent_type].values()))) > 10 ** 6).any():
            print('ISSUE')
            raise Exception('Wealth moments are too large. Break code.')

    return stats_wealth


def merge(a, b, path=None):
    """
    merges b into a
    :param a: dict
    :param b: dict
    :param path:
    :return: dict
    """
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a
