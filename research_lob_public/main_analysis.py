"""
Loops through orderbook simulations, calculating and storing statistics. Used for 1 test run.
"""

import analyzer as a
import plots as pl
import main_orderbook as mo
import pickle
import time
import numpy as np
import pandas as pd
import re
from scipy.stats import skew, kurtosis
from functools import reduce
import statsmodels.tsa.stattools as ts
import plotly.io as pio
import warnings
import copy
import os


pio.renderers.default = "browser"
warnings.filterwarnings("ignore")

PATH = 'C:/Users/WOstr/PycharmProjects/data/research_orderbook/june_2022b/'


def run_single(name, i, local_path, **params):
    """
    Runs single simulation and stores in locally.
    :param name: string
    :param i: int
    :param local_path: boolean
    :param params: dict
    :return: None
    """
    print(params)
    params_init = copy.deepcopy(params)

    start = time.time()

    # %% simulation
    # double deepcopy to avoid overwriting params (sensitivity_00) - happens when runs > 1
    params_sim = copy.deepcopy(params_init)
    orderbook, agent_portfolio, agent_types = mo.main(params_sim['sensitivity_base'])

    # store
    # pickle.dump(orderbook, open('C:/Users/WOstr/PycharmProjects/research_abm/abm_simulation/LOB/storage/'
    #                             + name + '_orderbook_' + str(i) + '.p', 'wb'))
    # pickle.dump(agent_portfolio, open('C:/Users/WOstr/PycharmProjects/research_abm/abm_simulation/LOB/storage/'
    #                                   + name + '_agent_portfolio_' + str(i) + '.p', 'wb'))

    # %% analysis
    orderbook = a.clean_price_data(orderbook)

    shares_temp = a.calc_shares(agent_portfolio, agent_types, orderbook, params_sim['sensitivity_base'].cutoff)
    cash_temp, costs_temp = a.calc_cash(agent_portfolio, agent_types, orderbook,
                                        params_sim['sensitivity_base'].cutoff)
    wealth_temp, wealth_evolution_temp = a.calc_wealth(shares_temp, cash_temp, costs_temp, agent_types, orderbook,
                                                        params_sim['sensitivity_base'].cutoff)
    wealth_share_simple_temp, wealth_share_normalized_temp = a.calc_wealth_share(
        shares_temp, cash_temp, costs_temp,
        agent_types, orderbook,
        params_sim['sensitivity_base'].cutoff)

    # dump variables temporarily as dict of dicts
    dict_temp = {
        'params_sim_dict': {i: params_sim},
        'orderbook_dict': {i: orderbook},
        # 'agent_portfolio_dict': {i: agent_portfolio},
        'agent_portfolio_dict': {i: list(set([obj.type for id, obj in agent_portfolio.__dict__.items()]))},
        'shares': {i: shares_temp},
        'cash': {i: cash_temp},
        'costs': {i: costs_temp},
        'wealth': {i: wealth_temp},
        'wealth_evolution': {i: wealth_evolution_temp},
        'wealth_share_simple': {i: wealth_share_simple_temp},
        'wealth_share_normalized': {i: wealth_share_normalized_temp},
    }

    # s = time.time()
    pickle.dump(dict_temp, open(local_path + 'results_temp/' + name + '_' + str(i) + '.p', 'wb'))

    print('Execution time (run={}): {}'.format(i, time.time() - start))

    return None


def run(name=None, runs=1, local_path=None, **params):
    """
    Loops through iterations and triggers single run.
    :param name:
    :param runs:
    :param local_path:
    :param params:
    :return: None
    """

    start = time.time()

    for i in range(0, runs):
        run_single(name, i, local_path=local_path, **params)

    print('Total execution time (runs): {}'.format(time.time() - start))

    return None


def run_stats(name=None, runs=1, local_path=None, **params):
    """
    Calculate statistics iteratively to avoid loading all data in memory - loads single simulation at a time.
    :param name: string
    :param runs: int
    :param local_path: string
    :param params: dict
    :return: None
    """
    start = time.time()

    for i in range(0, runs):
        # load pickles and re-create dict
        pickles = pickle.load(open(local_path + 'results_temp/' + name + '_' + str(i) + '.p', 'rb'))

        # delete pickled files
        os.remove(local_path + 'results_temp/' + name + '_' + str(i) + '.p')

        if i == 0:
            # unpack some attributes that are same across simulations
            agent_types = pickles['agent_portfolio_dict'][i]

            period_init = pickles['orderbook_dict'][i].period_init
            period_max = pickles['orderbook_dict'][i].period_max
            cutoff = pickles['params_sim_dict'][i]['sensitivity_base'].cutoff

            # define start:end in as dict here first to avoid repeating
            start_end_orderbook = {}
            dict_temp = {
                'start': pickles['orderbook_dict'][i].period_init
                if not pickles['params_sim_dict'][i]['sensitivity_base'].cutoff
                else pickles['params_sim_dict'][i]['sensitivity_base'].cutoff,
                'end': pickles['orderbook_dict'][i].period_max
            }
            start_end_orderbook.update({i: dict_temp})

            start_end_wealth = {}
            dict_temp = {
                'start': pickles['orderbook_dict'][i].period_init
                if not pickles['params_sim_dict'][i]['sensitivity_base'].cutoff else 0,
                'end': pickles['orderbook_dict'][i].period_max
                if not pickles['params_sim_dict'][i]['sensitivity_base'].cutoff
                else pickles['orderbook_dict'][i].period_max - pickles['params_sim_dict'][i]['sensitivity_base'].cutoff}
            start_end_wealth.update({i: dict_temp})

            # %% average out statistics
            wealth_share_normalized_avg, wealth_share_simple_avg = a.normalize_wealth_share(
                pickles['wealth_share_simple'],
                agent_types,
                period_init, period_max,
                True, cutoff)

            autocorr_raw_avg = a.calc_autocorr(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
            autocorr_abs_avg = a.calc_autocorr(
                abs(pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1]))

            # %% statistics
            # shares, cash and wealth
            stats_shares = a.calc_stats_shares(pickles['shares'][i],
                                               start=start_end_wealth[i]['start'],
                                               end=start_end_wealth[i]['end'])
            stats_cash = a.calc_stats_shares(pickles['cash'][i],
                                             start=start_end_wealth[i]['start'],
                                             end=start_end_wealth[i]['end'])
            stats_wealth = a.calc_stats_wealth(pickles['wealth_evolution'][i],
                                               start=start_end_wealth[i]['start'],
                                               end=start_end_wealth[i]['end'])

            # add wealth_share_normalized_avg
            stats_wealth = reduce(a.merge,
                                  [stats_wealth, {key: {'wealth_share_normalized_avg': wealth_share_normalized_avg[key][-1]}
                                                  for key in wealth_share_normalized_avg.keys()}])

            mean_ = np.mean(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
            # geometric_mean = (pickles['orderbook_dict'][i].price[start_end_orderbook[i]['end']][1] /
            #                            pickles['orderbook_dict'][i].price[start_end_orderbook[i]['start']][1]) ** (
            #                                   start_end_orderbook[i]['end'] / start_end_orderbook[i]['start']) - 1
            geometric_mean = (pickles['orderbook_dict'][i].price[start_end_orderbook[i]['end']][1] /
                                       pickles['orderbook_dict'][i].price[start_end_orderbook[i]['start']][1]) ** (
                                              1 / (start_end_orderbook[i]['end'] - start_end_orderbook[i]['start'])) - 1
            median_ = np.median(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
            std_ = np.std(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
            var_ = np.var(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
            skew_ = skew(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
            kurt_ = kurtosis(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])

            ac1_raw = autocorr_raw_avg[1]
            ac2_raw = autocorr_raw_avg[2]
            ac3_raw = autocorr_raw_avg[3]

            ac1_abs = autocorr_abs_avg[1]
            ac20_abs = autocorr_abs_avg[20]
            ac50_abs = autocorr_abs_avg[50]
            ac100_abs = autocorr_abs_avg[100]

            adf = ts.adfuller(pickles['orderbook_dict'][i].ret[
                              start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])[0]

            # for plots
            price = pickles['orderbook_dict'][i].price[
                    start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1]
            return_ = pickles['orderbook_dict'][i].ret[
                      start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1]

            price_all = [pickles['orderbook_dict'][i].price[
                         start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1]]

            r = re.compile('MarketMaker*')
            agent_mm = list(filter(r.match, agent_types))[0]
            shares_mm = pickles['shares'][i][agent_mm][start_end_wealth[i]['start']:start_end_wealth[i]['end']]
            cash_mm = pickles['cash'][i][agent_mm][start_end_wealth[i]['start']:start_end_wealth[i]['end']]
            costs_mm = pickles['costs'][i][agent_mm][start_end_wealth[i]['start']:start_end_wealth[i]['end']]
            wealth_mm = pickles['wealth'][i][agent_mm][start_end_wealth[i]['start']:start_end_wealth[i]['end']]
            price_mm = pickles['orderbook_dict'][i].price[start_end_orderbook[i]['start']:
                                                          start_end_orderbook[i]['end']][:, 1]

        else:
            # unpack some attributes that are same across simulations
            agent_types = pickles['agent_portfolio_dict'][i]

            period_init = pickles['orderbook_dict'][i].period_init
            period_max = pickles['orderbook_dict'][i].period_max
            cutoff = pickles['params_sim_dict'][i]['sensitivity_base'].cutoff

            # define start:end in as dict here first to avoid repeating
            start_end_orderbook = {}
            dict_temp = {
                'start': pickles['orderbook_dict'][i].period_init
                if not pickles['params_sim_dict'][i]['sensitivity_base'].cutoff
                else pickles['params_sim_dict'][i]['sensitivity_base'].cutoff,
                'end': pickles['orderbook_dict'][i].period_max
            }
            start_end_orderbook.update({i: dict_temp})

            start_end_wealth = {}
            dict_temp = {
                'start': pickles['orderbook_dict'][i].period_init
                if not pickles['params_sim_dict'][i]['sensitivity_base'].cutoff else 0,
                'end': pickles['orderbook_dict'][i].period_max
                if not pickles['params_sim_dict'][i]['sensitivity_base'].cutoff
                else pickles['orderbook_dict'][i].period_max - pickles['params_sim_dict'][i]['sensitivity_base'].cutoff}
            start_end_wealth.update({i: dict_temp})

            factor_1 = ((i + 1) - 1) / (i + 1)
            factor_2 = 1 / (i + 1)

            # %% average out statistics
            wealth_share_normalized_avg_temp, wealth_share_simple_avg_temp = a.normalize_wealth_share(
                pickles['wealth_share_simple'],
                agent_types,
                period_init, period_max,
                True, cutoff)
            wealth_share_normalized_avg.update((key, factor_1 * wealth_share_normalized_avg[key] + factor_2 * value)
                                               for key, value in wealth_share_normalized_avg_temp.items())
            wealth_share_simple_avg.update((key, factor_1 * wealth_share_simple_avg[key] + factor_2 * value)
                                           for key, value in wealth_share_simple_avg_temp.items())

            autocorr_raw_avg = factor_1 * autocorr_raw_avg + factor_2 * a.calc_autocorr(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
            autocorr_abs_avg = factor_1 * autocorr_abs_avg + factor_2 * a.calc_autocorr(
                abs(pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:,
                    1]))

            # %% statistics
            # shares, cash and wealth
            stats_shares_temp = a.calc_stats_shares(pickles['shares'][i],
                                                    start=start_end_wealth[i]['start'],
                                                    end=start_end_wealth[i]['end'])
            for agent_type, dict_ in stats_shares_temp.items():
                for stat, value in dict_.items():
                    stats_shares[agent_type].update(
                        {stat:
                             factor_1 * stats_shares[agent_type][stat] + factor_2 * value
                         }
                    )
            stats_cash_temp = a.calc_stats_shares(pickles['cash'][i],
                                                  start=start_end_wealth[i]['start'],
                                                  end=start_end_wealth[i]['end'])
            for agent_type, dict_ in stats_cash_temp.items():
                for stat, value in dict_.items():
                    stats_cash[agent_type].update(
                        {stat:
                             factor_1 * stats_cash[agent_type][stat] + factor_2 * value
                         }
                    )

            stats_wealth_temp = a.calc_stats_wealth(pickles['wealth_evolution'][i],
                                                    start=start_end_wealth[i]['start'],
                                                    end=start_end_wealth[i]['end'])
            for agent_type, dict_ in stats_wealth_temp.items():
                for stat, value in dict_.items():
                    stats_wealth[agent_type].update(
                        {stat:
                             factor_1 * stats_wealth[agent_type][stat] + factor_2 * value
                         }
                    )

            # add wealth_share_normalized_avg
            for agent_type, dict_ in stats_wealth.items():
                stats_wealth[agent_type].update(
                    {'wealth_share_normalized_avg':
                         factor_1 * stats_wealth[agent_type]['wealth_share_normalized_avg'] +
                         factor_2 * wealth_share_normalized_avg[agent_type][-1]})

            mean_ = factor_1 * mean_ + factor_2 * np.mean(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])

            # geometric_mean = factor_1 * geometric_mean + factor_2 * ((
            #         pickles['orderbook_dict'][i].price[start_end_orderbook[i]['end']][1] /
            #         pickles['orderbook_dict'][i].price[start_end_orderbook[i]['start']][1]) ** (
            #                          start_end_orderbook[i]['end'] / start_end_orderbook[i]['start']) - 1)
            geometric_mean = factor_1 * geometric_mean + factor_2 * ((
                pickles['orderbook_dict'][i].price[start_end_orderbook[i]['end']][1] /
                                       pickles['orderbook_dict'][i].price[start_end_orderbook[i]['start']][1]) ** (
                                              1 / (start_end_orderbook[i]['end'] - start_end_orderbook[i]['start'])) - 1)
            median_ = factor_1 * median_ + factor_2 * np.median(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
            std_ = factor_1 * std_ + factor_2 * np.std(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
            var_ = factor_1 * var_ + factor_2 * np.var(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
            skew_ = factor_1 * skew_ + factor_2 * skew(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
            kurt_ = factor_1 * kurt_ + factor_2 * kurtosis(
                pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])

            ac1_raw = factor_1 * ac1_raw + factor_2 * autocorr_raw_avg[1]
            ac2_raw = factor_1 * ac2_raw + factor_2 * autocorr_raw_avg[2]
            ac3_raw = factor_1 * ac3_raw + factor_2 * autocorr_raw_avg[3]

            ac1_abs = factor_1 * ac1_abs + factor_2 * autocorr_abs_avg[1]
            ac20_abs = factor_1 * ac20_abs + factor_2 * autocorr_abs_avg[20]
            ac50_abs = factor_1 * ac50_abs + factor_2 * autocorr_abs_avg[50]
            ac100_abs = factor_1 * ac100_abs + factor_2 * autocorr_abs_avg[100]

            adf = factor_1 * adf + factor_2 * ts.adfuller(pickles['orderbook_dict'][i].ret[
                              start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])[0]

            # for plots
            price = factor_1 * price + factor_2 * pickles['orderbook_dict'][i].price[
                    start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1]
            return_ = factor_1 * return_ + factor_2 * pickles['orderbook_dict'][i].ret[
                      start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1]

            price_all.append(pickles['orderbook_dict'][i].price[
                    start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])

            r = re.compile('MarketMaker*')
            agent_mm = list(filter(r.match, agent_types))[0]
            shares_mm = factor_1 * shares_mm + factor_2 * pickles['shares'][i][agent_mm][
                                                          start_end_wealth[i]['start']:start_end_wealth[i]['end']]
            cash_mm = factor_1 * cash_mm + factor_2 * pickles['cash'][i][agent_mm][
                                                      start_end_wealth[i]['start']:start_end_wealth[i]['end']]
            costs_mm = factor_1 * costs_mm + factor_2 * pickles['costs'][i][agent_mm][
                                                        start_end_wealth[i]['start']:start_end_wealth[i]['end']]
            wealth_mm = factor_1 * wealth_mm + factor_2 * pickles['wealth'][i][agent_mm][
                                                          start_end_wealth[i]['start']:start_end_wealth[i]['end']]
            price_mm = factor_1 * price_mm + factor_2 * pickles['orderbook_dict'][i].price[
                                                        start_end_orderbook[i]['start']:
                                                        start_end_orderbook[i]['end']][:, 1]

    # %% plots
    fig_price = pl.plot_price_avg_2(price, return_,
                                    title='price and return - ' + str(params['dict_sim']), show=False)
    # fig_price.write_image('C:/Users/WOstr/PycharmProjects/research_abm/abm_simulation/LOB/storage/'
    #                       + name + '_price' + '.jpeg', width=1980, height=1080)

    fig_price_all = pl.plot_price_all_2(price_all, title='price series - ' + str(params['dict_sim']), show=False)
    # fig_price_all.write_image('C:/Users/WOstr/PycharmProjects/research_abm/abm_simulation/LOB/storage/'
    #                           + name + '_price_all' + '.jpeg', width=1980, height=1080)

    fig_wealth_share = pl.plot_wealth_share(wealth_share_normalized_avg, drop_rows=0,
                                            title='wealth shares - ' + str(params['dict_sim']), show=False)

    fig_asset_mm = pl.plot_assets_mm(shares=shares_mm,
                                     cash=cash_mm,
                                     costs=costs_mm,
                                     wealth=wealth_mm,
                                     price=price_mm,
                                     drop_rows=0, title='market maker assets - ' + str(params['dict_sim']),
                                     show=False)
    # fig_asset_mm.write_image('C:/Users/WOstr/PycharmProjects/research_abm/abm_simulation/LOB/storage/'
    #                          + name + '_assets_mm' + '.jpeg', width=1980, height=1080)

    fig_autocorr_comb = pl.plot_autocorr_combined(autocorr_raw_avg, autocorr_abs_avg,
                                                  title='autocorrelation function - ' + str(params['dict_sim']),
                                                  show=False)
    # fig_autocorr_comb.write_image('C:/Users/WOstr/PycharmProjects/research_abm/abm_simulation/LOB/storage/'
    #                               + name + '_autocorr_comb' + '.jpeg', width=1980, height=1080)

    stats_dict = {
        'id_config': params['id_config'],
        'id_sim': params['id_sim'],
        'dict_sim': params['dict_sim'],
        'stats_market': {
            'mean': mean_,
            'geometric_mean': geometric_mean,
            'median': median_,
            'std': std_,
            'var': var_,
            'skew': skew_,
            'kurt': kurt_,
            'ac1_raw': ac1_raw,
            'ac2_raw': ac2_raw,
            'ac3_raw': ac3_raw,
            'ac1_abs': ac1_abs,
            'ac20_abs': ac20_abs,
            'ac50_abs': ac50_abs,
            'ac100_abs': ac100_abs,
            'adf': adf,
        },
        'figs': {
            'fig_price': fig_price,
            'fig_price_all': fig_price_all,
            'fig_wealth_share': fig_wealth_share,
            'fig_asset_mm': fig_asset_mm,
            'fig_autocorr_comb': fig_autocorr_comb,
        },

        'stats_shares': stats_shares,
        'stats_cash': stats_cash,
        'stats_wealth': stats_wealth,
    }

    print(stats_dict['stats_market'])
    print(stats_dict['stats_shares'])
    print(stats_dict['stats_wealth'])

    pickle.dump(stats_dict, open(local_path + 'results/' + name + '.p', 'wb'))

    print('')
    print(name + ' completed successfully.')
    print('Total execution time (run_stats): {}'.format(time.time() - start))

    return None


def run_stats_all(name=None, runs=1, local_path=None, **params):
    """
    Calculate stats for all simulations - loads all simulations in the memory.
    :param name: string
    :param runs: int
    :param local_path: string
    :param params: dict
    :return: None
    """
    start = time.time()

    # load pickles and re-create dict
    pickles = reduce(a.merge, [pickle.load(open(local_path + 'results_temp/' + name + '_' + str(i) + '.p', 'rb'))
                                    for i in range(0, runs)])

    # delete pickled files
    # [os.remove(PATH + 'results_temp/' + name + '_' + str(i) + '.p') for i in range(0, runs)]

    # unpack some attributes that are same across simulations
    agent_types = pickles['agent_portfolio_dict'][0]

    period_init = pickles['orderbook_dict'][0].period_init
    period_max = pickles['orderbook_dict'][0].period_max
    cutoff = pickles['params_sim_dict'][0]['sensitivity_base'].cutoff

    # define start:end in as dict here first to avoid repeating
    start_end_orderbook = {}
    for i in range(0, runs):
        dict_temp = {
            'start': pickles['orderbook_dict'][i].period_init
            if not pickles['params_sim_dict'][i]['sensitivity_base'].cutoff
            else pickles['params_sim_dict'][i]['sensitivity_base'].cutoff,
            'end': pickles['orderbook_dict'][i].period_max
        }
        start_end_orderbook.update({i: dict_temp})

    start_end_wealth = {}
    for i in range(0, runs):
        dict_temp = {
            'start': pickles['orderbook_dict'][i].period_init
            if not pickles['params_sim_dict'][i]['sensitivity_base'].cutoff else 0,
            'end': pickles['orderbook_dict'][i].period_max
            if not pickles['params_sim_dict'][i]['sensitivity_base'].cutoff
            else pickles['orderbook_dict'][i].period_max - pickles['params_sim_dict'][i]['sensitivity_base'].cutoff}
        start_end_wealth.update({i: dict_temp})

    # %% average out statistics
    # Note that behaviour is not specified if cutoff is zero
    wealth_share_normalized_avg, wealth_share_simple_avg = a.normalize_wealth_share(pickles['wealth_share_simple'],
                                                                                    agent_types,
                                                                                    period_init, period_max,
                                                                                    True, cutoff)

    autocorr_raw_avg = np.array(list(zip(*[a.calc_autocorr(
        pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
        for i in range(0, runs)]))).mean(axis=1)

    autocorr_abs_avg = np.array(list(zip(*[a.calc_autocorr(
        abs(pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1]))
        for i in range(0, runs)]))).mean(axis=1)

    # %% statistics
    # shares, cash and wealth
    stats_shares = pd.concat([pd.DataFrame(a.calc_stats_shares(pickles['shares'][i],
                                                               start=start_end_wealth[i]['start'],
                                                               end=start_end_wealth[i]['end']))
                              for i in range(0, runs)]).groupby(level=0).mean().to_dict()
    stats_cash = pd.concat([pd.DataFrame(a.calc_stats_shares(pickles['cash'][i],
                                                             start=start_end_wealth[i]['start'],
                                                             end=start_end_wealth[i]['end']))
                            for i in range(0, runs)]).groupby(level=0).mean().to_dict()
    stats_wealth = pd.concat([pd.DataFrame(a.calc_stats_wealth(pickles['wealth_evolution'][i],
                                                               start=start_end_wealth[i]['start'],
                                                               end=start_end_wealth[i]['end']))
                              for i in range(0, runs)]).groupby(level=0).mean().to_dict()
    # add wealth_share_normalized_avg
    stats_wealth = reduce(a.merge,
                          [stats_wealth, {key: {'wealth_share_normalized_avg': wealth_share_normalized_avg[key][-1]}
                                          for key in wealth_share_normalized_avg.keys()}])

    # %% plots
    fig_price = pl.plot_price_avg(pickles['orderbook_dict'], start_end_orderbook,
                                  title='price and return - ' + str(params['dict_sim']), show=False)
    # fig_price.write_image('C:/Users/WOstr/PycharmProjects/research_abm/abm_simulation/LOB/storage/'
    #                       + name + '_price' + '.jpeg', width=1980, height=1080)

    fig_price_all = pl.plot_price_all(pickles['orderbook_dict'], start_end_orderbook,
                                      title='price series - ' + str(params['dict_sim']), show=False)
    # fig_price_all.write_image('C:/Users/WOstr/PycharmProjects/research_abm/abm_simulation/LOB/storage/'
    #                           + name + '_price_all' + '.jpeg', width=1980, height=1080)

    fig_wealth_share = pl.plot_wealth_share(wealth_share_normalized_avg, drop_rows=0,
                                            title='wealth shares - ' + str(params['dict_sim']), show=False)

    r = re.compile('MarketMaker*')
    agent_mm = list(filter(r.match, agent_types))[0]
    fig_asset_mm = pl.plot_assets_mm(shares=np.array(list(zip(*[pickles['shares'][i][agent_mm][
                                                                start_end_wealth[i]['start']:
                                                                start_end_wealth[i]['end']]
                                                                for i in range(0, runs)]))).mean(axis=1),
                                     cash=np.array(list(zip(*[pickles['cash'][i][agent_mm][
                                                              start_end_wealth[i]['start']:
                                                              start_end_wealth[i]['end']]
                                                              for i in range(0, runs)]))).mean(axis=1),
                                     costs=np.array(list(zip(*[pickles['costs'][i][agent_mm][
                                                               start_end_wealth[i]['start']:
                                                               start_end_wealth[i]['end']]
                                                              for i in range(0, runs)]))).mean(axis=1),
                                     wealth=np.array(list(zip(*[pickles['wealth'][i][agent_mm][
                                                                start_end_wealth[i]['start']:
                                                                start_end_wealth[i]['end']]
                                                                for i in range(0, runs)]))).mean(axis=1),
                                     price=np.array([
                                         pickles['orderbook_dict'][i].price[start_end_orderbook[i]['start']:
                                                                 start_end_orderbook[i]['end']][:, 1]
                                         for i in range(0, runs)]).mean(axis=0),
                                     drop_rows=0, title='market maker assets - ' + str(params['dict_sim']),
                                     show=False)
    # fig_asset_mm.write_image('C:/Users/WOstr/PycharmProjects/research_abm/abm_simulation/LOB/storage/'
    #                          + name + '_assets_mm' + '.jpeg', width=1980, height=1080)

    fig_autocorr_comb = pl.plot_autocorr_combined(autocorr_raw_avg, autocorr_abs_avg,
                                                  title='autocorrelation function - ' + str(params['dict_sim']),
                                                  show=False)
    # fig_autocorr_comb.write_image('C:/Users/WOstr/PycharmProjects/research_abm/abm_simulation/LOB/storage/'
    #                               + name + '_autocorr_comb' + '.jpeg', width=1980, height=1080)

    mean_ = np.mean([np.mean(
        pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
        for i in range(0, runs)])
    geometric_mean = np.mean([(pickles['orderbook_dict'][i].price[start_end_orderbook[i]['end']][1] /
                                       pickles['orderbook_dict'][i].price[start_end_orderbook[i]['start']][1]) ** (
                                              1 / (start_end_orderbook[i]['end'] - start_end_orderbook[i]['start'])) - 1
                              for i in range(0, runs)])
    median_ = np.mean([np.median(
        pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
        for i in range(0, runs)])
    std_ = np.mean([np.std(
        pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
        for i in range(0, runs)])
    var_ = np.mean([np.var(
        pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
        for i in range(0, runs)])
    skew_ = np.mean([skew(
        pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
        for i in range(0, runs)])
    kurt_ = np.mean([kurtosis(
        pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])
        for i in range(0, runs)])

    ac1_raw = autocorr_raw_avg[1]
    ac2_raw = autocorr_raw_avg[2]
    ac3_raw = autocorr_raw_avg[3]

    ac1_abs = autocorr_abs_avg[1]
    ac20_abs = autocorr_abs_avg[20]
    ac50_abs = autocorr_abs_avg[50]
    ac100_abs = autocorr_abs_avg[100]

    adf = np.array([ts.adfuller(
        pickles['orderbook_dict'][i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1])[0]
        for i in range(0, runs)]).mean()

    stats_dict = {
        'id_config': params['id_config'],
        'id_sim': params['id_sim'],
        'dict_sim': params['dict_sim'],
        'stats_market': {
            'mean': mean_,
            'geometric_mean': geometric_mean,
            'median': median_,
            'std': std_,
            'var': var_,
            'skew': skew_,
            'kurt': kurt_,
            'ac1_raw': ac1_raw,
            'ac2_raw': ac2_raw,
            'ac3_raw': ac3_raw,
            'ac1_abs': ac1_abs,
            'ac20_abs': ac20_abs,
            'ac50_abs': ac50_abs,
            'ac100_abs': ac100_abs,
            'adf': adf,
        },
        'figs': {
            'fig_price': fig_price,
            'fig_price_all': fig_price_all,
            'fig_wealth_share': fig_wealth_share,
            'fig_asset_mm': fig_asset_mm,
            'fig_autocorr_comb': fig_autocorr_comb,
        },

        'stats_shares': stats_shares,
        'stats_cash': stats_cash,
        'stats_wealth': stats_wealth,
    }

    pickle.dump(stats_dict, open(local_path + 'results/' + name + '.p', 'wb'))

    print('')
    print(name + ' completed successfully.')
    print('Total execution time (run_stats): {}'.format(time.time() - start))

    return None


if __name__ == '__main__':
    from configs import sensitivity_00

    s_class = sensitivity_00.S()
    s_class.allocate()
    run(name='runtest', runs=1, local_path=PATH, sensitivity_base=s_class, id_config=1, id_sim=1, dict_sim='test')
    run_stats(name='runtest', runs=1, local_path=PATH, sensitivity_base=s_class, id_config=1, id_sim=1, dict_sim='test')
