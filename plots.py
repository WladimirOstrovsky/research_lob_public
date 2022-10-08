import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
# import plotly.io as pio
# pio.renderers.default = "browser"


def plot_wealth_share(wealth_share_normalized, drop_rows=0, title='wealth shares', show=False):
    """
    Creates wealth share plot by looping through key (agent type), value (wealth share of agent) pair.
    :param wealth_share_normalized: dict
    :param drop_rows: int
    :param title: string
    :param show: boolean
    :return: None
    """

    fig = make_subplots(rows=len(wealth_share_normalized.keys()), cols=1,
                        x_title='timestamp')

    keys_ = list(wealth_share_normalized.keys())
    keys_.sort()

    for idx, agent_type in enumerate(keys_, start=1):
        temp = pd.DataFrame(wealth_share_normalized[agent_type][drop_rows:])
        fig.add_trace(go.Scatter(x=temp.index,
                                 y=temp[0],
                                 mode='lines',
                                 name=agent_type),
                      row=idx, col=1)

    fig.update_layout(
        title=title,
        legend_title='agent types'
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_wealth_share_dist(wealth_share_normalized, distribution_param, distribution,
                           drop_rows=0, title='wealth shares', show=False):
    """
    Creates wealth share plot by looping through key (agent type), value (wealth share of agent) pair.
    :param wealth_share_normalized: dict
    :param distribution_param: string
    :param distribution: list
    :param drop_rows: int
    :param title: string
    :param show: boolean
    :return: None
    """

    fig = make_subplots(rows=len(wealth_share_normalized.keys()), cols=1,
                        x_title='timestamp',
                        subplot_titles=list(wealth_share_normalized.keys()))

    keys_ = list(wealth_share_normalized.keys())
    keys_.sort()

    for idx, agent_type in enumerate(keys_, start=1):
        for dist in distribution:
            temp = pd.DataFrame(wealth_share_normalized[agent_type][dist][drop_rows:])
            fig.add_trace(go.Scatter(x=temp.index,
                                     y=temp[0],
                                     mode='lines',
                                     name=dist,
                                     legendgroup=dist,
                                     showlegend=True if idx == 1 else False
                                     ),
                          row=idx, col=1)

    fig.update_layout(
        title=title,
        legend_title='agent types'
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_price_avg(orderbook_dict, start_end_orderbook=None, title='price and return', show=False):
    """
    Plots price and return.
    :param orderbook_dict: dict of objects
    :param title: string
    :param start_end_orderbook: dict
    :param show: boolean
    :return: None
    """
    fig = make_subplots(rows=2, cols=1,
                        x_title='timestamp')

    if not start_end_orderbook:
        price_avg = np.array([
            orderbook_dict[i].price[orderbook_dict[i].period_init:orderbook_dict[i].period_max][:, 1]
            for i in range(0, len(orderbook_dict.keys()))]).mean(axis=0)

        return_avg = np.array([
            orderbook_dict[i].ret[orderbook_dict[i].period_init:orderbook_dict[i].period_max][:, 1]
            for i in range(0, len(orderbook_dict.keys()))]).mean(axis=0)

    else:
        price_avg = np.array([
            orderbook_dict[i].price[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1]
            for i in range(0, len(orderbook_dict.keys()))]).mean(axis=0)

        return_avg = np.array([
            orderbook_dict[i].ret[start_end_orderbook[i]['start']:start_end_orderbook[i]['end']][:, 1]
            for i in range(0, len(orderbook_dict.keys()))]).mean(axis=0)

    price = pd.DataFrame(price_avg)
    return_ = pd.DataFrame(return_avg)

    fig.add_trace(go.Scatter(x=price.index,
                             y=price.iloc[:, 0],
                             mode='lines',
                             name='price'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=return_.index,
                             y=return_.iloc[:, 0],
                             mode='lines',
                             name='return'),
                  row=2, col=1)

    fig.update_layout(
        title=title,
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_price_avg_2(price, return_, title='price and return', show=False):
    """
    Plots price and return.
    :param price: np.array
    :param return_: np.array
    :param title: string
    :param show: boolean
    :return: None
    """
    fig = make_subplots(rows=2, cols=1,
                        x_title='timestamp')

    fig.add_trace(go.Scatter(x=np.arange(price.size),
                             y=price,
                             mode='lines',
                             name='price'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=np.arange(return_.size),
                             y=return_,
                             mode='lines',
                             name='return'),
                  row=2, col=1)

    fig.update_layout(
        title=title,
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_price_all(orderbook_dict, start_end_orderbook, max_id=1000, title='price series', show=False):
    """
    Plots price and return.
    :param orderbook_dict: dict of objects
    :param start_end_orderbook: dict
    :param max_id: int
    :param title: string
    :param show: boolean
    :return: None
    """
    fig = go.Figure()

    for id_, orderbook in orderbook_dict.items():
        if id_ < max_id:

            if not start_end_orderbook:
                price = pd.DataFrame(np.expand_dims(orderbook.price[orderbook.period_init:orderbook.period_max, 1], 1))

            else:
                price = pd.DataFrame(np.expand_dims(
                    orderbook.price[start_end_orderbook[id_]['start']:start_end_orderbook[id_]['end'], 1], 1))

            fig.add_trace(go.Scatter(x=price.index,
                                     y=price.iloc[:, 0],
                                     mode='lines',
                                     name=str(id_)))
        else:
            #
            break

    fig.update_layout(
        title=title,
        yaxis_title='price',
        xaxis_title='timestamp',
        legend_title='runs'
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_price_all_2(price_all, max_id=1000, title='price series', show=False):
    """
    Plots price and return.
    :param price_all: list of np.arrays
    :param max_id: int
    :param title: string
    :param show: boolean
    :return: None
    """
    fig = go.Figure()

    for id_, price in enumerate(price_all):
        if id_ < max_id:

            fig.add_trace(go.Scatter(x=np.arange(price.size),
                                     y=price,
                                     mode='lines',
                                     name=str(id_)))
        else:
            #
            break

    fig.update_layout(
        title=title,
        yaxis_title='price',
        xaxis_title='timestamp',
        legend_title='runs'
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_price(orderbook, drop_rows=0, show=False):
    """
    Plots price and return.
    :param orderbook: object
    :param drop_rows: int
    :param show: boolean
    :return: None
    """
    fig = make_subplots(rows=2, cols=1,
                        x_title='timestamp')

    price = pd.DataFrame(np.expand_dims(orderbook.price[drop_rows:, 1], 1))
    return_ = pd.DataFrame(np.expand_dims(orderbook.ret[drop_rows:, 1], 1))

    fig.add_trace(go.Scatter(x=price.index,
                             y=price.iloc[:, 0],
                             mode='lines',
                             name='price'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=return_.index,
                             y=return_.iloc[:, 0],
                             mode='lines',
                             name='return', ),
                  row=2, col=1)

    fig.update_layout(
        title='price and return',
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_assets_mm(shares, cash, costs, wealth, price, drop_rows=0, title='market maker assets', show=False):
    """
    Plots attributes of market maker including: wealth, shares, cash and market price.
    :param shares: np.array
    :param cash:  np.array
    :param wealth:  np.array
    :param price: np.array
    :param drop_rows: int
    :param title: string
    :param show: boolean
    :return: None
    """
    fig = make_subplots(rows=5, cols=1,
                        x_title='timestamp')

    wealth_mm_df = pd.DataFrame(wealth[drop_rows:, :])
    fig.add_trace(go.Scatter(x=wealth_mm_df.index,
                             y=wealth_mm_df.iloc[:, 0],
                             mode='lines',
                             name='wealth'),
                  row=1, col=1)

    shares_df = pd.DataFrame(shares[drop_rows:, :])
    fig.add_trace(go.Scatter(x=shares_df.index,
                             y=shares_df.iloc[:, 0],
                             mode='lines',
                             name='shares'),
                  row=2, col=1)

    cash_df = pd.DataFrame(cash[drop_rows:, :])
    fig.add_trace(go.Scatter(x=cash_df.index,
                             y=cash_df.iloc[:, 0],
                             mode='lines',
                             name='cash'),
                  row=3, col=1)

    costs_df = pd.DataFrame(costs[drop_rows:, :])
    fig.add_trace(go.Scatter(x=costs_df.index,
                             y=costs_df.iloc[:, 0],
                             mode='lines',
                             name='costs'),
                  row=4, col=1)

    price_df = pd.DataFrame(np.expand_dims(price[drop_rows:], 1))
    fig.add_trace(go.Scatter(x=price_df.index,
                             y=price_df.iloc[:, 0],
                             mode='lines',
                             name='price'),
                  row=5, col=1)

    fig.update_layout(
        title=title,
        legend_title='assets'
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_autocorr(autocorr, show=False):
    """
    Plots auto-correlation function.
    :param autocorr: np.array
    :param show: boolean
    :return: None
    """
    fig = go.Figure()
    autocorr_df = pd.DataFrame(np.expand_dims(autocorr, 1))

    fig.add_trace(go.Bar(x=autocorr_df.index,
                         y=autocorr_df.iloc[:, 0],
                         name='autocorrelation'))

    fig.update_layout(
        title='autocorrelation function',
        yaxis_title='autocorrelation',
        xaxis_title='lags',
        legend_title='assets'
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_autocorr_combined(autocorr_raw, autocorr_abs, title='autocorrelation function', show=False):
    """
    Plots auto-correlation function of both raw and absolute returns.
    :param autocorr_raw: np.array
    :param autocorr_abs: np.array
    :param title: string
    :param show: boolean
    :return: None
    """
    fig = make_subplots(rows=2, cols=1,
                        x_title='lag')

    autocorr_raw_df = pd.DataFrame(np.expand_dims(autocorr_raw, 1))
    autocorr_abs_df = pd.DataFrame(np.expand_dims(autocorr_abs, 1))

    fig.add_trace(go.Bar(x=autocorr_raw_df.index,
                         y=autocorr_raw_df.iloc[:, 0],
                         name='autocorrelation raw'),
                  row=1, col=1)
    fig.add_trace(go.Bar(x=autocorr_abs_df.index,
                         y=autocorr_abs_df.iloc[:, 0],
                         name='autocorrelation abs'),
                  row=2, col=1)

    fig.update_layout(
        title=title,
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_3d_chart_1(stats_market_select_enr, sensitivity_key, z_title, title, show=False):
    """
    Plots 3d scatter plot of 2-dim sensitivity. on x and y axis are the sensitivities and on the z axis the
    values of interest.
    :param stats_market_select_enr: pd.DataFrame
    :param sensitivity_key: list, sensitivity parameters
    :param z_title: string
    :param title: string
    :param show: boolean
    :return: fig
    """
    x, y, z = stats_market_select_enr[sensitivity_key[0]].values, \
              stats_market_select_enr[sensitivity_key[1]].values, \
              stats_market_select_enr['value'].values

    fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                       mode='markers')])

    margin = 1 / 100
    zaxis_range = [(1 - margin) * np.min(stats_market_select_enr['value'].values),
                   (1 + margin) * np.max(stats_market_select_enr['value'].values)]

    fig.update_scenes(
        zaxis={'title': z_title, 'tickformat': 'e', 'rangemode': 'tozero',
               'ticks': 'outside', 'range': zaxis_range}
        if abs(np.mean(stats_market_select_enr['value'].values)) < 0.001 else
        {'title': z_title, 'tickformat': '.4f', 'rangemode': 'tozero',
         'ticks': 'outside', 'range': zaxis_range},
        xaxis={'title': sensitivity_key[1], 'tickformat': '.4f', 'rangemode': 'tozero',
               'ticks': 'outside'},
        yaxis={'title': sensitivity_key[0], 'tickformat': '.4f', 'rangemode': 'tozero',
               'ticks': 'outside'}
    )
    fig.update_layout(title=title)

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_3d_chart_1_agents(stats_market_select_enr, sensitivity_key, z_title, title, show=False):
    """
    Plots 3d scatter plot of 2-dim sensitivity. on x and y axis are the sensitivities and on the z axis the
    values of interest.
    :param stats_market_select_enr: pd.DataFrame
    :param sensitivity_key: list, sensitivity parameters
    :param z_title: string
    :param title: string
    :param show: boolean
    :return: fig
    """

    fig = go.Figure()

    for agent_type in stats_market_select_enr['agent_type'].unique():
        stats_market_select_enr_agent = stats_market_select_enr[stats_market_select_enr['agent_type'] == agent_type]
        x, y, z = stats_market_select_enr_agent[sensitivity_key[0]].values, \
                  stats_market_select_enr_agent[sensitivity_key[1]].values, \
                  stats_market_select_enr_agent['value'].values
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', name=agent_type))

    margin = 1 / 100
    zaxis_range = [(1 - margin) * np.min(stats_market_select_enr['value'].values),
                   (1 + margin) * np.max(stats_market_select_enr['value'].values)]

    fig.update_scenes(
        zaxis={'title': z_title, 'tickformat': 'e', 'rangemode': 'tozero',
               'ticks': 'outside', 'range': zaxis_range}
        if abs(np.mean(stats_market_select_enr['value'].values)) < 0.001 else
        {'title': z_title, 'tickformat': '.4f', 'rangemode': 'tozero',
         'ticks': 'outside', 'range': zaxis_range},
        xaxis={'title': sensitivity_key[1], 'tickformat': '.2f', 'rangemode': 'tozero',
               'ticks': 'outside'},
        yaxis={'title': sensitivity_key[0], 'tickformat': '.2f', 'rangemode': 'tozero',
               'ticks': 'outside'},
    )

    fig.update_layout(title=title, legend_title='Agent Types')

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_3d_chart_1_agents_dist(stats_market_select_enr, sensitivity_key, z_title, title, show=False):
    """
    Plots 3d scatter plot of 2-dim sensitivity. on x and y axis are the sensitivities and on the z axis the
    values of interest. Using subplots for different agents.
    :param stats_market_select_enr: pd.DataFrame
    :param sensitivity_key: list, sensitivity parameters
    :param z_title: string
    :param title: string
    :param show: boolean
    :return: fig
    """

    agent_types = stats_market_select_enr['agent_type'].unique().tolist()

    fig = make_subplots(rows=1, cols=len(agent_types),
                        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                        subplot_titles=agent_types,
                        vertical_spacing=1.,
                        )

    for id, agent_type in enumerate(agent_types, start=1):
        for dist in stats_market_select_enr['distribution'].unique():
            stats_market_select_enr_agent = stats_market_select_enr[
                (stats_market_select_enr['agent_type'] == agent_type) &
                (stats_market_select_enr['distribution'] == dist)]
            x, y, z = stats_market_select_enr_agent[sensitivity_key[0]].values, \
                      stats_market_select_enr_agent[sensitivity_key[1]].values, \
                      stats_market_select_enr_agent['value'].values
            fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='markers', name=dist,
                                       legendgroup=dist,
                                       showlegend=True if id == 1 else False), row=1, col=id)

    margin = 1 / 100
    zaxis_range = [(1 - margin) * np.min(stats_market_select_enr['value'].values),
                   (1 + margin) * np.max(stats_market_select_enr['value'].values)]

    fig.update_scenes(
        zaxis={'title': z_title, 'tickformat': 'e', 'rangemode': 'tozero',
               'ticks': 'outside', 'range': zaxis_range}
        if abs(np.mean(stats_market_select_enr['value'].values)) < 0.001 else
        {'title': z_title, 'tickformat': '.4f', 'rangemode': 'tozero',
         'ticks': 'outside', 'range': zaxis_range},
        xaxis={'title': sensitivity_key[1], 'tickformat': '.2f', 'rangemode': 'tozero',
               'ticks': 'outside'},
        yaxis={'title': sensitivity_key[0], 'tickformat': '.2f', 'rangemode': 'tozero',
               'ticks': 'outside'},
    )

    fig.update_layout(title=title, legend_title='Agent Types',
                      height=500, width=1800)

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_2d_chart_1(stats_market_select_enr, sensitivity_key, z_title, title, show=False):
    """
    Plots 2d scatter plot of 2-dim sensitivity. on x axis are the sensitivities combined
    (because they are symmetric) and on the z axis the values of interest.
    :param stats_market_select_enr: pd.DataFrame
    :param sensitivity_key: list, sensitivity parameters
    :param z_title: string
    :param title: string
    :param show: boolean
    :return: fig
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=stats_market_select_enr['sym'].values,
                             y=stats_market_select_enr['value'].values,
                             mode='lines'))

    fig.update_layout(
        title=title,
        yaxis_title=z_title,
        xaxis_title='/'.join(sensitivity_key),
        yaxis_tickformat='.4e' if abs(np.mean(stats_market_select_enr['value'].values)) < 0.001 else None
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_2d_chart_2(stats_market_select_enr, sensitivity_key, z_title, title, show=False):
    """
    Plots 2d scatter plot of 1-dim sensitivity. on x axis are the sensitivities and on the
    z axis the values of interest.
    :param stats_market_select_enr: pd.DataFrame
    :param sensitivity_key: list, sensitivity parameters
    :param z_title: string
    :param title: string
    :param show: boolean
    :return: fig
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=stats_market_select_enr[sensitivity_key[0]].values,
                             y=stats_market_select_enr['value'].values,
                             mode='lines'))

    fig.update_layout(
        title=title,
        yaxis_title=z_title,
        xaxis_title='/'.join(sensitivity_key),
        yaxis_tickformat='.4e' if abs(np.mean(stats_market_select_enr['value'].values)) < 0.001 else None
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_2d_chart_1_agents(stats_enr, sensitivity_key, z_title, title, show=False):
    """
    Plots 2d scatter plot of 2-dim sensitivity. on x axis are the sensitivities combined
    (because they are symmetric) and on the z axis the values of interest.
    :param stats_enr: pd.DataFrame
    :param sensitivity_key: list, sensitivity parameters
    :param z_title: string
    :param title: string
    :param show: boolean
    :return: fig
    """
    fig = go.Figure()

    agent_types = stats_enr['agent_type'].unique().tolist()
    agent_types.sort()

    for agent_type in agent_types:
        fig.add_trace(go.Scatter(x=stats_enr[stats_enr['agent_type'] == agent_type]['sym'].values,
                                 y=stats_enr[stats_enr['agent_type'] == agent_type]['value'].values,
                                 mode='lines',
                                 name=agent_type))

    fig.update_layout(
        title=title,
        yaxis_title=z_title,
        xaxis_title='/'.join(sensitivity_key),
        yaxis_tickformat='.4e' if abs(np.mean(stats_enr['value'].values)) < 0.001 else None,
        legend_title='agent types'
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_2d_chart_2_agents(stats_enr, sensitivity_key, z_title, title, show=False):
    """
    Plots 2d scatter plot of 1-dim sensitivity. on x axis are the sensitivities and on the
    z axis the values of interest.
    :param stats_enr: pd.DataFrame
    :param sensitivity_key: list, sensitivity parameters
    :param z_title: string
    :param title: string
    :param show: boolean
    :return: fig
    """
    fig = go.Figure()

    agent_types = stats_enr['agent_type'].unique().tolist()
    agent_types.sort()

    for agent_type in agent_types:
        fig.add_trace(go.Scatter(x=stats_enr[stats_enr['agent_type'] == agent_type][sensitivity_key[0]].values,
                                 y=stats_enr[stats_enr['agent_type'] == agent_type]['value'].values,
                                 mode='lines',
                                 name=agent_type))

        # fig.add_trace(go.Scatter(x=stats_enr[sensitivity_key[0]].values,
        #                          y=stats_enr['value'].values,
        #                          mode='lines'))

    fig.update_layout(
        title=title,
        yaxis_title=z_title,
        xaxis_title='/'.join(sensitivity_key),
        yaxis_tickformat='.4e' if abs(np.mean(stats_enr['value'].values)) < 0.001 else None,
        legend_title='agent types'
    )

    if show:
        fig.show()
    else:
        pass

    return fig


def plot_2d_chart_2_agents_dist(stats_enr, sensitivity_key, z_title, title, show=False):
    """
    Plots 2d scatter plot of 1-dim sensitivity. on x axis are the sensitivities and on the
    z axis the values of interest. Using subplots for different agents.
    :param stats_enr: pd.DataFrame
    :param sensitivity_key: list, sensitivity parameters
    :param z_title: string
    :param title: string
    :param show: boolean
    :return: fig
    """
    agent_types = stats_enr['agent_type'].unique().tolist()
    agent_types.sort()

    fig = make_subplots(rows=len(agent_types), cols=1,
                        subplot_titles=agent_types,
                        shared_xaxes=True,
                        )

    for id, agent_type in enumerate(agent_types, start=1):
        for dist in stats_enr['distribution'].unique():
            fig.add_trace(go.Scatter(x=stats_enr[(stats_enr['agent_type'] == agent_type) &
                                                 (stats_enr['distribution'] == dist)][sensitivity_key[0]].values,
                                     y=stats_enr[stats_enr['agent_type'] == agent_type]['value'].values,
                                     mode='lines',
                                     name=dist,
                                     legendgroup=dist,
                                     showlegend=True if id == 1 else False),
                          row=id, col=1)

    fig.update_layout(
        title=title,
        # yaxis_title=z_title,
        # xaxis_title='/'.join(sensitivity_key),
        yaxis_tickformat='.4e' if abs(np.mean(stats_enr['value'].values)) < 0.001 else None,
        legend_title='distribution'
    )

    fig.update_xaxes(
        title='/'.join(sensitivity_key),
    )

    fig.update_yaxes(
        title=z_title,
    )

    if show:
        fig.show()
    else:
        pass

    return fig