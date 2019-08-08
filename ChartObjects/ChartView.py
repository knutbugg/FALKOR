"""Displaying a trading chart. """

import plotly.graph_objects as go
from datetime import datetime

from backtesting.HistoricalDataSaver import HistoricalDataSaver


def show_chart(path, symbol, duration):
    hds = HistoricalDataSaver(path)
    df = hds.load_data()

    # Create a deep copy of df so we can change it without mutation side-effect
    chart_df = df.copy(deep=True)
    # change each timestamp into a date
    for index, entry in chart_df.open_time.items():
        # we divide by 1000 to remove milliseconds to larger units
        chart_df.open_time.iloc[index] = datetime.fromtimestamp(entry / 1000)

    # create the plotly chart
    fig = go.Figure(data=[go.Candlestick(x=chart_df.open_time,
                                         open=chart_df.open,
                                         high=chart_df.high,
                                         low=chart_df.low,
                                         close=chart_df.close)])

    # Label the chart
    fig.update_layout(
        title='Trading View for {} - {}'.format(symbol, duration),
        yaxis_title='{} Price'.format(symbol))

    fig.show()

def show_chart_chunked(path, symbol, duration):
    hds = HistoricalDataSaver(path)
    df = hds.load_data()

    # Create a deep copy of df so we can change it without mutation side-effect
    chart_df = df.copy(deep=True)
    # change each timestamp into a date
    for index, entry in chart_df.open_time.items():
        # we divide by 1000 to remove milliseconds to larger units
        chart_df.open_time.iloc[index] = datetime.fromtimestamp(entry / 1000)
    x = len(chart_df)
    for chart_df in [chart_df[:int(x*0.3)], chart_df[int(x*0.33):int(x*0.66)],
                     chart_df[int(x*0.66):]]:
        fig = go.Figure(data=[go.Candlestick(x=chart_df.open_time,
                                             open=chart_df.open,
                                             high=chart_df.high,
                                             low=chart_df.low,
                                             close=chart_df.close)])

        # Label the chart
        fig.update_layout(
            title='Trading View for {} - {}'.format(symbol, duration),
            yaxis_title='{} Price'.format(symbol))

        fig.show()
show_chart_chunked('../strategy/training_data.csv', 'ETHBTC', '1h')

