"""Displaying a trading chart. """

import plotly.graph_objects as go
from Backtesting.HistoricalDataSaver import HistoricalDataSaver
from datetime import datetime


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
