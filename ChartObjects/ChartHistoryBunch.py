import pandas as pd

from backtesting.HistoricalDataSaver import HistoricalDataSaver
from exchangeoperators.BinanceLiveDataFeed import BinanceLiveDataFeed


class ChartHistoryBunch:
    """ChartHistoryBunch holds all necessary historical candlestick information
    about financial markets. It also keeps the data updated real-time. """

    def __init__(self, symbol, data_loc, interval):
        """Initialize ChartHistoryBunch"""
        # Location of .csv file containing candlestick data
        self.save_loc = data_loc
        self.symbol = symbol

        # Convert .csv file data into pandas DataFrame
        self.past_data = pd.read_csv(self.save_loc)
        # Candlestick interval time
        self.interval = interval
        # Specify the LiveDataFeed used to pull current financial candlestjcks
        self.data_feed = BinanceLiveDataFeed()

        # Specify what DataSaver to use to save the candlesticks
        self.data_saver = HistoricalDataSaver(data_loc)

    def update(self, symbol, interval):
        """Updates self.past_data, and the .csv file to contain present
         candlestick information. """
        new_data = self.data_feed.pull(symbol, interval)
        self.data_saver.add_recent_candles(new_data, snip_amount=50)
