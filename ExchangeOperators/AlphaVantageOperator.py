from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.timeseries import TimeSeries
from backtesting.HistoricalDataSaver import HistoricalDataSaver

class AlphaVantageOperator:

    def __init__(self, key):
        self.api_key = key

    def hist_ohlcv(self, symbol):
        """Returns DataFrame in the format of
        [ Time | 1. open | 2. high | 3. low | 4.close | 5. volume ]"""
        ts = TimeSeries(key=self.api_key, output_format='pandas')
        x = ts.get_daily(symbol, outputsize='full')
        return x
