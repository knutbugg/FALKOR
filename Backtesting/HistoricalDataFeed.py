from ExchangeOperators.BinanceOperator import BinanceOperator
from Backtesting.HistoricalDataSaver import HistoricalDataSaver
from binance.client import Client

class HistoricalDataFeed:
    """Used to get historial candlestick data and save it into a .csv file"""
    def __init__(self, exchange: str):
        operator = None
        if exchange == "binance":
            operator = BinanceOperator()

        # Continue with if statements initializing for all supported exchanges

        self.operator = operator

    def historical_data(self, symbol, interval, start_time, end_time=None):
        """Returns candlestick data for a particular interval. If end_time=None,
        returns all candlestick data from start_time to the present. """
        if interval == '1m':
            interval = Client.KLINE_INTERVAL_1MINUTE
        elif interval == '3m':
            interval = Client.KLINE_INTERVAL_3MINUTE
        elif interval == '5m':
            interval = Client.KLINE_INTERVAL_5MINUTE
        elif interval == '15m':
            interval = Client.KLINE_INTERVAL_15MINUTE
        elif interval == '30':
            interval = Client.KLINE_INTERVAL_30MINUTE
        elif interval == '1h':
            interval = Client.KLINE_INTERVAL_1HOUR
        elif interval == '2h':
            interval = Client.KLINE_INTERVAL_2HOUR
        elif interval == '6h':
            interval = Client.KLINE_INTERVAL_6HOUR
        elif interval == '12h':
            interval = Client.KLINE_INTERVAL_12HOUR
        elif interval == '1d':
            interval = Client.KLINE_INTERVAL_1DAY
        elif interval == '3d':
            interval = Client.KLINE_INTERVAL_3DAY
        elif interval == '1w':
            interval = Client.KLINE_INTERVAL_1WEEK
        elif interval == '1M':
            interval = Client.KLINE_INTERVAL_1MONTH
        else:
            raise NameError('The interval specified is incorrect. ')

        return self.operator.get_historical_candlesticks(symbol, interval,
                                                         start_time, end_time)

    def save_historical_data(self, data, loc):
        hds = HistoricalDataSaver(loc)
        hds.save_data(data)

