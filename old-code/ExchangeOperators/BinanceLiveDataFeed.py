from binance.client import Client

from exchangeoperators.BinanceOperator import BinanceOperator


class BinanceLiveDataFeed:
    """LiveDataFeed for Binance Exchange. Provides .pull() method that pulls
    new candlestick data. """
    def __init__(self):
        self.binance_op = BinanceOperator()

    def pull(self, symbol, interval):
        """ Pulls candlesticks for the last day of trading. Format however you
        will. """
        return self.binance_op.get_historical_candlesticks(symbol, interval,
                                                           start_time="1 day")

