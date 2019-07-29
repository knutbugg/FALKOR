from unittest.mock import Mock
from binance.client import Client


class BinanceOperator:

    def __init__(self, client_id, client_secret):
        """Initialilze BinanceOperator with api credentials."""

        self.client = Client(client_id, client_secret)

    def buy_order(self):
        # CURRENTLY UNDER CONSTRUCTION
        # place a test market buy order, to place an actual order use the create_order function
        order = self.client.create_test_order(
            symbol='BNBBTC',
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_MARKET,
            quantity=100)

    def get_candlesticks(self, symbol, interval=Client.KLINE_INTERVAL_30MINUTE):
        candles = self.client.get_klines(symbol=symbol,
                                         interval=interval)

    def get_historical_candlesticks(self, symbol, interval,
                                                  start_time,
                                                  end_time=None
                                                  ):

        if end_time is None:
            candles = self.client.get_historical_klines(symbol,
                                                        interval,
                                                        start_time)
                                                        # end_time)
        else:
            candles = self.client.get_historical_klines(symbol, interval,
                                                        start_time,
                                                        end_time)
        return candles
