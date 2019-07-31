from binance.client import Client
from ExchangeOperators.ExchangeOperator import ExchangeOperator
from Run.credentials import creds


class BinanceOperator(ExchangeOperator):
    """Used to directly operate with the Binance exchange api. Has all relevant
    api methods in their raw form. """
    def __init__(self):
        """Initialize BinanceOperator with api credentials."""
        client_id, client_secret = creds[0], creds[1]
        self.client = Client(client_id, client_secret)

    def get_market_depth(self, symbol):
        depth = self.client.get_order_book(symbol=symbol)
        return depth

    def get_recent_trades(self, symbol):
        trades = self.client.get_recent_trades(symbol=symbol)
        return trades

    def get_historical_trades(self, symbol):
        trades = self.client.get_historical_trades(symbol=symbol)
        return trades

    def get_aggregate_trades(self, symbol):
        trades = self.client.get_aggregate_trades(symbol=symbol)
        return trades

    def get_24hr_tickers(self):
        tickers = self.client.get_ticker()
        return tickers

    def get_symbol_info(self, symbol):
        info = self.client.get_symbol_info(symbol)
        return info

    def get_candlesticks(self, symbol, interval):
        candles = self.client.get_klines(symbol=symbol,
                                         interval=interval)
        return candles

    def get_historical_candlesticks(self, symbol, interval, start_time,
                                    end_time=None):
        if end_time is None:
            candles = self.client.get_historical_klines(symbol, interval,
                                                        start_time)
        else:
            candles = self.client.get_historical_klines(symbol, interval,
                                                        start_time, end_time)
        return candles

    def get_account_info(self):
        return self.client.get_account()

    def get_asset_balance(self, asset):
        return self.client.get_asset_balance(asset=asset)

    def get_trade_fees(self):
        return self.client.get_trade_fee()
