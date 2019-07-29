"""Abtract class"""
class ExchangeOperator():
    def get_market_depth(self, symbol):
        raise NotImplementedError

    def get_recent_trades(self, symbol):
        raise NotImplementedError

    def get_historical_trades(self, symbol):
        raise NotImplementedError

    def get_aggregate_trades(self, symbol):
        raise NotImplementedError

    def get_24hr_ticker(self, symbol):
        raise NotImplementedError

    def get_symbol_info(self, symbol):
        raise NotImplementedError

    def get_candlesticks(self, symbol, interval):
        raise NotImplementedError

    def get_historical_candlesticks(self, symbol, interval, start_time,
                                    end_time=None):
        raise NotImplementedError

    def get_account_info(self):
        raise NotImplementedError

    def get_asset_balance(self, asset):
        raise NotImplementedError

    def get_trade_fees(self):
        raise NotImplementedError
