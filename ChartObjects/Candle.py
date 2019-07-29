class Candlestick:
    """ Simple class representing a financial candlestick """

    def __init__(self, open_price, close_price, volume, time_open,
                 time_close, high, low, num_trades):
        self.open_price = open_price
        self.close_price = close_price
        self.volume = volume
        self.time_open = time_open
        self.time_close = time_close
        self.high = high
        self.low = low
        self.num_trades = num_trades
        self.duration = time_close - time_open


class BinanceCandlestick(Candlestick):
    """Binance h_data candlestick"""

    def __init__(self, open_price, close_price, duration, volume, time_open,
                 time_close, high, low, num_trades, quote_asset_vol):
        Candlestick(self, open_price, close_price, duration, volume,
                       time_open,
                       time_close, high, low, num_trades)
        self.quote_asset_volume = quote_asset_vol
