
def Candle():
    """ Simple class representing a financial candlestick """
    def __init__(open_price, close_price, duration, volume, time_open, time_close):

        self.open_price = open_price
        self.close_price = close_price
        self.duration = duration
        self.volume = volume
        self.time_open = time_open
        self.time_close = time_close
