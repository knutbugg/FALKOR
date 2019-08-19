from TradeExecution import PaperTrader, BinanceMoneyTrader


class Account:
    """Account contains all relevant information to the maintenance of your
    trading portfolio, and communication with the broker. """
    def __init__(self, trader: str):
        if trader == 'paper':
            self.trader = PaperTrader()

        elif trader == 'binance':
            self.trader = BinanceMoneyTrader()


