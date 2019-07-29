from TradeExecution import PaperTrader, BinanceMoneyTrader

class Account:
    def __init__(self, trader: str):
        if trader == 'paper':
            self.trader = PaperTrader()

        elif trader == 'binance':
            self.trader = BinanceMoneyTrader()


