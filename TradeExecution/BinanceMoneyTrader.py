from TradeExecution.MoneyTrader import MoneyTrader
from ExchangeOperators.BinanceOperator import BinanceOperator

class BinanceMoneyTrader(MoneyTrader):
    def __init__(self, client_id, client_secret):
        self.op = BinanceOperator(client_id, client_secret)

    def buy_limit_order(self, symbol, quantity, price):
        from binance.enums import *
        order = client.create_order(
            symbol='BNBBTC',
            side=SIDE_BUY,
            type=ORDER_TYPE_LIMIT,
            timeInForce=TIME_IN_FORCE_GTC,
            quantity=100,
            price='0.00001')

    def sell_limit_order(self, symbol, quantity, price):
        raise NotImplementedError

    def buy_market_order(self, symbol, quantity, price):
        raise NotImplementedError

    def sell_market_order(self, symbol, quantity, price):
        raise NotImplementedError

    def order_status(self, symbol, order_id):
        raise NotImplementedError

    def cancel_order(self, symbol, order_id):
        raise NotImplementedError

    def get_open_orders(self):
        raise NotImplementedError

    def get_all_orders(self):
        raise NotImplementedError
