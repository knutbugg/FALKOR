class TradeExecutor:
    """Class used by Automated to perform all trades with the specified exchange
    """

    def buy_limit_order(self, symbol, quantity, price):
        raise NotImplementedError

    def sell_limit_order(self, symbol, quantity, price):
        raise NotImplementedError

    def buy_market_order(self, symbol, quantity):
        raise NotImplementedError

    def sell_market_order(self, symbol, quantity):
        raise NotImplementedError

    def order_status(self, symbol, order_id):
        raise NotImplementedError

    def cancel_order(self, symbol, order_id):
        raise NotImplementedError

    def get_open_orders(self):
        raise NotImplementedError

    def get_all_orders(self):
        raise NotImplementedError
