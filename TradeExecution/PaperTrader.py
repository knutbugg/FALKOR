from tradeexecution.TradeExecutor import TradeExecutor
from exchangeoperators.BinanceOperator import BinanceOperator


class PaperTrader(TradeExecutor):
    """TradeExecutor running on a paper trading account"""

    def __init__(self, wallet):
        """Initialize a PaperTrader class"""
        self.wallet = wallet
        self.op = BinanceOperator()

        self.orders = {}

    def buy_limit_order(self, symbol, quantity, price):
        """Send a buy order for quantity amount of symbol"""
        pass

    def sell_limit_order(self, symbol, quantity, price):
        """Send a sell order for quantity amount of symbol"""
        raise NotImplementedError

    def buy_market_order(self, symbol, quantity) -> str:
        """Buy quantity of symbol for the best market prices offered"""

        market_depth = self.op.get_market_depth(symbol)
        qty_to_buy = quantity
        price_spent = 0

        if price_spent > self.wallet:
            raise Exception(
                '{} in the wallet, order costs at least {}'.format(
                    self.wallet, price_spent
                ))

        while qty_to_buy > 0:
            if price_spent > self.wallet:
                raise Exception(
                    '{} in the wallet, order costs at least {}'.format(
                        self.wallet, price_spent
                    ))
            for sell_order in market_depth.get('asks'):
                price, qty = float(sell_order[0]), float(sell_order[1])

                if qty_to_buy > qty:
                    qty_to_buy -= qty
                    price_spent += price
                elif qty_to_buy <= qty:
                    qty_to_buy = 0
                    price_spent += price

        self.buy_at_price(symbol, quantity, price_spent)

    def sell_market_order(self, symbol, quantity):
        raise NotImplementedError

    def buy_at_price(self, symbol, qty, total_price):
        """Simple buy at price method"""
        self._add_order(symbol, qty, total_price)

        print(
            "Successfully purchased {} shares of {} for a total of {}.".format(
                qty, symbol, total_price))

    def sell_at_price(self, symbol, qty, total_price):
        """Simple buy at price method"""
        self._sell_order(symbol, qty, total_price)

        print(
            "Successfully sold {} shares of {} for a total of {}.".format(
                qty, symbol, total_price))

    def order_status(self, symbol, order_id):
        raise NotImplementedError

    def cancel_order(self, symbol, order_id):
        raise NotImplementedError

    def get_open_orders(self):
        raise NotImplementedError

    def get_all_orders(self):
        raise NotImplementedError

    def get_balance(self) -> float:
        return self.wallet

    def _add_order(self, symbol, qty, price_bought):
        if symbol in self.orders:
            self.orders[symbol].append([qty, price_bought])
        else:
            self.orders[symbol] = [[qty, price_bought]]

        self.wallet -= price_bought

    def _sell_order(self, symbol, sell_qty):
        sell_remaining = sell_qty
        total_sold_value = 0

        for order in self.orders.get(symbol):
            if sell_remaining > 0:
                if order[0] > sell_qty:
                    order[0] -= sell_qty

                    total_sold_value = order[1]
                    sell_remaining = 0

                elif order[0] <= sell_qty:
                    sell_remaining -= order[0]
                    order[:] = []
                    # TODO check what this does to the storage of the order. Is it
                    # empty list, and if so, we need to delete the list in an outer loop
                    total_sold_value += order[1]

            self.wallet += total_sold_value
            return total_sold_value

    def trade_from_signals(self, signals):
        """Perform specified trades inside the TradingSignals object. """
        for buy_signal in signals.get_buy_signals():
            print(buy_signal)
        for sell_signal in signals.get_sell_signals():
            print(sell_signal)
