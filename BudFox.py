from PaperTrader import PaperTrader

class BudFox:
    """
    Works with APIWrappers and Gekko. Receives trading signals from Gekko, and sends them to specified API Wrapper

    Attributes:

        paper_trade: bool
            - if True, BudFox will not send trading signals to APIWrapper, but will paper trade with imaginary assets 
    """
    def __init__(self):
        """Initialize BudFox instance"""
        self.paper_trade = False # begin with real trading unless specified otherwise explicitly
        self.paper_trader = PaperTrader(liquid=1000)

    def send_trading_signal(self, symbol: str, signal: str, amount: int, api_wrapper, price="market"):
        """Sends buy and sell signals to specified api_wrapper. Returns string of trade receipt"""
        # replace api_wrapper with PaperTrader instance if specified
        if self.paper_trade:
            api_wrapper = self.paper_trader

        if signal == "buy":
            message = api_wrapper.buy_order(symbol, amount, price)
        
        elif signal == "sell":
            message = api_wrapper.sell_order(symbol, amount, price)

        return "BudFox: {}".format(message)
