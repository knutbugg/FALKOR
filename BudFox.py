from PaperTrader import PaperTrader

class BudFox:
    """
    Works with APIWrappers and Gekko. Receives trading signals from Gekko, 
    and sends them to specified API Wrapper

    Attributes:

        paper_trade: bool
            - if True, BudFox will not send trading signals to APIWrapper, 
            but will paper trade with imaginary assets 
    
        paper_trader: PaperTrader
            - instance of PaperTrader class used instead of APIWrapper
            if paper_trade == True

    Methods:

        send_trading_signal(self, symbol: str, signal: str, amount: int, 
                            api_wrapper, price: int or str)
    """
    def __init__(self):
        """Initialize BudFox instance"""

        # begin with real trading unless specified otherwise explicitly
        self.paper_trade = False 
        self.paper_trader = PaperTrader(liquid=1000)

    def send_trading_signal(self, symbol: str, signal: str, amount: int, 
                            api_wrapper, price="market"):
        """
        Sends buy and sell signals to specified api_wrapper. Returns 
        string of trade receipt
        """
        
        # If market order, sell for market price
        if price == "market":
            price = api_wrapper.tickers()[symbol]
        
        # replace api_wrapper with PaperTrader instance if specified
        if self.paper_trade:
            api_wrapper = self.paper_trader



        if signal == "buy":
            trade_info = api_wrapper.buy_order(symbol, amount, price)

        elif signal == "sell":
            trade_info = api_wrapper.sell_order(symbol, amount, price)

        return trade_info
