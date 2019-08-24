
class Portfolio:
    """
    Class representing a users trading portfolio. A Portfolio instance 
    contains all current and past trades, balance, and portfolio performance 
    metrics.
    
    Attributes:

        securities_trading: List
            - list of owned Security objects

        past_trades: List(TradeInfo)
            - a list of TradeInfo objects, each containing information about a completed trade
        

    Methods:
        
        
    """

    def __init__(self):
        """Initialize Portfolio instance"""
        self.securities_trading = []
        self.past_trades = []

    def add_security(self, sec):
        self.securities_trading.append(sec)

    def display(self):
        print("Portfolio: \n {}".format([str(sec) for sec in self.securities_trading]))

    def add_past_trade(self, ):
        pass