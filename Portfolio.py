
class Portfolio:
    """
    Class representing a users trading portfolio. A Portfolio instance contains all current
    and past trades, balance, and portfolio performance metrics.
    
    Attributes:

        securities_trading: List
            - list of owned Security objects

        past_trades: List(Trade)
            - a list of Trade objects, each containing information about a completed trade
        
        current_trades: List(Trade)
            - a list of Trade objects, each containing an in-progress trade
    """

    def __init__(self, paper_trade=False):
        """Initialize Portfolio instance"""
        
        self.paper_trade = paper_trade

        self.securities_trading = []
        self.past_trades = []
        self.current_trades = []

    def add_security(self, sec):
        self.securities_trading.append(sec)

    def display(self):
        print("Portfolio: \n {}".format([str(sec) for sec in self.securities_trading]))

    def add_past_trade(self, ):
        pass

    def add_current_trade(self, ):
        pass
