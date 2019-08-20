
class Portfolio:
    """
    Class representing a users trading portfolio. A Portfolio instance contains all current
    and past trades, balance, and portfolio performance metrics.
    
    Attributes:

        securities_trading: dict == {'ETHBTC': ['1d', Ex1Strategy, BinanceWrapper], 'TSLA': ['5m', Ex2Strategy, IBWrapper]}
            - dictionary containing security str as key, and [interval, Strategy, APIWrapper] as the value

        past_trades: List(Trade)
            - a list of Trade objects, each containing information about a completed trade
        
        current_trades: List(Trade)
            - a list of Trade objects, each containing an in-progress trade
    """

    def __init__(self, paper_trade=False):
        """Initialize Portfolio instance"""
        
        self.paper_trade = paper_trade

        self.securities_trading = {}
        self.past_trades = []
        self.current_trades = []

    def add_security(self, name: str, interval: str, strategy, api_wrapper):
        self.securities_trading[name] = [interval, strategy, api_wrapper]

    def add_past_trade(self, ):
        pass

    def add_current_trade(self, ):
        pass
