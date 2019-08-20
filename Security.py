
class Security:
    """
    Represents all information regarding an owned security asset

    Attributes:

        self.symbol: str
            - Name of the security on its trading exchange

        self.trading_interval: str
            - time period of candlestick data used in algo trading.
              Also the sleep time between iterations
    
        self.shares: int
            - number of shares

        self.api_wrapper: APIWrapper
            - APIWrapper used for trading this security

        self.strategy: Strategy
            - Strategy used to generate trading signals on live data for this Security
    """

    def __init__(self, symbol, trading_interval, shares. api_wrapper, strategy):
        """Initialize Security"""
        self.symbol = symbol
        self.trading_interval = trading_interval

        self.shares = shares

        self.api_wrapper
        self.strategy

    def __str__(self):
        """Return self as string"""
        return "{}: own {} shares. Trading at a {} interval. Total asset worth is {}.".format(self.symbol, self.shares, self.trading_interval, self.worth())

    def add_shares(self, amount_shares):
        self.shares += amount_shares

    def worth(self):
        """Return self.shares * current ticker price"""
        return self.shares * self.api_wrapper.tickers()[symbol]

    def remove_shares(self, amount_shares):
        self.shares -= amount_shares