
class Security:
    """
    Represents all information regarding an owned security asset

    Attributes:

        symbol: str
            - Name of the security on its trading exchange

        trading_interval: str
            - time period of candlestick data used in algo trading.
              Also the sleep time between iterations in .run() in TheTerminal
    
        shares: int
            - number of shares

        api_wrapper: APIWrapper
            - APIWrapper used for trading this security

        strategy: Strategy
            - Strategy used to generate trading signals

        status: str
            - status for this security. 
                * if status == "ready", then this Security can be traded
                * if status == "waiting", then this Security is waiting
                to be bought/sold at a later period, DO NOT trade

        _waiting_periods_left: int
            - if status == "waiting", then _waiting_periods_left is how many
            iterations status should remain waiting

    Methods:

        add_shares(self, amount_shares):
            - add amount_shares to self.shares

        remove_shares(self, amount_shares):
            - remove amount_shares from self.shares

        worth(self):
            returns self.shares * current ticker price
    """

    def __init__(self, symbol, interval, shares, api_wrapper, strategy):
        """Initialize Security"""
        self.symbol = symbol
        self.interval = interval
        self.shares = shares
        self.api_wrapper = api_wrapper
        self.strategy = strategy

        self.status = "ready"
        self._waiting_periods_left = 0

    def __str__(self):
        """Return self as string"""
        return "{}: own {} shares. Trading at a {} interval. Total asset worth is {}.".format(self.symbol, self.shares, self.interval, self.worth())

    def add_shares(self, amount_shares):
        self.shares += amount_shares

    def remove_shares(self, amount_shares):
        self.shares -= amount_shares

    def worth(self):
        """Return self.shares * current ticker price"""
        return self.shares * self.api_wrapper.tickers()[self.symbol]

    def set_status(self, status: str, periods: int):
        """set self.status to status"""
        self.status = status
        self._waiting_periods_left = periods

    def update_waiting_status(self, amount: int):
        """
        Update self._waiting_periods_left by amount. If 
        self._waiting_periods_left <= 0, self.status is set to 'ready'
        """

        if self._waiting_periods_left - amount <= 0:
            self.set_status(status="ready", periods=0)
        else:
            self._waiting_periods_left += amount
