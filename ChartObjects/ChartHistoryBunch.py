class ChartHistoryBunch:
    """Data Class used to hold all necessary information to be used alongside
    incoming data about the financial markets. """

    def __init__(self, symbol, data):
        self.symbol = symbol
        self.past_data = data

    def update(self):
        """Updates self.data to include present ticker info. """

