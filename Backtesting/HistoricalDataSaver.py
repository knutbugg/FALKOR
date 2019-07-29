import pandas as pd
from pathlib import Path

"""
Candlestick data response from Binance is in the following format:

[
  [
    1499040000000,      // Open time
    "0.01634790",       // Open
    "0.80000000",       // High
    "0.01575800",       // Low
    "0.01577100",       // Close
    "148976.11427815",  // Volume
    1499644799999,      // Close time
    "2434.19055334",    // Quote asset volume
    308,                // Number of trades
    "1756.87402397",    // Taker buy base asset volume
    "28.46694368",      // Taker buy quote asset volume
    "17928899.62484339" // Ignore.
  ]
]

"""


class HistoricalDataSaver:
    """Saves Historical Candlestick data to a .csv file. Can also load up the
    data. """

    def __init__(self, storage_location):
        """Initialize the HistoricalDataSaver with a storage_location, and
        initialize a blank .csv file. """
        self.loc = Path(storage_location)
        self.initialize_storage_file()

    def initialize_storage_file(self):
        """Create an empty .csv file in the directory specified. """
        df = pd.DataFrame(columns=['open_time', 'open', 'high', 'low', 'close',
                                   'volume', 'close_time','quote_asset_volume',
                                   'num_trades', 'tkbbav', 'tkqav', 'ign.'])
        df.to_csv(self.loc, index=False)

    def save_data(self, data):
        """Save Binance formatted candlestick data into the .csv file, at the
        end of the file. """
        df = self.load_data()
        for c_data in data:
            new_df = pd.DataFrame(pd.DataFrame(data=c_data).T)
            new_df.columns = ['open_time', 'open', 'high', 'low', 'close',
                                   'volume', 'close_time','quote_asset_volume',
                                   'num_trades', 'tkbbav', 'tkqav', 'ign.']

            df = df.append(new_df)
        df.to_csv(self.loc, index=False)


    def load_data(self):
        """Return the saved candlestick data as a DataFrame"""
        df = pd.read_csv(self.loc)
        return df

