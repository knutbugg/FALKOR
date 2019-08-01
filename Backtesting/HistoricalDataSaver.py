import pandas as pd
from pathlib import Path

class HistoricalDataSaver:
    """Saves Historical Candlestick data to a .csv file. Can also load up the
    data. """

    def __init__(self, storage_location):
        """Initialize the HistoricalDataSaver with a storage_location, and
        initialize a blank .csv file. """
        self.loc = Path(storage_location)
        self.initialize_storage_file()

    def initialize_storage_file(self):
        """Create an empty .csv file in the directory specified, or loads
         up the already existing file"""
        if self.loc.is_file():
            pass
        else:
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

    def add_recent_candles(self, candles, snip_amount):
        """Take the raw candlestick data, snip it down, and add the last entries
        into the data. Save the .csv file. To be used to keep the data updated.
        """
        # TODO Check behavior if bugs exist its in here!
        df = self.load_data()

        # Get the last candles
        last_candles = candles[len(candles)-snip_amount:]
        time_of_first_candle = last_candles[0][0]

        # remove rows in .csv file that come after the last five candles so we
        # can smoothly insert them

        # convert raw candlestick data to DataFrame
        new_df = self.candles_to_df(last_candles)

        # Delete rows that have the timestamp larger than our first new candle
        df = df.drop(df.loc[df['open_time'] >= time_of_first_candle], axis=1)
        # append and save the data
        df = df.append(new_df, sort=True)
        df.to_csv(self.loc, index=False)

    @staticmethod
    def candles_to_df(candles):
        """Create a DataFrame in the same format as the one which holds the
        historical candlestick data, and put the raw candlestick data into it.
        """
        df = pd.DataFrame(columns=['open_time', 'open', 'high', 'low', 'close',
                          'volume', 'close_time', 'quote_asset_volume',
                          'num_trades', 'tkbbav', 'tkqav', 'ign.'])

        for c_data in candles:
            new_df = pd.DataFrame(pd.DataFrame(data=c_data).T)
            new_df.columns = ['open_time', 'open', 'high', 'low', 'close',
                              'volume', 'close_time', 'quote_asset_volume',
                              'num_trades', 'tkbbav', 'tkqav', 'ign.']

            df = df.append(new_df)
        return df
