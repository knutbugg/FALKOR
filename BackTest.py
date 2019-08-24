from Gekko.Gekko import Gekko
from api_wrappers.APIWrapper import APIWrappers
from api_wrappers.BinanceWrapper import BinanceWrapper
from strategies import CNN_Strategy

from helpers import data_processing

from pandas import DataFrame

class BackTest:
	"""
	BackTest is used to run FALKOR on historical candlestick data
	and measure the profitability of chosen strategy
	
	Attributes:

	Methods:

	"""

	def __init__(self, api_wrapper, strategy):
		"""Initialize BackTest instance"""
		self.api_wrapper = api_wrapper
		self.strategy = strategy

		# Initialize with an empty portfolio
		self.portfolio = Portfolio()
		self.gekko = Gekko(self.portfolio)

	def get_historical_candles(self, api_wrapper: APIWrapper, symbol: str, interval:str, start_time: str, end_time: str) -> DataFrame:
		"""Returns a DataFrame of containing ochlv candles"""
		
		return api_wrapper.historical_candles(symbol, interval, start_time, end_time)

	def preprocess_data(self, candles_df: DataFrame) -> DataFrame
		"""
		Returns a DataFrame oc ochlv candles with specified processing
		steps performed ex. added technical indicators
		"""
		return data_processing.add_ti()

	def split_data(self, df: DataFrame, period: int, step_size: int) -> List(DataFrame), List(float):
		"""
		Returns a tuple of Lists. The first is a List of DataFrames, each 
		being an input. The second is a List of ints, each being the 
		price for the Security at period = curr_period + 5
		"""
		df_splits = data_processing.split_dataset(df)
		price_splits = data_processing.price_labels(df_splits)

		# ensure len(df_splits) == len(price_splits)
		df_splits = df_splits[:len(price_splits)]

		return df_splits, price_splits

	def run(self):
		# Initialize what we are backtesting 
		api_wrapper = BinanceWrapper(
			'nBjgb83VMNvqq45b3JdWUIsJDalWlXxHI2bvDz9oLdW7KgOLPvJCp30CHnthjfNJ',
			'5bBN7s7h37kUvmGIpF9FTAtspBY93WirwhTh39PV7AlKSlUE2S4EEe9b3OZVYIqd'
			)

		symbol, interval = 'ETHBTC', '1m'
		start_time, end_time = 'January 1 2018', 'January 1 2019'


		# Get historical OCHLV candlestick DataFrame
		candles_df = self.get_historical_candles(api_wrapper, symbol, interval,
											start_time, end_time)

		# Perform Preprocessing steps on the data
		processed_df = self.preprocess_data(candles_df)

		# Get a list of input df's and their corresponding future price
		df_splits, price_splits = self.split_data(df=processed_df, period=30, step_size=5)

		# Run a backtest with profitabililty tracking on df_splits and price_splits
		results = self.gekko.backtest(df_splits, price_splits)

		# Show backtesting results. Profitabilty, trades made,
		print(results)

if __name__ == '__main__':
    bt = BackTest()
    bt.run()
