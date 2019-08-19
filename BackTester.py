
class BackTester:
	"""BackTest on historical financial price data in order to calculate profits, locations of profits, and locations of losses"""

	def import_data(file_path: str):
		"""Reads in the data from file_path"""
		pass

	def prepare_data():
		"""Prepare the imported data before sending it to Gekko. ex. chunk data into 30 period dataframes"""
		pass

	def backtest(strategy, trading_fee=0.0):
		"""Backtest with strategy with trading_fee. Returns profitability metrics and chart image of best and worst predictions. """
		pass