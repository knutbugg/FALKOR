

class APIWrapper:
	"""Abstract class representing an API Wrapper for a financial security. All required methods must be completed by child class"""

	def historical_daily(start: str, end: str):
		"""Returns DataFrame with columns = ['time', 'open', 'close', 'high', 'low', 'volume'] for the security. Number of rows is equal to time interval divided by days"""

		raise NotImplementedError

	def historical_intraday(start: str, end: str, time_period: int):
		"""Returns DataFrame with columns = ['time', 'open', 'close', 'high', 'low', 'volume'] for the security. Number of rows is equal to time interval divided by time periods"""

		raise NotImplementedError

	def live_tickers():
		"""Returns a dictionary containing current ticker info"""

		raise NotImplementedError

	def buy_order(amount: int, price="market"):
		"""Created a buy order for amount. If price != "market", then a buy order will be created at price=price"""

		raise NotImplementedError

	def sell_order(amount:int, price="market"):
		"""Created a sell order for amount. If price != "market", then a sell order will be created at price=price"""

		raise NotImplementedError

	def check_balance():
		"""Returns portfolio balance"""

		raise NotImplementedError

	def trade_status(trade_id: str):
		"""Returns the status for trade with id trade_id"""

		raise NotImplementedError