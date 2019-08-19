

class APIWrapper:
	"""Abstract class representing an API Wrapper for a financial security. All required methods must be completed by child class"""

	def historical_candles(self, symbol: str, interval:str, start_time: str, end_time: str):
		"""Returns DataFrame with columns = ['time', 'open', 'close', 'high', 'low', 'volume'] for the security. Number of rows is equal to time interval divided by days"""

		raise NotImplementedError

	def last_candles(self, symbol, interval):
		"""Returns DataFrame containing the most recent candlesticks information"""

		raise NotImplementedError

	def live_tickers(self):
		"""Returns a dictionary containing current ticker info"""

		raise NotImplementedError

	def buy_order(self, amount: int, price="market"):
		"""Created a buy order for amount. If price != "market", then a buy order will be created at price=price"""

		raise NotImplementedError

	def sell_order(self, amount:int, price="market"):
		"""Created a sell order for amount. If price != "market", then a sell order will be created at price=price"""

		raise NotImplementedError

	def check_balance(self,):
		"""Returns portfolio balance"""

		raise NotImplementedError

	def trade_status(self, trade_id: str):
		"""Returns the status for trade with id trade_id"""

		raise NotImplementedError

	def cancel_order(self, trade_id: str):
		"""Cancel order with trade_id"""

		raise NotImplementedError
