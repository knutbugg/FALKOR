
class Portfolio:
	"""
	Class representing a users trading portfolio. A Portfolio instance contains all current
	and past trades, balance, and portfolio performance metrics.
	
	Attributes:

		securities_trading: dict == {'ETHBTC': [Ex1Strategy, BinanceWrapper], 'TSLA': [Ex2Strategy, IBWrapper]}
			- dictionary containing security str as key, and [Strategy, APIWrapper] as the value

		past_trades: List(Trade)
			- a list of Trade objects, each containing information about a completed trade
		
		current_trades: List(Trade)
			- a list of Trade objects, each containing an in-progress trade
	"""



	def __init__():
		securities_trading = {}
		past_trades = []
		current_trades = []

	def add_security(name: str, strategy, api_wrapper):
		self.securities_trading[name] = [strategy, api_wrapper]

	def add_past_trade():
		pass

	def add_current_trade():
		pass