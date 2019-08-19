
class Gekko:
	"""
	The engine room of Falkor. Every iteration, Gekko receives an input of live ochl data. It sends this 
	data to the selected Strategy, which generates buy/sell signals and real-time model updating. 
	Gekko takes these signals and sends them to BudFox who	will realize them. 
	Gekko then updates portfolio with any profits made
	
	Attributes:
		
		bud_fox: BudFox
			- An instance of BudFox class used to send buy/sell signals to an APIWrapper for realization

		portfolio: Portfolio
			- An instance of Portfolio class that stores all trades and securities	

		book_worm: BookWorm
			- An instance of BookWorm class used for pulling live data from APIWrappers

	"""

	def __init__(portfolio):
		"""Initialize Gekko"""
		
		self.portfolio = portfolio
		self.bud_fox = BudFox()
		self.book_worm = BookWorm()

	def _trade(security: str, interval: str, strategy, api_wrapper):
		"""Helper method for self.trade_portfolio(). Runs strategy for security and sends signals to api_wrapper"""
		
		# Get 30 most recent candles of data
		last_candles = self.book_worm.last_candles(30, api_wrapper, security, interval)

		# Get trading signals from strategy
		strategy.feed_data(last_candles)
		signals = strategy.generate_signals()
		strategy.update()

		# Send signals to BudFox for realization
		self.bud_fox.receive_trading_signals(signals)
		self.bud_fox.realize_signals()

	def trade_portfolio():
		"""Iterates through every security in self.portfolio, using their specified Strategy and APIWrapper"""

		for security, specs_list in self.portfolio:
			interval, strategy, api_wrapper = specs_list[0], specs_list[1], specs_list[2]

			self._trade(security, interval, strategy, api_wrapper)
