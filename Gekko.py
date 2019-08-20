from BudFox import BudFox
from BookWorm import BookWorm

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

	def __init__(self, portfolio):
		"""Initialize Gekko"""
		
		self.portfolio = portfolio
		self.bud_fox = BudFox()
		self.book_worm = BookWorm()

	def _trade(self, security: str, interval: str, strategy, api_wrapper):
		"""Helper method for self.trade_portfolio(). Runs strategy for security and sends signals to api_wrapper"""
		
		# Get 100 most recent candles of data
		# NOTE: Creating technical indicators cuts 100 candles down to ~ 50-80 depending on the indicator. We want 30 
		# periods, so we take 100 to leave us with plenty of room to spare

		last_candles = self.book_worm.last_candles(100, api_wrapper, security, interval)

		# Get trading signals from strategy
		strategy.feed_data(last_candles)
		signal = strategy.predict()
		strategy.update()

		# Send signal to BudFox for realization

		# tell BudFox to paper trade instead of real trade
		if self.portfolio.paper_trade:
			self.bud_fox.paper_trade = True

		trade_info = self.bud_fox.send_trading_signal(signal)

		return trade_info

	def trade_portfolio(self):
		"""Iterates through every security in self.portfolio, using their specified Strategy and APIWrapper"""

		for security, specs_list in self.portfolio.securities_trading.items():
			interval, strategy, api_wrapper = specs_list[0], specs_list[1], specs_list[2]

			trade_info = self._trade(security, interval, strategy, api_wrapper)

			print(trade_info)
