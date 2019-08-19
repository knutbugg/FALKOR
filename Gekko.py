
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

	def _trade(security: str, strategy, api_wrapper):
		"""Helper method for self.trade_portfolio(). Runs strategy for security and sends signals to api_wrapper"""
		


	def trade_portfolio():
		"""Iterates through every security in self.portfolio, using their specified Strategy and APIWrapper"""

		for security, specs_list in self.portfolio:
			security, strategy, api_wrapper = security, specs_list[0], specs_list[1]

			self._trade(security, strategy, api_wrapper)


g = Gekko(Portfolio())