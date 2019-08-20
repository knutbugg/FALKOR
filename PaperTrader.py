
class PaperTrader:
	"""
	Handles BudFox requests to buy/sell securities on a Paper Trading account. 

	Attributes:

		liquid: float
			- total money avaliable to use for withdrawal or investing

		securities_owned: {'ETHBTC': [(12, 10.0), (24, 12.0)], 'TSLA': [(2, 200), (3, 450)]}
			- dictionary containing security name as key. and list of tuples containing order amount and price.

	"""

	def __init__(self, liquid: float):
		"""Initialize PaperTrader instance"""

		self.liquid = liquid

		self.securities_owned = {}


	def buy_order(security: str, amount: int, price: float):
		"""Adds an order of security with amount of shares at price value to self.securities_owned"""

		if security in self.securities_owned:
			self.securities_owned[security].append( (amount, price) )

		else:
			self.securities_owned[security] = [ (amount, price) ]

		# subtract cost of investment from self.liquid
		self.liquid -= amount * price 

	def sell_order(security: str, amount: int, price: float):
		"""Sell all shares of security regardless of amount"""

		shares = self.securities_owned[security]

		for index, tup in enumerate(shares):
			bought_amount, bought_price = tup

			# add profit to self.liquid
			self.liquid += (bought_amount * bought_price) - (bought_amount * price) 

			# remove sold shares
			shares.pop(index)
