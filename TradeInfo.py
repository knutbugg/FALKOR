import datetime

class TradeInfo:
	"""Contains all information about a trade that FALKOR has completed"""
	
	def __init__(self, symbol: str,  kind: str, share_amount: int, share_price, success=True):
		self.success = success
		self.kind = kind
		self.share_amount = share_amount
		self.share_price = share_price
		self.timestamp = datetime.datetime.now()

	def __str__(self):
		if self.success: 
			went_through = "successful"
		else:
			went_through = "not completed"
			
		return "The {} trade for #{} of {} was {}".format( self.kind, self.share_amount, self.share_price, went_through)
