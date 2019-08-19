from Strategy import Strategy

class CNN_Strategy(Strategy):
	"""Abstract class representing a Strategy used by Gekko. The child class must create all NotImplemented methods"""

	def load_model(model):
		


	def feed_data(live_df):
		"""Feed in a DataFrame of the last 30 ochl periods."""

		

	def generate_signals():
		"""Returns a list of trading signals"""

		raise NotImplementedError

	def update():
		"""Run whatever operations necessary to keep the strategy up-to-date with current data"""

		raise NotImplementedError