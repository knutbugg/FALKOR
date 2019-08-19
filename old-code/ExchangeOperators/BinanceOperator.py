from binance.client import Client
from exchangeoperators.credentials import creds
from exchangeoperators.ExchangeOperator import ExchangeOperator
import pandas as pd


def type_format_df(df, wanted_type):
	return df.applymap(wanted_type)

class BinanceOperator(ExchangeOperator):
	"""Used to directly operate with the Binance exchange api. Has all relevant
	api methods in their raw form. """
	def __init__(self):
		"""Initialize BinanceOperator with api credentials."""
		client_id, client_secret = creds[0], creds[1]
		self.client = Client(client_id, client_secret)

	def get_market_depth(self, symbol):
		depth = self.client.get_order_book(symbol=symbol)
		return depth

	def get_recent_trades(self, symbol):
		trades = self.client.get_recent_trades(symbol=symbol)
		return trades

	def get_historical_trades(self, symbol):
		trades = self.client.get_historical_trades(symbol=symbol)
		return trades

	def get_aggregate_trades(self, symbol):
		trades = self.client.get_aggregate_trades(symbol=symbol)
		return trades

	def get_24hr_tickers(self):
		tickers = self.client.get_ticker()
		return tickers

	def get_symbol_info(self, symbol):
		info = self.client.get_symbol_info(symbol)
		return info

	def get_candlesticks(self, symbol, interval):
		candles = self.client.get_klines(symbol=symbol,
										 interval=interval)
		return candles

	"""
	Candlestick data response from Binance is in the following format:

	[
	  [
		1499040000000,      // Open time
		"0.01634790",       // Open
		"0.80000000",       // High
		"0.01575800",       // Low
		"0.01577100",       // Close
		"148976.11427815",  // Volume
		1499644799999,      // Close time
		"2434.19055334",    // Quote asset volume
		308,                // Number of trades
		"1756.87402397",    // Taker buy base asset volume
		"28.46694368",      // Taker buy quote asset volume
		"17928899.62484339" // Ignore.
	  ]
	]

	"""

	def get_historical_candlesticks(self, symbol, interval, start_time,
									end_time=None):
		"""Returns candlestick data in the form
		['time', 'open', 'high', 'low', 'close', 'volume']
		"""
		if end_time is None:
			candles = self.client.get_historical_klines(symbol, interval,
														start_time)
		else:
			candles = self.client.get_historical_klines(symbol, interval,
														start_time, end_time)

		data_df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close',
								   'volume', 'close_time','quote_asset_volume',
								   'num_trades', 'tkbbav', 'tkqav', 'ign.'])
		data_df = data_df[['time', 'open', 'high', 'low', 'close', 'volume']]
		# Convert all numbers from str to float
		data_df = type_format_df(data_df, float)
		return data_df

	def get_account_info(self):
		return self.client.get_account()

	def get_asset_balance(self, asset):
		return self.client.get_asset_balance(asset=asset)

	def get_trade_fees(self):
		return self.client.get_trade_fee()
