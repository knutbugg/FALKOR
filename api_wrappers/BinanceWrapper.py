from api_wrappers.APIWrapper import APIWrapper
from binance.client import Client
import pandas as pd

class BinanceWrapper(APIWrapper):
	"""APIWrapper for the Binance Exchange"""

	def __init__(self, client_id, client_secret):
		"""Initialize BinanceWrapper"""
		self.client = Client(client_id, client_secret)

	def historical_candles(self, symbol: str, interval:str, start_time: str,
						   end_time: str):
		"""
		Returns DataFrame with columns = ['time', 'open', 'close', 'high', 'low', 'volume'] for the security. Number of rows is equal to time interval divided by days

		>>> id = 'nBjgb83VMNvqq45b3JdWUIsJDalWlXxHI2bvDz9oLdW7KgOLPvJCp30CHnthjfNJ'
		>>> sec = '5bBN7s7h37kUvmGIpF9FTAtspBY93WirwhTh39PV7AlKSlUE2S4EEe9b3OZVYIqd'
		>>> b = BinanceWrapper(id, sec)
		>>> b.historical_daily("ETHBTC", "1 Jan, 2017", "now", "1d")
		"""

		candles = self.client.get_historical_klines(symbol, interval,
													start_str=start_time, end_str=end_time)

		data_df = pd.DataFrame(candles, columns=['time', 'open', 'high', 'low', 'close',
								   'volume', 'close_time','quote_asset_volume',
								   'num_trades', 'tkbbav', 'tkqav', 'ign.'])

		data_df = data_df[['time', 'open', 'high', 'low', 'close', 'volume']]

		# Convert all numbers from str to float
		data_df = data_df.applymap(float)

		return data_df

	def last_candles(self, num, symbol, interval):
		"""Returns DataFrame with columns =['time', 'open', 'close', 'high', 'low', 'volume']
		with only one row for the last candle created

		NOTE: only goes back to num*interval candles max

		>>> id = 'nBjgb83VMNvqq45b3JdWUIsJDalWlXxHI2bvDz9oLdW7KgOLPvJCp30CHnthjfNJ'
		>>> sec = '5bBN7s7h37kUvmGIpF9FTAtspBY93WirwhTh39PV7AlKSlUE2S4EEe9b3OZVYIqd'
		>>> b = BinanceWrapper(id, sec)
		>>> b.last_candle("ETHBTC", "1m")
		"""

		candles = self.client.get_historical_klines(symbol, interval,
													start_str='2 days ago')

		data_df = pd.DataFrame(candles,
							   columns=['time', 'open', 'high', 'low', 'close',
										'volume', 'close_time',
										'quote_asset_volume',
										'num_trades', 'tkbbav', 'tkqav',
										'ign.'])

		data_df = data_df[['time', 'open', 'high', 'low', 'close', 'volume']]
		data_df = data_df.iloc[data_df.shape[0]-num:, :] # take only last row

		# Convert all numbers from str to float
		data_df = data_df.applymap(float)

		return data_df


	def buy_order(self, symbol:str, amount: int, price="market"):
		"""
		Created a buy order for amount. If price != "market", then a buy order
		will be created at price=price
		"""

		if price == "market":
			order = self.client.order_market_buy(symbol=symbol, quantity=amount)
		else:
			order = self.client.order_limit_buy(symbol=symbol, quantity=amount,
												price=price)
		return order

        def tickers(self):
            """Return a dictionary of all symbols as key and their current price as value"""
            return self.client.get_all_tickers()
        
        def sell_order(self, symbol: str, amount:int, price="market"):
		"""
		Created a sell order for amount. If price != "market", then a sell order
		will be created at price=price
		"""

		if price == "market":
			order = self.client.order_market_sell(symbol=symbol, quantity=amount)
		else:
			order = self.client.order_limit_sell(symbol=symbol, quantity=amount,
												price=price)
		return order

	def check_balance(self, ):
		"""Returns portfolio balance"""

		raise NotImplementedError

	def trade_status(self, symbol:str, trade_id: str):
		"""Returns the status for trade with id trade_id"""

		return self.client.get_order(symbol=symbol, orderId=trade_id)

	def cancel_order(self, symbol: str, trade_id: str):
		"""Cancel order with trade_id"""
		result = self.client.cancel_order(symbol=symbol, orderId=trade_id)

	def account_info(self):
		return self.client.get_account()

	def asset_balance(self, symbol: str):
		return self.client.get_asset_balance(asset=symbol)

	def get_trades(self, symbol: str):
		return self.client.get_my_trades(symbol=symbol)

	def get_trade_fee(self, symbol: str):
		return self.client.get_trade_fee(symbol=symbol)
