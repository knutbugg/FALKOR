from binance.client import Client

from chartobjects.Charting import Charting
from exchangeoperators.BinanceOperator import BinanceOperator
from strategy.TechnicalIndicators import ema, sma, bollinger_bands, macd, obv


from strategy.TINNStrategy import prepare_data


operator = BinanceOperator()

data_df = operator.get_historical_candlesticks('ETHBTC', Client.KLINE_INTERVAL_1MINUTE, 'January 1 2015 6:00', 'August 10 2019 6:00')
data_df.to_csv('ETHBTC-4year-1min')




