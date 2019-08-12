from binance.client import Client

from chartobjects.Charting import Charting
from exchangeoperators.BinanceOperator import BinanceOperator
from strategy.TechnicalIndicators import ema, sma, bollinger_bands, macd, obv


from strategy.TINNStrategy import prepare_data


operator = BinanceOperator()

data_df = operator.get_historical_candlesticks('ETHBTC', Client.KLINE_INTERVAL_1MINUTE, 'January 1 2017 6:00', 'January 1 2019 6:00')
data_df.to_csv('ETHBTC-2year-1min')




