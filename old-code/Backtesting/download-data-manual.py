from binance.client import Client
import pandas as pd
from backtesting.HistoricalDataSaver import HistoricalDataSaver
from exchangeoperators.BinanceOperator import BinanceOperator


operator = BinanceOperator()
data = operator.get_historical_candlesticks('ETHBTC', Client.KLINE_INTERVAL_1MINUTE, '2018', end_time=None)
data.to_csv('ETHBTC.csv', index=False)
