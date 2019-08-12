from binance.client import Client

from chartobjects.Charting import Charting
from exchangeoperators.BinanceOperator import BinanceOperator
from strategy.TechnicalIndicators import ema, sma, bollinger_bands, macd, obv


from strategy.TINNStrategy import prepare_data


operator = BinanceOperator()

data_df = operator.get_historical_candlesticks('ETHBTC', Client.KLINE_INTERVAL_1MINUTE, 'January 19 2019 6:00', 'January 19 2019 6:53')
tech_data_df = prepare_data(data_df)

tis = ['sma10', 'bb10_low', 'bb10_mid',
               'bb10_up', 'bb20_low', 'bb20_mid', 'bb20_up']

chart = Charting(df=tech_data_df, col_label='time', row_label='close', tech_inds=tis)
chart.chart_to_image('chart_image.png')




