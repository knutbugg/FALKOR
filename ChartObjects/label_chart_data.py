from binance.client import Client

from chartobjects.Charting import Charting
from exchangeoperators.BinanceOperator import BinanceOperator
from strategy.TechnicalIndicators import ema, sma, bollinger_bands, macd, obv

def prepare_data(dataset_df):

    dataset_df = dataset_df[['time', 'open', 'high', 'low', 'close', 'volume']].copy()

    # Create Technical Indicators for dataset_df
    price_df = dataset_df.close
    # Convert possible strings into floats
    price_list = [float(x) for x in price_df.tolist()]

    volume_df = dataset_df.volume
    volume_list = [float(x) for x in volume_df.tolist()]

    ema10_list = ema(price_list, n=10)
    ema20_list = ema(price_list, n=20)
    ema50_list = ema(price_list, n=20)

    sma10_list = sma(price_list, n=20)
    sma20_list = sma(price_list, n=20)
    sma50_list = sma(price_list, n=20)

    bb10 = bollinger_bands(price_list, 10, mult=2)
    bb10_low = [x[0] for x in bb10]
    bb10_mid = [x[1] for x in bb10]
    bb10_up = [x[2] for x in bb10]

    bb20 = bollinger_bands(price_list, 20, mult=2)
    bb20_low = [x[0] for x in bb20]
    bb20_mid = [x[1] for x in bb20]
    bb20_up = [x[2] for x in bb20]

    macd_list = macd(price_list)

    obv_list = obv(volume_list, price_list)

    # We have indicators for a recent slice of data, so we cut out previous candlesticks

    ti_dict = {'ema10': ema10_list, 'ema20': ema20_list,
               'ema50': ema50_list, 'sma10': sma10_list,
               'sma20': sma20_list, 'sma50': sma50_list,
               'macd': macd_list, 'obv': obv_list,
               'bb10_low': bb10_low, 'bb10_mid': bb10_mid,
               'bb10_up': bb10_up, 'bb20_low': bb20_low,
               'bb20_mid': bb20_mid, 'bb20_up': bb20_up}
    l = ''
    c = 100 ** 10
    for label, data in ti_dict.items():
        if len(data) < c:
            l, c = label, len(data)

    # First cut down our TI's
    for label, data in ti_dict.items():
        cut_amount = len(data) - len(macd_list)
        ti_dict[label] = data[cut_amount:]

    # Next cut down our dataset_df
    dataset_df = dataset_df.iloc[dataset_df.shape[0] - len(macd_list):]

    # Add data
    for label, data in ti_dict.items():
        dataset_df[label] = data

    return dataset_df


operator = BinanceOperator()

data_df = operator.get_historical_candlesticks('ETHBTC', Client.KLINE_INTERVAL_1MINUTE, 'January 18 2019 6:00', 'January 18 2019 7:30')
tech_data_df = prepare_data(data_df)

tis = ['sma10', 'sma20', 'sma50', 'bb10_low', 'bb10_mid',
               'bb10_up', 'bb20_low', 'bb20_mid', 'bb20_up']

chart = Charting(df=tech_data_df, col_label='time', row_label='close', tech_inds=tis)
chart.chart_to_image()
