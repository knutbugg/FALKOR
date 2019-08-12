from strategy.Strategy import Strategy
from strategy.TechnicalIndicators import sma, ema, macd, bollinger_bands, obv



class TINNStrategy(Strategy):
    """Strategy with a Neural Network trained on feature data of technical
     signals.
     """

    def __init__(self, chart_history_bunch):
        """Initialize the strategy with its ChartHistoryBunch. """
        self.chart_history_bunch = chart_history_bunch

        self.sma = sma
        self.ema = ema
        self.macd = macd
        self.bollinger_bands = bollinger_bands
        self.obv = obv


    # def prepare_data(self):
    #     """Perform any initial computations. Train a nn, etc. """
    #
    #     data_path = Path('../data/')
    #     datasets = ['ETHBTC.csv', ]
    #
    #     for dataset in datasets:
    #         dataset_path = data_path / dataset
    #         dataset_df = pd.read_csv(dataset_path)
    #
    #         # # Format dataset into
    #         # dataset_df = dataset_df[
    #         #     ['date', 'minute', 'label', 'open', 'high', 'low', 'close',
    #         #      'volume']].copy()
    #         #
    #         # # TODO Either deal with missing data - or ignore it and see if results remain unaffected
    #         #
    #         # # Convert date minute label columns into one columns, time.
    #         # timestamps = []
    #         # for index, row in dataset_df.iterrows():
    #         #     f = str(row.date) + '|' + row.minute
    #         #     timestamp = datetime.timestamp(
    #         #         datetime.strptime(f, '%Y%m%d|%H:%M'))
    #         #     timestamps.append(timestamp)
    #         #
    #         # # Add the time column and remove the redundent columns
    #         # dataset_df['time'] = timestamps
    #         dataset_df = dataset_df[
    #             ['time', 'open', 'high', 'low', 'close', 'volume']]
    #
    #         print("Loaded {}".format(dataset))
    #         ##############################################################################
    #
    #         # Create Technical Indicators for dataset_df
    #         price_df = dataset_df.open
    #         volume_df = dataset_df.volume
    #
    #         ema10_list = ema(price_df.tolist(), n=10)
    #         ema20_list = ema(price_df.tolist(), n=20)
    #         ema50_list = ema(price_df.tolist(), n=20)
    #
    #         sma10_list = sma(price_df.tolist(), n=20)
    #         sma20_list = sma(price_df.tolist(), n=20)
    #         sma50_list = sma(price_df.tolist(), n=20)
    #
    #         bb10 = bollinger_bands(price_df.tolist(), 10, mult=2)
    #         bb10_low = [x[0] for x in bb10]
    #         bb10_mid = [x[1] for x in bb10]
    #         bb10_up = [x[2] for x in bb10]
    #
    #         bb20 = bollinger_bands(price_df.tolist(), 20, mult=2)
    #         bb20_low = [x[0] for x in bb20]
    #         bb20_mid = [x[1] for x in bb20]
    #         bb20_up = [x[2] for x in bb20]
    #
    #         macd_list = macd(price_df.tolist())
    #
    #         obv_list = obv(volume_df.tolist(), price_df.tolist())
    #
    #         # We have indicators for a recent slice of data, so we cut out previous candlesticks
    #
    #         ti_dict = {'ema10': ema10_list, 'ema20': ema20_list,
    #                    'ema50': ema50_list, 'sma10': sma10_list,
    #                    'sma20': sma20_list, 'sma50': sma50_list,
    #                    'macd': macd_list, 'obv': obv_list,
    #                    'bb10_low': bb10_low, 'bb10_mid': bb10_mid,
    #                    'bb10_up': bb10_up, 'bb20_low': bb20_low,
    #                    'bb20_mid': bb20_mid, 'bb20_up': bb20_up}
    #         l = ''
    #         c = 100 ** 10
    #         for label, data in ti_dict.items():
    #             if len(data) < c:
    #                 l, c = label, len(data)
    #
    #         # First cut down our TI's
    #         for label, data in ti_dict.items():
    #             cut_amount = len(data) - len(macd_list)
    #             ti_dict[label] = data[cut_amount:]
    #
    #         # Next cut down our dataset_df
    #         dataset_df = dataset_df.iloc[dataset_df.shape[0] - len(macd_list):]
    #         dataset_df.shape[0]
    #
    #         # Add data
    #         for label, data in ti_dict.items():
    #             dataset_df[label] = data
    #
    #         print("Created Technical Indicators")
    #
    #         ##############################################################################################
    #
    #         # Format Technical Indicators used in plotting into dictionary {color : [(x1, y1), (x2, y2), (x3, y3)]
    #
    #         tech_inds = ['bb20_low', 'bb20_mid', 'bb20_up']
    #
    #         ###############################################################################################
    #         print("Launching Label Tool")
    #         # Run our labelling software to hand label buy and sell points on each dataset_df
    #         col_label = 'time'
    #         row_label = 'open'
    #         csv_path = dataset[:-3] + '_labels.csv'
    #         tool = LabelTool(dataset_df, col_label, row_label, csv_path,
    #                          tech_inds=tech_inds)
    #         tool.show_chart()
    #         print("Completed {}".format(dataset))

    def generate_signals(self, new_data):
        """Generate all signals from the current data. """
        signals = self.nn.predict(new_data)
        return signals

    def update(self, update_info):
        """Update the corresponding data files with any changes. """
        pass

    def get_technical_indicators(self):
        pass


def prepare_data(dataset_df):
    dataset_df = dataset_df[
        ['time', 'open', 'high', 'low', 'close', 'volume']].copy()

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
