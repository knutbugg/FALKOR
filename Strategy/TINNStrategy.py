from Strategy.Strategy import Strategy
from Strategy.TechnicalIndicators import sma, ema, macd, bollinger_bands, obv
from Strategy.TINN import *
import pandas as pd
import numpy as np

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

    def prepare(self):
        """Perform any initial computations. Train a nn, etc. """
        """
        self.nn = NN()
        training_data = ...
        results = self.nn.train(training_data, ... )
        print(results)
        
        """


    def generate_signals(self, new_data):
        """Generate all signals from the current data. """
        signals = self.nn.predict(new_data)
        return signals

    def update(self, update_info):
        """Update the corresponding data files with any changes. """
        pass

    def get_technical_indicators(self):
        pass


