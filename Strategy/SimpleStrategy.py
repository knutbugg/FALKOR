from Strategy.Strategy import Strategy
import pandas as pd
import numpy as np

class SimpleStrategy(Strategy):
    """Simple template for creating signals on the data. """
    def __init__(self, chart_history_bunch):
        """Initialize the strategy with its ChartHistoryBunch"""
        self.chart_history_bunch = chart_history_bunch

    def prepare(self):
        """Perform any initial computations. Train a nn, etc. """
        pass

    def generate_signals(self):
        """Generate all signals from the current data"""
        pass

    def update(self, update_info):
        """Update the corresponding data files with any changes """
        pass

    def get_technical_indicators(self):
        pass

