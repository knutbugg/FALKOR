
class SimpleStrategy:
    """Simple template for creating signals on the data. """
    def __init__(self, chart_history_bunch, current_data):
        self.past_data = chart_history_bunch
        self.current_data = current_data

    def generate_signals(self):


    def update(self, ChartHistoryBunch_update_info):
        """Update the corresponding data files to the next data. """
        pass
