import time
from TradeExecution.PaperTrader import PaperTrader
from ChartObjects.ChartHistoryBunch import ChartHistoryBunch
from Strategy.SimpleStrategy import SimpleStrategy


class Automator:
    """Automation class used to take signals from Strategy and send them to
    TradeExecutor. It also communicates with ChartHistoryBunch to keep it
    updated. """

    def __init__(self):
        # TradeExecutor - select here what trader to use
        self.trading_interface = PaperTrader(wallet=1000)

    def back_test(self, symbol, history):
        """Select a specific starting amount of data to be used to create a
        ChartHistoryBunch. Then, after the cutoff slice, begin to live_trade
        through the data, recording all made signals and recording profit. """
        pass

    def live_trade(self, symbol, history, interval):
        """ Live trade on a symbol. Take signals from our Strategy and perform
        necessary trades with TradeExecutor. Update ChartHistoryBunch
        accordingly.
        """
        # Create a ChartHistoryBunch to store given historical candlesticks
        chart_history_bunch = ChartHistoryBunch(symbol, history, interval)
        # Create a Strategy, sending the ChartHistoryBunch as a param
        strategy = SimpleStrategy(chart_history_bunch)
        # Run possible preparation code before loop (ex. training a nn)
        strategy.prepare()

        # Automation loop
        while True:
            # Generate buy / sell signals in a {}
            signals = strategy.generate_signals()
            output = self.trading_interface.trade_from_signals(signals)

            # Print out all trades made, status, account info etc.
            print(output)

            # Update dataset and strategy with the new data
            update_info = chart_history_bunch.update(symbol, interval)
            strategy.update(update_info)
            # sleep for specified time (s)
            time.sleep(6)
