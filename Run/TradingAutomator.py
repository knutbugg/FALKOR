from TradeExecution.BinanceMoneyTrader import BinanceMoneyTrader
from TradeExecution.PaperTrader import PaperTrader
from ChartObjects.ChartHistoryBunch import ChartHistoryBunch
from TradeExecution.BinanceMoneyTrader import BinanceOperator
from ExchangeOperators.BinanceLiveDataFeed import BinanceLiveDataFeed
from Strategy.SimpleStrategy import SimpleStrategy



class TradingAutomator:
    def __init__(self):
        creds = {
            'client_id': '5lJ0uGit9PuUxHka3hBWhPmsi7dWyxEwvEntUZFKmm0xfNz3VjHWi5WSr5W1VBJV',
            'client_secret': 'BFWVs8ko7Cd4sjdQ9amGJTnToGWy9TbQWIjeorSCj23FGiwFaknzkgLPcrgWrxsw'
        }
        self.binance_operator = BinanceOperator(creds['client_id'], creds['client_secret'])
        # Paper Trading - select here what trader to use
        self.trading_interface = PaperTrader(wallet=1000, operator=self.binance_operator)
        # self.trading_interface = BinanceMoneyTrader(creds['client_id'],
        #                                             creds['client_secret'])

    def backtest(self, symbol, history):
        """Select a specific starting amount of data to be used to create a ChartHistoryBunch. Then,
        after the cutoff slice, begin to live_trade through the data, recording all made signals
        and recording profit. """
    def live_trade(self, symbol, history):
        """ have an object - ChartHistoryBunch - contain all the data required from
        historical candles. Then, have financial data be pulled to exist alongside
        the ChartHistoryBunch with the LiveDataFeed. Perform any desired computations on the set of
        ChartHistoryBunch + new data to generate any buy/sell/... signal and perform
        that action. Next, merge the ChartHistoryBunch + new data -> a new ChartHistoryBunch.
        Then repeat.
        """
        chart_history_bunch = ChartHistoryBunch(symbol, history)
        data_feed = BinanceLiveDataFeed()
        strategy = SimpleStrategy(chart_history_bunch, data_feed)
        while True:
            # Generate buy / sell signals in a {}
            signals = strategy.generate_signals()
            output = self.trading_interface.trade_from_signals(signals)
            # Print out all trades made, status, account info etc.
            print(output)

            # Update dataset and strategy with the new data
            chart_history_bunch_update_info = chart_history_bunch.update()
            strategy.update(chart_history_bunch_update_info)


x = TradingAutomator()
x.live_trade('ETHBTC', '../Run/eth-1D-lastyear.csv')

