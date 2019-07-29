from ExchangeOperators.BinanceOperator import BinanceOperator

class BinanceLiveDataFeed:
    def __init__(self):
        self.binance_op = BinanceOperator(
            '5lJ0uGit9PuUxHka3hBWhPmsi7dWyxEwvEntUZFKmm0xfNz3VjHWi5WSr5W1VBJV',
            'BFWVs8ko7Cd4sjdQ9amGJTnToGWy9TbQWIjeorSCj23FGiwFaknzkgLPcrgWrxsw')

    def pull(self, symbol):
        """Return a Tuple containing a summary of recent trades, and the raw
        trade details of recent trades. """
        recent_trades = self.binance_op.get_recent_trades(symbol)
        summary_of_trades = {}
        summary_of_trades['price_last_5_trades'] = 0.2 * sum([
            float(trade['price']) for trade in recent_trades[len(recent_trades)-5:]
        ])
        print(summary_of_trades)
        return summary_of_trades, recent_trades

