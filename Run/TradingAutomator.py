from TradeExecution.BinanceMoneyTrader import BinanceMoneyTrader


class TradingAutomator:
    def __init__(self):
        creds = {
            'client_id': '5lJ0uGit9PuUxHka3hBWhPmsi7dWyxEwvEntUZFKmm0xfNz3VjHWi5WSr5W1VBJV',
            'client_secret': 'BFWVs8ko7Cd4sjdQ9amGJTnToGWy9TbQWIjeorSCj23FGiwFaknzkgLPcrgWrxsw'
        }

        self.trading_interface = BinanceMoneyTrader(creds['client_id'],
                                                    creds['client_secret'])

