import time
from Portfolio import Portfolio
from Gekko import Gekko
from BudFox import BudFox

from strategies.CNN_Strategy import CNN_Strategy
from api_wrappers.BinanceWrapper import BinanceWrapper

from models.CNN.CNN import CNN

class TheTerminal:
    """TheTerminal is used to interact with the running Falkor program. It allows the user to add new securities
    to trade on, viewing portfolio, current trades, alerting, etc. """
	
    def __init__(self):
        """Initialize TheTerminal"""
            
        self.portfolio = Portfolio(paper_trade=True)
        self.gekko = Gekko(self.portfolio)

	# Add ETHBTC as a security 
        model = CNN()
        cnn_strat = CNN_Strategy(model=model, weights_path='models/cnn/weights/', image_path='models/cnn/images/')
        api_wrapper = BinanceWrapper(client_id='nBjgb83VMNvqq45b3JdWUIsJDalWlXxHI2bvDz9oLdW7KgOLPvJCp30CHnthjfNJ', client_secret='5bBN7s7h37kUvmGIpF9FTAtspBY93WirwhTh39PV7AlKSlUE2S4EEe9b3OZVYIqd')
        self.add_security_to_portfolio('ETHBTC', '1m', cnn_strat, api_wrapper)
    
    def add_security_to_portfolio(self, name, interval, strategy, api_wrapper):
    	"""Adds a security to portfolio"""

    	self.portfolio.add_security(name, interval, strategy, api_wrapper)

    def run(self, interval_secs: int):
        """Begin pulling recent candles, generating trading signals, and trading."""
        while True:
            # trade portfolio
            self.gekko.trade_portfolio()

            # sleep until next candlestick of data is avaliable
            print("Done One Iteration")
            time.sleep(interval_secs)


if __name__ == '__main__':
    terminal = TheTerminal()
    terminal.run(5)
