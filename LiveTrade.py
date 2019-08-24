import time
from Portfolio import Portfolio
from Gekko import Gekko
from BudFox import BudFox

from strategies.CNN_Strategy import CNN_Strategy
from api_wrappers.BinanceWrapper import BinanceWrapper
from backtesting.CNNBacktest import CNNBacktest
from Security import Security

from models.CNN.CNN import CNN


class LiveTrade:
    """
    LiveTrade is used to interact with the running Falkor program. 
    It allows the user to add new securities to trade on, viewing 
    portfolio, current trades, alerting, etc. 
    
    Attributes:

        portfolio: Portfolio
            - portfolio containing all Security's trading

        gekko: Gekko
            - Gekko instance

    Methods:

        run(self, interval_secs: int)
            - Main method used to run FALKOR
    """
    
    def __init__(self):
        """Initialize LiveTrade"""
            
        self.portfolio = Portfolio()
        self.gekko = Gekko(self.portfolio)

        # Add ETHBTC as a security 
        model = CNN()
        cnn_strat = CNN_Strategy(model=model, weights_path='models/cnn/weights/', image_path='strategies/images/')
        api_wrapper = BinanceWrapper(client_id='nBjgb83VMNvqq45b3JdWUIsJDalWlXxHI2bvDz9oLdW7KgOLPvJCp30CHnthjfNJ', client_secret='5bBN7s7h37kUvmGIpF9FTAtspBY93WirwhTh39PV7AlKSlUE2S4EEe9b3OZVYIqd')
        ethbtc = Security('ETHBTC', '1m', shares=0, api_wrapper=api_wrapper, strategy=cnn_strat)

        self.portfolio.add_security(ethbtc)

    def run(self, interval_secs: int):
        """Begin pulling recent candles, generating trading signals, and trading."""
        while True:

            # trade portfolio
            self.gekko.trade_portfolio()

            # display portfolio information
            self.portfolio.display()

            # sleep until next candlestick of data is avaliable
            time.sleep(interval_secs)

if __name__ == '__main__':
    lt = LiveTrade()
    lt.run(5)
