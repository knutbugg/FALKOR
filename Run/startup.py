from Backtesting.HistoricalDataFeed import HistoricalDataFeed
from Backtesting.HistoricalDataSaver import HistoricalDataSaver
creds = {'client_id': '5lJ0uGit9PuUxHka3hBWhPmsi7dWyxEwvEntUZFKmm0xfNz3VjHWi5WSr5W1VBJV',
         'client_secret':'BFWVs8ko7Cd4sjdQ9amGJTnToGWy9TbQWIjeorSCj23FGiwFaknzkgLPcrgWrxsw'
         }


x = HistoricalDataFeed('binance', creds)
data = x.historical_data('ETHBTC', '1h', 'yesterday')
x.save_historical_data(data, 'data.csv')
