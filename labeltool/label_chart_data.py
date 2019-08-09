from backtesting.HistoricalDataSaver import HistoricalDataSaver
from exchangeoperators.AlphaVantageOperator import AlphaVantageOperator
from TradingSignalLabellingTool import TradingSignalLabellingTool
import pandas as pd
# Download lots of financial data

while True:
    symbol = input("Enter symbol: ")
    path = symbol + '.csv'

    av = AlphaVantageOperator("PYXRW3T85X6XF44V")
    hds = HistoricalDataSaver(path)
    hds.hist_ohlcv_csv(symbol, av)

    df = pd.read_csv(path)
    col_label = 'date'
    row_label = 'close'
    csv_path = symbol + "_labels.csv"

    t = TradingSignalLabellingTool(df, col_label, row_label, csv_path)
    t.show_chart()
