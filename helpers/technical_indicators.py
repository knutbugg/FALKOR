import numpy as np
import pandas as pd

def sma(price_list, n):
    """Calculates the sma for each row in price_series and outputs sma_list
    containing the ma calculations
    REPRESENTATION INVARIANTS:
        sma_list = len(price_list) - n

    >>> s = SimpleStrategy('mock')
    >>> s.sma([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 3)
    [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]
    """
    sma_list = []

    # populate last_n_prices with the first n prices
    last_n_prices = []
    for index, price in enumerate(price_list):
        if index < n:
            last_n_prices.append(price)
        else:
            # proceed with ma calculations
            sma = sum(last_n_prices) / n
            sma_list.append(sma)

            # update last_n_prices
            last_n_prices.pop(0)

            last_n_prices.append(price)
    return sma_list


def ema(price_list, n):
    """Calculates the ema for each row in price_series and outputs ema_list
    containing the ema calculations
    REPRESENTATION INVARIANTS:
        ema_list = len(price_list) - n

    >>> ema([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], 3)
    """
    multi = (2 / (n + 1))

    # Initialize the first entries of ema_list with a sma
    ema_list = [0] * len(price_list)
    sma_list = sma(price_list, n)[:n]

    for i in range(len(ema_list)):
        if i < n:
            curr_ema = sma_list[i]
        else:
            past_ema = ema_list[i-1]
            curr_ema = (price_list[i] - (past_ema * multi)) + past_ema

        ema_list[i] = curr_ema

    return ema_list[n:]


def rsi(price_list, n):
    pass
    # rsi_list = []
    # last_n_periods = []
    #
    # for index, price in enumerate(price_list):
    #     # while index < n, simply fill up last_n_periods so we can have
    #     # data to compute the rest of the data
    #     if index < n:
    #         last_n_periods.append(price)
    #     else:
    #         avg_gain = _compute_avg_gain(last_n_periods)
    #         avg_loss = _compute_avg_loss(last_n_periods)
    #         if index < 2*n:
    #             # Begin without Smooth RS since not enough data until 2n
    #             first_rs = avg_gain / avg_loss
    #             rsi = 100 - (100 / (1 + first_rs))
    #             rsi_list.append(rsi)
    #         else:
    #             smoothed_rs = ((avg_gain * 13) + )


def _compute_avg_gain(last_n_periods):
    """Returns the avg gain from a list of prices"""
    total_gain = 0
    n = len(last_n_periods)

    last_price = last_n_periods[0]

    for price in last_n_periods[1:]:
        if price > last_price:
            total_gain += price - last_price
    return total_gain / n


def _compute_avg_loss(last_n_periods):
    """Returns the avg loss from a list of prices"""
    total_loss = 0
    n = len(last_n_periods)

    last_price = last_n_periods[0]

    for price in last_n_periods[1:]:
        if price < last_price:
            total_loss += price - last_price
    return total_loss / n


def bollinger_bands(price_list, n, mult=2):
    """Returns the middle, upper, and lower, Bollinger Bands. Returns
    len(price_list) - n length list of bbs.

    >>> s = SimpleStrategy('')
    >>> s.bollinger_bands([10, 20, 30, 40, 40, 41 ,42, 43, 44, 45, 45, 60, 80, 100], 3)
    """
    bb_list = []
    sma_list = sma(price_list, n)

    for index, price in enumerate(price_list[n:]):
        n_day_dev = np.std(price_list[index:index + n])
        middle_band = sma_list[index]
        upper_band = middle_band + (n_day_dev * mult)
        lower_band = middle_band - (n_day_dev * mult)
        bb_list.append([lower_band, middle_band, upper_band])

    return bb_list


def obv(volume_list, price_list):
    """Returns obv_list contains the obv for each entry in price_list and
    volume_list. NOTE - no difference in len(volume_list) and (obv_list)
    """
    obv_list = []
    last_obv = 0
    price_prev = price_list[0]
    for index, price in enumerate(price_list):
        volume = volume_list[index]

        if price > price_prev:
            last_obv += volume
        elif price == price_prev:
            pass
        else:
            last_obv -= volume

        price_prev = price
        obv_list.append(last_obv)
    return obv_list


def macd(price_list):

    a = ema(price_list, 12)
    b = ema(price_list, 24)
    return list(np.array(a[len(a)-len(b):]) - np.array(b))

def stochastic_oscillator():
    pass

