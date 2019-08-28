# FALKOR | Automated trading platform designed for PyTorch models
> Minimalistic, lightweight, and powerful stock and cryptocurrency trading platform designed fully in Python. Build custom PyTorch models for stock price prediction. Real-time portfolio management and custom trading strategies. To maximize safety, FALKOR monitors stop-losses and provides real-time alerting. 

![](arch.jpg)

### Features
1. PyTorch models for price prediction
2. Backtesting for profitability
3. Easy to extend to a new security with custom APIWrappers
4. Live automated trading with stop loss support

#### Extras: 
  * tool for hand-labelling buy/sell points on historical candlestick chart
  * generate technical indicators for OCHLV candlestick data
  * alerting

### Prerequisites
* GPU CUDA enabled device
* Knowledge of PyTorch models

### Packages
* Anaconda
* Python 3
* PyTorch
* Pandas
* + python packages for api wrappers

### Installation 
```
git clone git@github.com:vdyagilev/FALKOR.git
cd FALKOR
```

### Running Live Trade

Edit the settiings within LiveTrade and run the following command:

```
python LiveTrade.py
```

### Running a Back Test

Edit the settings within BackTest and run the following command:

```
python BackTest.py
```

### Creating your own trading Strategy

Extend the Abstract Strategy class and implement the following methods:

``` python
class Strategy:
    """Abstract class representing a Strategy used by Gekko. The child class must create all NotImplemented methods"""

    def feed_data(live_df):
        """Feed in a DataFrame of the last 30 ochl periods."""

        raise NotImplementedError

    def generate_signals():
        """Returns a list of trading signals"""

        raise NotImplementedError

    def update():
        """Run whatever operations necessary to keep the strategy up-to-date with current data"""

        raise NotImplementedError
```

### Tips and tricks


## Release History

1.0 - ... - Vladimir Dyagilev

2.0 - ... - Eric Hasegawa, Vladimir Dyagilev

### Founder

Vladimir Dyagilev – [My Projects](https://vladimirdyagilev.com) 
[My GitHub](https://github.com/vdyagilev/)

### Contributing

1. Fork it (<https://github.com/vdyagilev/FALKOR/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

**NOTE: We are not responsible for any assets gained or lost through the use of FALKOR. Use at your own risk, and please, ensure you have backtested your Strategy and implemented stop losses**
