import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

import pandas as pd
from itertools import compress
from PIL import Image

import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import TICKLEFT, TICKRIGHT, Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from six.moves import xrange, zip

def _check_input(opens, closes, highs, lows, miss=-1):
    """Checks that *opens*, *highs*, *lows* and *closes* have the same length.
    NOTE: this code assumes if any value open, high, low, close is
    missing (*-1*) they all are missing
    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        sequence of opening values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    closes : sequence
        sequence of closing values
    miss : int
        identifier of the missing data
    Raises
    ------
    ValueError
        if the input sequences don't have the same length
    """

    def _missing(sequence, miss=-1):
        """Returns the index in *sequence* of the missing data, identified by
        *miss*
        Parameters
        ----------
        sequence :
            sequence to evaluate
        miss :
            identifier of the missing data
        Returns
        -------
        where_miss: numpy.ndarray
            indices of the missing data
        """
        return np.where(np.array(sequence) == miss)[0]

    same_length = len(opens) == len(highs) == len(lows) == len(closes)
    _missopens = _missing(opens)
    same_missing = ((_missopens == _missing(highs)).all() and
                    (_missopens == _missing(lows)).all() and
                    (_missopens == _missing(closes)).all())

    if not (same_length and same_missing):
        msg = ("*opens*, *highs*, *lows* and *closes* must have the same"
               " length. NOTE: this code assumes if any value open, high,"
               " low, close is missing (*-1*) they all must be missing.")
        raise ValueError(msg)

def candlestick2_ochl(ax, opens, closes, highs, lows, width=4,
                      colorup='k', colordown='r',
                      alpha=0.75):
    """Represent the open, close as a bar line and high low range as a
    vertical line.
    Preserves the original argument order.
    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        sequence of opening values
    closes : sequence
        sequence of closing values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    width : int
        size of open and close ticks in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    alpha : float
        bar transparency
    Returns
    -------
    ret : tuple
        (lineCollection, barCollection)
    """

    return candlestick2_ohlc(ax, opens, highs, lows, closes, width=width,
                             colorup=colorup, colordown=colordown,
                             alpha=alpha)


def candlestick2_ohlc(ax, opens, highs, lows, closes, width=4,
                      colorup='k', colordown='r',
                      alpha=0.75):
    """Represent the open, close as a bar line and high low range as a
    vertical line.
    NOTE: this code assumes if any value open, low, high, close is
    missing they all are missing
    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        sequence of opening values
    highs : sequence
        sequence of high values
    lows : sequence
        sequence of low values
    closes : sequence
        sequence of closing values
    width : int
        size of open and close ticks in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    alpha : float
        bar transparency
    Returns
    -------
    ret : tuple
        (lineCollection, barCollection)
    """

    _check_input(opens, highs, lows, closes)

    delta = width / 2.
    barVerts = [((i - delta, open),
                 (i - delta, close),
                 (i + delta, close),
                 (i + delta, open))
                for i, open, close in zip(xrange(len(opens)), opens, closes)
                if open != -1 and close != -1]

    rangeSegments = [((i, low), (i, high))
                     for i, low, high in zip(xrange(len(lows)), lows, highs)
                     if low != -1]

    colorup = mcolors.to_rgba(colorup, alpha)
    colordown = mcolors.to_rgba(colordown, alpha)
    colord = {True: colorup, False: colordown}
    colors = [colord[open < close]
              for open, close in zip(opens, closes)
              if open != -1 and close != -1]

    useAA = 0,  # use tuple here
    lw = 0.5,   # and here
    rangeCollection = LineCollection(rangeSegments,
                                     colors=colors,
                                     linewidths=lw,
                                     antialiaseds=useAA,
                                     )

    barCollection = PolyCollection(barVerts,
                                   facecolors=colors,
                                   edgecolors=colors,
                                   antialiaseds=useAA,
                                   linewidths=lw,
                                   )

    minx, maxx = 0, len(rangeSegments)
    miny = min([low for low in lows if low != -1])
    maxy = max([high for high in highs if high != -1])

    corners = (minx, miny), (maxx, maxy)
    ax.update_datalim(corners)
    ax.autoscale_view()

    # add these last
    ax.add_collection(rangeCollection)
    ax.add_collection(barCollection)
    return rangeCollection, barCollection


class Charting:
    """Create customizable financial price charts, with technical indicators.
    Features include saving charts as images, and hand-labelling buy/sell points.
    """
    def __init__(self, df, col_label, row_label, tech_inds=None):
        """Initialize with df with data, col_label for time, row_label for price
        and path to save .csv labelled data in csv_path. """

        # Storage of matplotlib lassoing points
        self.buy_points = []
        self.sell_points = []

        self.df = df
        self.col_label = col_label
        self.row_label = row_label

        # Convert all points into tuples for easier interaction
        self.x = self.df[col_label].tolist()
        self.y = self.df[row_label].tolist()
        self.points = list(zip(self.x, self.y))

        # Lasso properties for lassoing points with Labelling Tool
        self.lineprops_sell = {'color': 'red', 'linewidth': 2, 'alpha': 0.8}
        self.lineprops_buy = {'color': 'green', 'linewidth': 2, 'alpha': 0.8}
        self.lineprops_clear = {'color': 'orange', 'linewidth': 2, 'alpha': 0.8}

        # Create matplotlib figure
        self.fig = plt.figure()

        # Used to redraw unselected points
        self._recently_deleted_points = []

        self.tech_inds = tech_inds

    def chart_to_image(self, file_name):
        """Creates the specified chart and saves it to an image. """

        # Create axis for the price and technical indicator graph
        ax0 = self.fig.add_subplot(411)
        plt.axis('off')

        # Plot Price
        self.df.plot(x=self.col_label, y=self.row_label, ax=ax0, color='black', label='_nolegend_', linewidth=3)

        # Plot Technical Indicators
        if self.tech_inds:
            for col_name in self.tech_inds:
                ti_df = self.df[['time', col_name]].copy()
                ti_df.plot(x='time', y=col_name, ax=ax0, label='_nolegend_')


        # Plot Volume as Bar Chart on the bottom
        # Turn off the axes and background lines
        ax1 = self.fig.add_subplot(412)
        plt.axis('off')

        # Plot candlesticks
        candlestick2_ochl(width=0.4, colorup='g', colordown='r',
                          ax=ax1, opens=self.df['open'],
                          closes=self.df['close'],
                          highs=self.df['high'], lows=self.df['low'], )


        # Create axis for the volume bar chart
        ax2 = self.fig.add_subplot(413)


        time_list = self.x
        volume_list = self.df.volume.tolist()

        norm_volume_list = normalize_by_dataset(volume_list, self.y, from_origin=True)

        # Plot the volume graph
        plt.axis('off')
        vol_df = pd.DataFrame(list(zip(time_list, norm_volume_list)), columns=['time', 'volume'])
        vol_df.plot.bar(x='time', y='volume', ax=ax2, label='_nolegend_')



        # Create axis for special technical indicators, obv, macd, etc.
        ax3 = self.fig.add_subplot(414)
        plt.axis('off')

        # Normalize macd
        macd_list = self.df['macd'].tolist()
        norm_macd_list = normalize_by_dataset(macd_list, self.y, from_origin=True)

        # Plot the macd graph
        vol_df = pd.DataFrame(list(zip(time_list, norm_macd_list)),
                              columns=['time', 'macd'])
        vol_df.plot(x='time', y='macd', ax=ax3, label='_nolegend_')

        # Save to an image

        # low dpi and low quality
        plt.savefig(file_name, legend=False, bbox_inches='tight', dpi=45)

        # Resize to 230, 175
        with Image.open(file_name) as img:
            img = img.resize((224, 224), Image.ANTIALIAS)
            img.save(file_name)
            
            # close image
            img.close()
        
        # close all open plots to save memory
        plt.close('all')

    def label_chart(self, csv_path):
        """Show and Label the Chart. Call this method. """
        # Create axis for the price and technical indicator graph
        ax1 = self.fig.add_suplot(211)

        # Turn on axes and background lines
        self.path = csv_path
        plt.axis('on')

        self.df.plot(x=self.col_label, y=self.row_label, ax=ax1, label='price', color='black')

        # Plot Technical Indicators
        if self.tech_inds:
            for col_name in self.tech_inds:
                ti_df = self.df[['time', col_name]].copy()
                ti_df.plot(x='time', y=col_name, ax=ax1, label=col_name)


        # 1 = left mouse, 2 = scroll wheel, 3 = right mouse
        lsso_sell = LassoSelector(ax=self.ax, onselect=self.onSelectSell,
                                  lineprops=self.lineprops_sell, button=1)
        lsso_buy = LassoSelector(ax=self.ax, onselect=self.onSelectBuy,
                                 lineprops=self.lineprops_buy, button=3)
        lsso_clear = LassoSelector(ax=self.ax, onselect=self.onSelectClear,
                                 lineprops=self.lineprops_clear, button=2)

        # On Keyboard Enter press, export the labelled data to .csv
        self.fig.canvas.mpl_connect('key_press_event',
                                    self.enter_press_export_event)

        plt.show()

    def enter_press_export_event(self, event):
        if event.key == "enter":
            self.export_labelled_data(self.path)

    def export_labelled_data(self, path):
        """ Exports the labelled buy/sell/hold signal data and saves it as a
        .csv in path with [time, price, signal] format """
        export_df = pd.DataFrame(columns=['time', 'price', 'signal'])
        export_df.time = self.x
        export_df.price = self.y

        # populate and add the signal_list as a column
        for index, row in export_df.iterrows():
            if row.time in [x[0] for x in self.sell_points]:
                export_df = export_df.set_value(index, 'signal', 'sell')
            elif row.time in [x[0] for x in self.buy_points]:
                export_df = export_df.set_value(index, 'signal', 'buy')
            else:
                export_df = export_df.set_value(index, 'signal', 'hold')

        export_df.to_csv(path, index=False)
        print("Finished Exporting Labelled Data")

    def _add_no_dup(self, to_add, add_to):
        """Adds every element from to_add into add_to, ensuring no duplicated"""
        for x in to_add:
            if x not in add_to:
                add_to.append(x)

    def clear_sell_points(self, points):
        """Removes (x, y) in points currently labelled as sell points. """
        del_points = []
        for sell_point in self.sell_points:
            if sell_point in points:
                del_points.append(sell_point)
        for del_p in del_points:
            self.sell_points.remove(del_p)

        self._recently_deleted_points += del_points

        self.redrawGraph()

    def clear_buy_points(self, points):
        """Removes (x, y) in points currently labelled as buy points. """
        del_points = []
        for buy_point in self.buy_points:
            if buy_point in points:
                del_points.append(buy_point)
        for del_p in del_points:
            self.buy_points.remove(del_p)

        self._recently_deleted_points += del_points

        self.redrawGraph()

    def onSelectBuy(self, x):
        """When lasso selects a buy signal selection. Color graph segment,
        and deal with internal storage models. """
        # path is the internal selection of lassoed points
        path = Path(x)
        filter = path.contains_points(self.points)
        points = list(compress(self.points, filter))

        self._add_no_dup(points, self.buy_points)

        self.clear_sell_points(points)

    def onSelectSell(self, x):
        """When lasso selects a sell signal selection"""
        # path is the internal selection of lassoed points
        path = Path(x)
        filter = path.contains_points(self.points)
        points = list(compress(self.points, filter))

        self._add_no_dup(points, self.sell_points)

        self.clear_buy_points(points)

    def onSelectClear(self, x):
        """When lasso selects a sell signal selection"""
        # path is the internal selection of lassoed points
        path = Path(x)
        filter = path.contains_points(self.points)
        points = list(compress(self.points, filter))

        self.clear_buy_points(points)
        self.clear_sell_points(points)

    def redrawGraph(self):
        """Redraw the graph with new colors"""
        if self.buy_points:
            x_val = [p[0] for p in self.buy_points]
            y_val = [p[1] for p in self.buy_points]
            self.ax.plot(x_val, y_val, 'go', markersize=2)

        if self.sell_points:
            x_val = [p[0] for p in self.sell_points]
            y_val = [p[1] for p in self.sell_points]
            self.ax.plot(x_val, y_val, 'ro', markersize=2)

        if self._recently_deleted_points:
            x_val = [p[0] for p in self._recently_deleted_points]
            y_val = [p[1] for p in self._recently_deleted_points]
            self.ax.plot(x_val, y_val, 'bo', markersize=2)
            self._recently_deleted_points = []

        # Draw the chart
        plt.draw()


def normalize_by_dataset(list1, list2, from_origin=False):
    """Normalize list1 to be in the same interval
    as list2 [0 - max(list2)]"""

    if not from_origin:
        # To scale variable x from dataset X into range [a,b] we use:
        # x_norm = ( (b-a) * ( (x-min(X)) / (max(X)-min(X)) ) + a
        normalized_list1 = []
        for i in range(len(list1)):
            x_norm = ((max(list2) - min(list2)) * (
                        (list1[i] - min(list1)) / (
                            max(list1) - min(
                        list1))) + min(list2))
            normalized_list1.append(x_norm)

        return normalized_list1

    else:
        normalized_list1 = []
        for i in range(len(list1)):
            x_norm = ((max(list2) - 0) * (
                    (list1[i] - min(list1)) / (
                    max(list1) - min(
                list1))) + 0)
            normalized_list1.append(x_norm)

        return normalized_list1
