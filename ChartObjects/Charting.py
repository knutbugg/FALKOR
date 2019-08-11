import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

import pandas as pd
from itertools import compress

from sklearn.preprocessing import StandardScaler

from mpl_finance import candlestick2_ochl

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
        self.df.plot(x=self.col_label, y=self.row_label, ax=ax0, color='black', label='_nolegend_', linewidth=5)

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
        plt.savefig(file_name, legend=False, bbox_inches='tight')

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
