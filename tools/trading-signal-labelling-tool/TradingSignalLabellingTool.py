import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

import pandas as pd
from itertools import compress

class TradingSignalLabellingTool:
    """ GUI Tool used to display historical financial chart data, and label
    Buy and Sell points within the data with a lasso tool. """

    def __init__(self, df, col_label, row_label, csv_path):
        self.path = csv_path

        self.buy_points = []
        self.sell_points = []

        self.df = df
        self.col_label = col_label
        self.row_label = row_label

        # Convert all points into tuples for easier interaction
        self.x = self.df[col_label].tolist()
        self.y = self.df[row_label].tolist()
        self.points = list(zip(self.x, self.y))

        self.lineprops_sell = {'color': 'red', 'linewidth': 2, 'alpha': 0.8}
        self.lineprops_buy = {'color': 'green', 'linewidth': 2, 'alpha': 0.8}

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

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

        self.redrawGraph()

    def clear_buy_points(self, points):
        """Removes (x, y) in points currently labelled as buy points. """
        del_points = []
        for buy_point in self.buy_points:
            if buy_point in points:
                del_points.append(buy_point)
        for del_p in del_points:
            self.buy_points.remove(del_p)

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


        # Draw the chart
        plt.draw()


    def show_chart(self):
        """Show and Label the Chart. Call this method. """


        self.ax.plot(*zip(*self.points))

        # 1 = left mouse, 2 = scroll wheel, 3 = right mouse
        lsso_sell = LassoSelector(ax=self.ax, onselect=self.onSelectSell,
                                  lineprops=self.lineprops_sell, button=1)
        lsso_buy = LassoSelector(ax=self.ax, onselect=self.onSelectBuy,
                                 lineprops=self.lineprops_buy, button=3)


        # On Keyboard Enter press, export the labelled data to .csv
        self.fig.canvas.mpl_connect('key_press_event', self.enter_press_export_event)


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

        export_df.to_csv(path)
        print("Finished Exporting Labelled Data")


