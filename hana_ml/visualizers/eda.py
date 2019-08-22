"""
This module represents an eda plotter. Matplotlib is used for all visualizations.
"""
#pylint: disable=too-many-lines
#pylint: disable=invalid-encoded-data
#pylint: disable=superfluous-parens
#pylint: disable=old-style-class
#pylint: disable=no-init

import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hana_ml import dataframe
from hana_ml.visualizers.visualizer_base import Visualizer
from hana_ml.algorithms.pal import stats

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

if sys.version_info.major == 2:
    #pylint: disable=undefined-variable
    _INTEGER_TYPES = (int, long)
    _STRING_TYPES = (str, unicode)
else:
    _INTEGER_TYPES = (int,)
    _STRING_TYPES = (str,)

class EDAVisualizer(Visualizer): #pylint: disable=invalid-name
    """
    Class for all EDA visualizations, including:
        - Distribution plot
        - Pie plot
        - Correlation plot
        - Scatter plot
        - Bar plot
        - Box plot

    Parameters
    ----------
    ax : matplotlib.Axes, optional
        The axes to use to plot the figure.
        Default value : Current axes
    size : tuple of integers, optional
        (width, height) of the plot in dpi
        Default value: Current size of the plot
    title : str, optional
        This plot's title.
        Default value : Empty str
    """

    def __init__(self, ax=None, size=None, cmap=None, title=None):
        super(EDAVisualizer, self).__init__(ax=ax, size=size, cmap=cmap, title=title)

    def distribution_plot(self, data, column, bins, title=None, x_axis_fontsize=10, #pylint: disable= too-many-locals, too-many-arguments
                          x_axis_rotation=90, debrief=True):
        """
        Returns a distribution plot for the HANA DataFrame column specified.

        Parameters
        ----------
        data : DataFrame
            DataFrame to use for the plot.
        column : str
            Column in the DataFrame being plotted.
        bins : int
            Number of bins to create based on the value of column.
        title : str, optional
            Title for the plot.
        x_axis_fontsize : int, optional
            Size of x axis labels
        x_axis_rotation : int, optional
            Rotation of x axis labels
        debrief : bool, optional
            Include skewness debrief

        Returns
        -------
        ax : Axes
            The axes for the plot.
        bin_data : pandas.DataFrame
            The data used in the plot.
        """
        conn_context = data.connection_context
        query = "SELECT MAX({}) FROM ({})".format(column, data.select_statement)
        maxi = conn_context.sql(query).collect().values[0][0]
        query = "SELECT MIN({}) FROM ({})".format(column, data.select_statement)
        mini = conn_context.sql(query).collect().values[0][0]
        diff = maxi-mini
        bins = bins
        bin_size = round(diff/bins)
        if bin_size < 1:
            bin_size = 1
        query = "SELECT {0}, FLOOR({0}/{1}) AS BAND,".format(column, bin_size)
        query += " '[' || FLOOR({0}/{1})*{1} || ', ".format(column, bin_size)
        query += "' || ((FLOOR({0}/{1})*{1})+{1}) || ')'".format(column, bin_size)
        query += " AS BANDING FROM ({}) ORDER BY BAND ASC".format(data.select_statement)
        bin_data = conn_context.sql(query)
        bin_data = bin_data.agg([('count', column, 'COUNT'),
                                 ('avg', 'BAND', 'ORDER')], group_by='BANDING')
        bin_data = bin_data.sort('ORDER').collect()
        ax = self.ax  #pylint: disable=invalid-name
        ax.bar(bin_data['BANDING'], bin_data['COUNT'])
        for item in [ax.xaxis.label] + ax.get_xticklabels():
            item.set_fontsize(x_axis_fontsize)
        ax.xaxis.set_tick_params(rotation=x_axis_rotation)
        if title is not None:
            ax.set_title(title)
        ax.grid(which="major", axis="y", color='black', linestyle='-', linewidth=1, alpha=0.2)
        ax.set_axisbelow(True)
        # Turn spines off
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if debrief:
            query = "SELECT (A.RX3 - 3*A.RX2*A.AV + 3*A.RX*A.AV*A.AV - "
            query += "A.RN*A.AV*A.AV*A.AV) / (A.STDV*A.STDV*A.STDV) * A.RN /"
            query += " (A.RN-1) / (A.RN-2) AS SKEWNESS FROM (SELECT SUM(1.0*{})".format(column)
            query += " AS RX, SUM(POWER(1.0*{},2)) AS RX2, ".format(column)
            query += "SUM(POWER(1.0*{},3))".format(column)
            query += " AS RX3, COUNT(1.0*{0}) AS RN, STDDEV(1.0*{0}) AS STDV,".format(column)
            query += " AVG(1.0*{0}) AS AV FROM ({1})) A".format(column, data.select_statement)
            # Calculate skewness
            skewness = conn_context.sql(query)
            skewness = skewness.collect()['SKEWNESS'].values[0]
            ax.text(max(bin_data.index)*0.9, max(bin_data['COUNT'].values)*0.95,
                    'Skewness: {:.2f}'.format(skewness), style='italic',
                    bbox={'facecolor':'white', 'pad':0.65, 'boxstyle':'round'})
        else:
            pass
        # Turn off y ticks
        ax.yaxis.set_ticks_position('none')
        return ax, bin_data

    def pie_plot(self, data, column, explode=0.03, title=None, legend=True): #pylint: disable=too-many-arguments
        """
        Returns a pie plot for the HANA DataFrame column specified.

        Parameters
        ----------
        data : DataFrame
            DataFrame to use for the plot.
        column : str
            Column in the DataFrame being plotted.
        explode : float, optional
            Relative spacing between pie segments.
        title : str, optional
            Title for the plot.
        legend : bool, optional
            Legend for the plot.

        Returns
        -------
        ax : Axes
            The axes for the plot.
            This can be used to set specific properties for the plot.
        pie_data : pandas.DataFrame
            The data used in the plot.
        """
        data = data.agg([('count', column, 'COUNT')], group_by=column).sort(column)
        pie_data = data.collect()
        explode = (explode,)*len(pie_data)
        ax = self.ax  #pylint: disable=invalid-name
        ax.pie(x=pie_data['COUNT'], explode=explode, labels=pie_data[column],
               autopct='%1.1f%%', shadow=True)
        if legend:
            ax.legend(pie_data[column], loc='best', edgecolor='w')
        else:
            pass
        if title is not None:
            ax.set_title(title)
        return ax, pie_data

    def correlation_plot(self, data, corr_cols=None, label=True, cmap="RdYlBu"): #pylint: disable=too-many-locals
        """
        Returns a correlation plot for the HANA DataFrame columns specified.

        Parameters
        ----------
        data : DataFrame
            DataFrame to use for the plot.
        corr_cols : list of str, optional
            Columns in the DataFrame being plotted. If None all numeric columns will be plotted.
        label : bool, optional
            Plot a colorbar.
        cmap : matplotlib.pyplot.colormap, optional
            Color map to use for the plot.

        Returns
        -------
        ax : Axes
            The axes for the plot.
            This can be used to set specific properties for the plot.
        corr : pandas.DataFrame
            The data used in the plot.
        """
        conn_context = data.connection_context
        if not isinstance(data, dataframe.DataFrame):
            raise TypeError('Parameter data must be a DataFrame')
        if corr_cols is None:
            cols = data.columns
        else:
            cols = corr_cols
        message = 'Parameter corr_cols must be a string or a list of strings'
        if isinstance(cols, _STRING_TYPES):
            cols = [cols]
        if (not cols or not isinstance(cols, list) or
                not all(isinstance(col, _STRING_TYPES) for col in cols)):
            raise TypeError(message)
        # Get only the numerics
        corr_cols = [col for col in cols if data.is_numeric(col)]
        # Better have 2 columns
        if len(corr_cols) < 2:
            raise ValueError('Must have at least 2 correlation columns that are numeric')
        if data.hasna():
            numeric = [i for i in data.columns if data.is_numeric(i)]
            data_wo_na = data.dropna(subset=numeric)
            corr = stats.pearsonr_matrix(conn_context=conn_context, data=data_wo_na,
                                         cols=corr_cols).collect()
        else:
            corr = stats.pearsonr_matrix(conn_context=conn_context, data=data,
                                         cols=corr_cols).collect()
        corr = corr.set_index(list(corr.columns[[0]]))
        ax = self.ax  #pylint: disable=invalid-name
        cp = ax.matshow(corr, cmap=cmap) #pylint: disable=invalid-name
        for (i, j), z in np.ndenumerate(corr): #pylint: disable=invalid-name
            ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center')
        ticks = np.arange(0, len(corr_cols), 1)
        ax.set_xticks(ticks)
        ax.xaxis.set_tick_params(rotation=90)
        ax.set_yticks(ticks)
        ax.set_xticklabels(corr_cols)
        ax.set_yticklabels(corr_cols)
        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items(): #pylint: disable=unused-variable
            spine.set_visible(False)
        ticks = np.arange(0, len(corr_cols), 1)-0.5
        ax.set_xticks(ticks, minor=True)
        ax.set_yticks(ticks, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
        ax.tick_params(which="minor", bottom=False, right=False, left=False, top=False)
        ax.tick_params(which="both", bottom=False, right=False)
        if label:
            cb = ax.get_figure().colorbar(cp, ax=ax) #pylint: disable=invalid-name
            cb.set_clim(-1, 1)
            cb.set_label("Pearson's correlation (r)")
        return ax, corr

    def scatter_plot(self, data, x, y, x_bins, y_bins, title=None, label=True, #pylint: disable=too-many-locals, too-many-arguments, too-many-statements, invalid-name
                     cmap="Blues", debrief=True):
        """
        Returns a scatter plot for the HANA DataFrame columns specified.

        Parameters
        ----------
        data : DataFrame
            DataFrame to use for the plot.
        x : str
            Column to be plotted on the x axis.
        y : str
            Column to be plotted on the y axis.
        x_bins : int
            Number of x axis bins to create based on the value of column.
        y_bins : int
            Number of y axis bins to create based on the value of column.
        title : str, optional
            Title for the plot.
        label : str, optional
            Label for the color bar.
        cmap : matplotlib.pyplot.colormap, optional
            Color map to use for the plot.
        debrief : bool, optional
            Include correlation debrief.

        Returns
        -------
        ax : Axes
            The axes for the plot.
        bin_matrix : pandas.DataFrame
            The data used in the plot.
        """
        conn_context = data.connection_context
        x_max = "SELECT MAX({}) FROM ({})".format(x, data.select_statement)
        x_maxi = conn_context.sql(x_max).collect().values[0][0]
        x_min = "SELECT MIN({}) FROM ({})".format(x, data.select_statement)
        x_mini = conn_context.sql(x_min).collect().values[0][0]
        x_diff = x_maxi-x_mini
        x_bins = x_bins
        x_bin_size = round(x_diff/x_bins)
        y_max = "SELECT MAX({}) FROM ({})".format(y, data.select_statement)
        y_maxi = conn_context.sql(y_max).collect().values[0][0]
        y_min = "SELECT MIN({}) FROM ({})".format(y, data.select_statement)
        y_mini = conn_context.sql(y_min).collect().values[0][0]
        y_diff = y_maxi-y_mini
        y_bins = y_bins
        y_bin_size = round(y_diff/y_bins)
        query = "SELECT *, FLOOR({0}/{1}) AS BAND_X,".format(x, x_bin_size)
        query += " '[' || FLOOR({0}/{1})*{1} || ', ' || ".format(x, x_bin_size)
        query += "((FLOOR({0}/{1})*{1})".format(x, x_bin_size)
        query += "+{0}) || ')' AS BANDING_X,".format(x_bin_size)
        query += " FLOOR({0}/{1}) AS BAND_Y, '[' || ".format(y, y_bin_size)
        query += "FLOOR({0}/{1})*{1} || ', ' || ".format(y, y_bin_size)
        query += "((FLOOR({0}/{1})*{1})+{1}) || ')'".format(y, y_bin_size)
        query += " AS BANDING_Y FROM ({})".format(data.select_statement)
        bin_data = conn_context.sql(query)
        bin_data = bin_data.agg([('count', x, 'COUNT'), ('avg', 'BAND_X', 'ORDER_X'),
                                 ('avg', 'BAND_Y', 'ORDER_Y')], group_by=['BANDING_X',
                                                                          'BANDING_Y']).collect()
        bin_matrix = pd.crosstab(bin_data['BANDING_Y'],
                                 bin_data['BANDING_X'],
                                 values=bin_data['COUNT'],
                                 aggfunc='sum').sort_index(ascending=False)
        x_order = bin_data.sort_values(by="ORDER_X")['BANDING_X'].unique()
        y_order = bin_data.sort_values(by="ORDER_Y", ascending=False)['BANDING_Y'].unique()
        bin_matrix = bin_matrix[x_order]
        bin_matrix = bin_matrix.reindex(y_order)
        ax = self.ax  #pylint: disable=invalid-name
        cp = ax.imshow(bin_matrix, cmap=cmap) #pylint: disable=invalid-name
        ax.set_xticks(np.arange(len(bin_matrix.columns)))
        ax.set_xticklabels(bin_matrix.columns)
        ax.set_xlabel(x, fontdict={'fontsize':12})
        ax.xaxis.set_tick_params(rotation=90)
        ax.set_yticks(np.arange(len(bin_matrix.index)))
        ax.set_yticklabels(bin_matrix.index)
        ax.set_ylabel(y, fontdict={'fontsize':12})
        # Turn spines off and create white grid.
        for edge, spine in ax.spines.items(): #pylint: disable=unused-variable
            spine.set_visible(False)
        ticks = np.arange(0, len(bin_matrix.columns), 1)-0.5
        ax.set_xticks(ticks, minor=True)
        ticks = np.arange(0, len(bin_matrix.index), 1)-0.5
        ax.set_yticks(ticks, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, right=False, left=False, top=False)
        ax.tick_params(which="both", top=False, right=False)
        if debrief:
            # Calculate correlation
            corr = data.corr(x, y).collect().values[0][0]
            ax.text(0, 0, 'Correlation: {:.2f}'.format(corr), style='italic',
                    bbox={'facecolor':'white', 'pad':0.65, 'boxstyle':'round'})
        else:
            pass
        if title is not None:
            ax.set_title(title)
        if label:
            cb = ax.get_figure().colorbar(cp, ax=ax) #pylint: disable=invalid-name
            cb.set_label('Count')
        return ax, bin_matrix

    def bar_plot(self, data, column, aggregation, title=None): #pylint: disable=too-many-branches, too-many-statements
        """
        Returns a bar plot for the HANA DataFrame column specified.

        Parameters
        ----------
        data : DataFrame
            DataFrame to use for the plot.
        column : str
            Column to be aggregated.
        aggregation : dict
            Aggregation conditions ('avg', 'count', 'max', 'min').
        title : str, optional
            Title for the plot.

        Returns
        -------
        ax : Axes
            The axes for the plot.
        bar_data : pandas.DataFrame
            The data used in the plot.

        Examples
        --------
        >>> ax1 = f.add_subplot(111)
        >>> eda = EDAVisualizer(ax1)
        >>> ax, bar_data = eda.bar_plot(data=data, column='COLUMN',
                                        aggregation={'COLUMN':'count'})

        Returns : bar plot (count) of 'COLUMN'

        >>> ax1 = f.add_subplot(111)
        >>> eda = EDAVisualizer(ax1)
        >>> ax, bar_data = eda.bar_plot(data=data, column='COLUMN',
                                        aggregation={'OTHER_COLUMN':'avg'})

        Returns : bar plot (avg) of 'COLUMN' against 'OTHER_COLUMN'
        """
        if list(aggregation.values())[0] == 'count':
            data = data.agg([('count', column, 'COUNT')], group_by=column).sort(column)
            bar_data = data.collect()
            if len(bar_data.index) <= 20:
                ax = self.ax #pylint: disable=invalid-name
                ax.barh(bar_data[column].values.astype(str), bar_data['COUNT'].values)
                for item in [ax.xaxis.label] + ax.get_xticklabels():
                    item.set_fontsize(10)
                ax.set_ylabel(column, fontdict={'fontsize':12})
                ax.set_xlabel('COUNT', fontdict={'fontsize':12})
                if title is not None:
                    ax.set_title(title)
                ax.grid(which="major", axis="x", color='black', linestyle='-',
                        linewidth=1, alpha=0.2)
                ax.set_axisbelow(True)
                # Turn spines off
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Turn off y ticks
                ax.xaxis.set_ticks_position('none')
            else:
                ax = self.ax #pylint: disable=invalid-name
                ax.bar(bar_data[column].values.astype(str), bar_data['COUNT'].values)
                for item in [ax.xaxis.label] + ax.get_xticklabels():
                    item.set_fontsize(10)
                ax.set_xlabel(column, fontdict={'fontsize':12})
                ax.set_ylabel('COUNT', fontdict={'fontsize':12})
                if title is not None:
                    ax.set_title(title)
                ax.grid(which="major", axis="y", color='black', linestyle='-',
                        linewidth=1, alpha=0.2)
                ax.set_axisbelow(True)
                # Turn spines off
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Turn off y ticks
                ax.yaxis.set_ticks_position('none')
        elif list(aggregation.values())[0] == 'avg':
            data = data.agg([('avg', list(aggregation.keys())[0], 'AVG')],
                            group_by=column).sort(column)
            bar_data = data.collect()
            if len(bar_data.index) <= 20:
                ax = self.ax #pylint: disable=invalid-name
                ax.barh(bar_data[column].values.astype(str), bar_data['AVG'].values)
                for item in [ax.xaxis.label] + ax.get_xticklabels():
                    item.set_fontsize(10)
                ax.set_ylabel(column, fontdict={'fontsize':12})
                ax.set_xlabel('Average '+list(aggregation.keys())[0],
                              fontdict={'fontsize':12})
                if title is not None:
                    ax.set_title(title)
                ax.grid(which="major", axis="x", color='black', linestyle='-',
                        linewidth=1, alpha=0.2)
                ax.set_axisbelow(True)
                # Turn spines off
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Turn off y ticks
                ax.xaxis.set_ticks_position('none')
            else:
                ax = self.ax #pylint: disable=invalid-name
                ax.bar(bar_data[column].values.astype(str), bar_data['AVG'].values)
                for item in [ax.xaxis.label] + ax.get_xticklabels():
                    item.set_fontsize(10)
                ax.set_xlabel(column, fontdict={'fontsize':12})
                ax.set_ylabel('Average '+list(aggregation.keys())[0],
                              fontdict={'fontsize':12})
                if title is not None:
                    ax.set_title(title)
                ax.grid(which="major", axis="y", color='black', linestyle='-',
                        linewidth=1, alpha=0.2)
                ax.set_axisbelow(True)
                # Turn spines off
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Turn off y ticks
                ax.yaxis.set_ticks_position('none')
        elif list(aggregation.values())[0] == 'min':
            data = data.agg([('min', list(aggregation.keys())[0], 'MIN')],
                            group_by=column).sort(column)
            bar_data = data.collect()
            if len(bar_data.index) <= 20:
                ax = self.ax #pylint: disable=invalid-name
                ax.barh(bar_data[column].values.astype(str), bar_data['MIN'].values)
                for item in [ax.xaxis.label] + ax.get_xticklabels():
                    item.set_fontsize(10)
                ax.set_ylabel(column, fontdict={'fontsize':12})
                ax.set_xlabel('Min '+list(aggregation.keys())[0], fontdict={'fontsize':12})
                if title is not None:
                    ax.set_title(title)
                ax.grid(which="major", axis="x", color='black', linestyle='-',
                        linewidth=1, alpha=0.2)
                ax.set_axisbelow(True)
                # Turn spines off
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Turn off y ticks
                ax.xaxis.set_ticks_position('none')
            else:
                ax = self.ax #pylint: disable=invalid-name
                ax.bar(bar_data[column].values.astype(str), bar_data['MIN'].values)
                for item in [ax.xaxis.label] + ax.get_xticklabels():
                    item.set_fontsize(10)
                ax.set_xlabel(column, fontdict={'fontsize':12})
                ax.set_ylabel('Min '+list(aggregation.keys())[0], fontdict={'fontsize':12})
                if title is not None:
                    ax.set_title(title)
                ax.grid(which="major", axis="y", color='black', linestyle='-',
                        linewidth=1, alpha=0.2)
                ax.set_axisbelow(True)
                # Turn spines off
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Turn off y ticks
                ax.yaxis.set_ticks_position('none')
        elif list(aggregation.values())[0] == 'max':
            data = data.agg([('max', list(aggregation.keys())[0], 'MAX')],
                            group_by=column).sort(column)
            bar_data = data.collect()
            if len(bar_data.index) <= 20:
                ax = self.ax #pylint: disable=invalid-name
                ax.barh(bar_data[column].values.astype(str), bar_data['MAX'].values)
                for item in [ax.xaxis.label] + ax.get_xticklabels():
                    item.set_fontsize(10)
                ax.set_ylabel(column, fontdict={'fontsize':12})
                ax.set_xlabel('Max '+list(aggregation.keys())[0], fontdict={'fontsize':12})
                if title is not None:
                    ax.set_title(title)
                ax.grid(which="major", axis="x", color='black', linestyle='-',
                        linewidth=1, alpha=0.2)
                ax.set_axisbelow(True)
                # Turn spines off
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Turn off y ticks
                ax.xaxis.set_ticks_position('none')
            else:
                ax = self.ax #pylint: disable=invalid-name
                ax.bar(bar_data[column].values.astype(str), bar_data['MAX'].values)
                #for item in ([ax.xaxis.label] + ax.get_xticklabels()):
                for item in [ax.xaxis.label] + ax.get_xticklabels():
                    item.set_fontsize(10)
                ax.set_xlabel(column, fontdict={'fontsize':12})
                ax.set_ylabel('Max '+list(aggregation.keys())[0], fontdict={'fontsize':12})
                if title is not None:
                    ax.set_title(title)
                ax.grid(which="major", axis="y", color='black', linestyle='-',
                        linewidth=1, alpha=0.2)
                ax.set_axisbelow(True)
                # Turn spines off
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                # Turn off y ticks
                ax.yaxis.set_ticks_position('none')
        return ax, bar_data

    def box_plot(self, data, column, outliers=False, title=None, groupby=None): #pylint: disable=too-many-locals, too-many-arguments, too-many-branches, too-many-statements
        """
        Returns a box plot for the HANA DataFrame column specified.

        Parameters
        ----------
        data : DataFrame
            DataFrame to use for the plot.
        column : str
            Column in the DataFrame being plotted.
        outliers : bool
            Whether to plot suspected outliers and outliers.
        title : str, optional
            Title for the plot.
        groupby : str, optional
            Column to group by and compare.

        Returns
        -------
        ax : Axes
            The axes for the plot.
        cont : pandas.DataFrame
            The data used in the plot.
        """
        conn_context = data.connection_context
        if not outliers and groupby is None:
            cont, cat = stats.univariate_analysis(conn_context=conn_context, #pylint: disable=unused-variable, invalid-name
                                                  data=data, cols=[column])
            median = cont.collect()['STAT_VALUE']
            median = median.loc[cont.collect()['STAT_NAME'] == 'median']
            median = median.values[0]
            mini = cont.collect()['STAT_VALUE'].loc[cont.collect()['STAT_NAME'] == 'min'].values[0]
            maxi = cont.collect()['STAT_VALUE'].loc[cont.collect()['STAT_NAME'] == 'max'].values[0]
            lq = cont.collect()['STAT_VALUE'] #pylint: disable=invalid-name, unused-variable
            lq = lq.loc[cont.collect()['STAT_NAME'] == 'lower quartile'] #pylint: disable=invalid-name
            lq = lq.values[0] #pylint: disable=invalid-name
            uq = cont.collect()['STAT_VALUE'] #pylint: disable=invalid-name
            uq = uq.loc[cont.collect()['STAT_NAME'] == 'upper quartile'] #pylint: disable=invalid-name
            uq = uq.values[0] #pylint: disable=invalid-name
            # Create axis
            ax = self.ax  #pylint: disable=invalid-name
            ax.set_yticks(np.arange(0, 1, 1)+0.5)
            ax.set_yticklabels([column])
            # Add verticle lines
            ax.axvline(x=mini, ymin=0.4, ymax=0.6, color='black')
            ax.axvline(x=median, ymin=0.33, ymax=0.67, color='black',
                       linestyle='--', linewidth=2, label='Median')
            ax.axvline(x=maxi, ymin=0.4, ymax=0.6, color='black')
            # Add horizontal lines
            ax.hlines(y=0.5, xmin=mini, xmax=lq)
            ax.hlines(y=0.5, xmin=uq, xmax=maxi)
            # Add box
            ax.axvspan(xmin=lq, xmax=uq, ymin=0.35, ymax=0.65)
            # Add legend
            ax.legend(loc='best', edgecolor='w')
            # Turn spines off
            for i in ['top', 'bottom', 'right', 'left']:
                ax.spines[i].set_visible(False)
            # Add gridlines
            ax.grid(which="major", axis="x", color='black', linestyle='-',
                    linewidth=1, alpha=0.2)
            ax.set_axisbelow(True)
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            if title is not None:
                ax.set_title(title)
        elif not outliers and isinstance(groupby, str):
            query = "SELECT DISTINCT({}) FROM ({})".format(groupby, data.select_statement)
            values = conn_context.sql(query).collect().values
            values = [i[0] for i in values]
            tick_values = int(len(values)+1)
            ax = self.ax  #pylint: disable=invalid-name
            ax.set_yticks(np.arange(0, 1, 1/tick_values)+1/tick_values)
            ax.set_yticklabels(values)
            median = []
            mini = []
            maxi = []
            lq = [] #pylint: disable=invalid-name
            uq = [] #pylint: disable=invalid-name
            for i in values:
                data_groupby = data.filter("{} = '{}'".format(groupby, i))
                cont, cat = stats.univariate_analysis(conn_context=conn_context,
                                                      data=data_groupby, cols=[column])
                median.append(cont.collect()['STAT_VALUE'].
                              loc[cont.collect()['STAT_NAME'] == 'median'].values[0])
                mini.append(cont.collect()['STAT_VALUE'].
                            loc[cont.collect()['STAT_NAME'] == 'min'].values[0])
                maxi.append(cont.collect()['STAT_VALUE'].
                            loc[cont.collect()['STAT_NAME'] == 'max'].values[0])
                lq.append(cont.collect()['STAT_VALUE'].
                          loc[cont.collect()['STAT_NAME'] == 'lower quartile'].values[0])
                uq.append(cont.collect()['STAT_VALUE'].
                          loc[cont.collect()['STAT_NAME'] == 'upper quartile'].values[0])
            # Plot vertical lines
            n = 0 #pylint: disable=invalid-name
            m = 1 #pylint: disable=invalid-name
            height = (1/len(values))/4
            while n < len(values):
                if m == 1:
                    # Plot min
                    ax.axvline(x=float(mini[n]), ymin=(m/tick_values)-(height*0.5),
                               ymax=(m/tick_values)+(height*0.5), color='black')
                    # Plot max
                    ax.axvline(x=float(maxi[n]), ymin=(m/tick_values)-(height*0.5),
                               ymax=(m/tick_values)+(height*0.5), color='black')
                    # Plot median
                    ax.axvline(x=float(median[n]), ymin=(m/tick_values)-height,
                               ymax=(m/tick_values)+height, color='black',
                               linestyle='--', linewidth=2, label='Median')
                else:
                    # Plot min
                    ax.axvline(x=float(mini[n]), ymin=(m/tick_values)-(height*0.5),
                               ymax=(m/tick_values)+(height*0.5), color='black')
                    # Plot max
                    ax.axvline(x=float(maxi[n]), ymin=(m/tick_values)-(height*0.5),
                               ymax=(m/tick_values)+(height*0.5), color='black')
                    # Plot median
                    ax.axvline(x=float(median[n]), ymin=(m/tick_values)-height,
                               ymax=(m/tick_values)+height, color='black',
                               linestyle='--', linewidth=2)
                n += 1 #pylint: disable=invalid-name
                m += 1 #pylint: disable=invalid-name
            # Plot horizontal lines
            ax.set_ylim([0, 1])
            n = 0 #pylint: disable=invalid-name
            m = 1 #pylint: disable=invalid-name
            while n < len(values):
                # Add horizontal line
                ax.hlines(y=m/tick_values, xmin=mini[n], xmax=lq[n])
                # Add horizontal line
                ax.hlines(y=m/tick_values, xmin=uq[n], xmax=maxi[n])
                # Add box
                ax.axvspan(xmin=lq[n], xmax=uq[n], ymin=(m/tick_values)-(height*0.75),
                           ymax=(m/tick_values)+(height*0.75))
                n += 1 #pylint: disable=invalid-name
                m += 1 #pylint: disable=invalid-name
            # Add legend
            ax.legend(loc='best', edgecolor='w')
            # Turn spines off
            for i in ['top', 'bottom', 'right', 'left']:
                ax.spines[i].set_visible(False)
            # Add gridlines
            ax.grid(which="major", axis="x", color='black', linestyle='-',
                    linewidth=1, alpha=0.2)
            ax.set_axisbelow(True)
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            if title is not None:
                ax.set_title(title)
        elif outliers and groupby is None:
            cont, cat = stats.univariate_analysis(conn_context=conn_context,
                                                  data=data, cols=[column])
            median = cont.collect()['STAT_VALUE']
            median = median.loc[cont.collect()['STAT_NAME'] == 'median'].values[0]
            mini = cont.collect()['STAT_VALUE']
            mini = mini.loc[cont.collect()['STAT_NAME'] == 'min'].values[0]
            maxi = cont.collect()['STAT_VALUE']
            maxi = maxi.loc[cont.collect()['STAT_NAME'] == 'max'].values[0]
            lq = cont.collect()['STAT_VALUE'] #pylint: disable=invalid-name, unused-variable
            lq = lq.loc[cont.collect()['STAT_NAME'] == 'lower quartile'].values[0] #pylint: disable=invalid-name
            uq = cont.collect()['STAT_VALUE'] #pylint: disable=invalid-name
            uq = uq.loc[cont.collect()['STAT_NAME'] == 'upper quartile'].values[0] #pylint: disable=invalid-name
            iqr = uq-lq
            suspected_upper_outlier_fence = uq+(1.5*iqr)
            suspected_lower_outlier_fence = lq-(1.5*iqr)
            upper_outlier_fence = uq+(3*iqr)
            lower_outlier_fence = lq-(3*iqr)
            # Create axis
            ax = self.ax #pylint: disable=invalid-name
            ax.set_yticks(np.arange(0, 1, 1)+0.5)
            ax.set_yticklabels([column])
            # Add vertical lines
            ax.axvline(x=lower_outlier_fence, ymin=0.4, ymax=0.6,
                       color='black', linestyle=':', label='Outlier fence')
            ax.axvline(x=suspected_lower_outlier_fence, ymin=0.4, ymax=0.6,
                       color='black', label='Suspected outlier fence')
            ax.axvline(x=median, ymin=0.33, ymax=0.67, color='black',
                       linewidth=2, linestyle='--', label='Median')
            ax.axvline(x=suspected_upper_outlier_fence, ymin=0.4, ymax=0.6,
                       color='black')
            ax.axvline(x=upper_outlier_fence, ymin=0.4, ymax=0.6, color='black',
                       linestyle=':')
            # Add horizontal lines
            ax.hlines(y=0.5, xmin=suspected_lower_outlier_fence, xmax=lq)
            ax.hlines(y=0.5, xmin=uq, xmax=suspected_upper_outlier_fence)
            # Add box
            ax.axvspan(xmin=lq, xmax=uq, ymin=0.35, ymax=0.65)
            # Fetch and plot suspected outliers and true outliers
            query = "SELECT DISTINCT({}) FROM ({})".format(column, data.select_statement)
            query += " WHERE {} > {} ".format(column, suspected_upper_outlier_fence)
            query += "OR {} < {}".format(column, suspected_lower_outlier_fence)
            suspected_outliers = conn_context.sql(query)
            n = 0 #pylint: disable=invalid-name
            for i in suspected_outliers.collect().values:
                if n == 0:
                    ax.plot(i, 0.5, 'o', color='grey', markersize=5, alpha=0.3,
                            label='Suspected outlier')
                    n += 1 #pylint: disable=invalid-name
                else:
                    ax.plot(i, 0.5, 'o', color='grey', markersize=5, alpha=0.3)
            query = "SELECT DISTINCT({}) FROM ".format(column)
            query += "({}) WHERE {} > ".format(data.select_statement, column)
            query += "{} OR {} < {}".format(upper_outlier_fence, column, lower_outlier_fence)
            outliers = conn_context.sql(query)
            n = 0 #pylint: disable=invalid-name
            for i in outliers.collect().values:
                if n == 0:
                    ax.plot(i, 0.5, 'o', color='red', markersize=5, alpha=0.3,
                            label='Outlier')
                    n += 1 #pylint: disable=invalid-name
                else:
                    ax.plot(i, 0.5, 'o', color='red', markersize=5, alpha=0.3)
            # Add legend
            ax.legend(loc='best', edgecolor='w')
            # Turn spines off
            for i in ['top', 'bottom', 'right', 'left']:
                ax.spines[i].set_visible(False)
            # Add gridlines
            ax.grid(which="major", axis="x", color='black', linestyle='-',
                    linewidth=1, alpha=0.2)
            ax.set_axisbelow(True)
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            if title is not None:
                ax.set_title(title)
        elif outliers and isinstance(groupby, str):
            query = "SELECT DISTINCT({}) FROM ({})".format(groupby, data.select_statement)
            values = conn_context.sql(query).collect().values
            values = [i[0] for i in values]
            tick_values = int(len(values)+1)
            ax = self.ax #pylint: disable=invalid-name
            ax.set_yticks(np.arange(0, 1, 1/tick_values)+1/tick_values)
            ax.set_yticklabels(values)
            median = []
            mini = []
            maxi = []
            lq = [] #pylint: disable=invalid-name
            uq = [] #pylint: disable=invalid-name
            iqr = []
            suspected_upper_outlier_fence = []
            suspected_lower_outlier_fence = []
            upper_outlier_fence = []
            lower_outlier_fence = []
            suspected_outliers = []
            outliers = []
            for i in values:
                data_groupby = data.filter("{} = '{}'".format(groupby, i))
                # Get statistics
                cont, cat = stats.univariate_analysis(conn_context=conn_context,
                                                      data=data_groupby, cols=[column])
                median_val = cont.collect()['STAT_VALUE']
                median_val = median_val.loc[cont.collect()['STAT_NAME'] == 'median']
                median_val = median_val.values[0]
                median.append(median_val)
                minimum = cont.collect()['STAT_VALUE']
                minimum = minimum.loc[cont.collect()['STAT_NAME'] == 'min']
                minimum = minimum.values[0]
                mini.append(minimum)
                maximum = cont.collect()['STAT_VALUE']
                maximum = maximum.loc[cont.collect()['STAT_NAME'] == 'max']
                maximum = maximum.values[0]
                maxi.append(maximum)
                low_quart = cont.collect()['STAT_VALUE']
                low_quart = low_quart.loc[cont.collect()['STAT_NAME'] == 'lower quartile']
                low_quart = low_quart.values[0]
                lq.append(low_quart)
                upp_quart = cont.collect()['STAT_VALUE']
                upp_quart = upp_quart.loc[cont.collect()['STAT_NAME'] == 'upper quartile']
                upp_quart = upp_quart.values[0]
                uq.append(upp_quart)
                int_quart_range = upp_quart-low_quart
                iqr.append(int_quart_range)
                sus_upp_out_fence = upp_quart+(1.5*int_quart_range)
                suspected_upper_outlier_fence.append(sus_upp_out_fence)
                sus_low_out_fence = low_quart-(1.5*int_quart_range)
                suspected_lower_outlier_fence.append(sus_low_out_fence)
                upp_out_fence = upp_quart+(3*int_quart_range)
                upper_outlier_fence.append(upp_out_fence)
                low_out_fence = low_quart-(3*int_quart_range)
                lower_outlier_fence.append(low_out_fence)
                # Fetch and plot suspected outliers and true outliers
                query = "SELECT DISTINCT({}) FROM ({}) ".format(column,
                                                                data_groupby.select_statement)
                query += "WHERE {} > {} ".format(column, sus_upp_out_fence)
                query += "OR {} < {}".format(column, sus_low_out_fence)
                suspected_outliers.append(list(conn_context.sql(query).collect().values))
                query = "SELECT DISTINCT({}) FROM ({}) ".format(column,
                                                                data_groupby.select_statement)
                query += "WHERE {} > {} ".format(column, upp_out_fence)
                query += "OR {} < {}".format(column, low_out_fence)
                outliers.append(list(conn_context.sql(query).collect().values))
            n = 0 #pylint: disable=invalid-name
            m = 1 #pylint: disable=invalid-name
            height = (1/len(values))/4
            while n < len(values):
                # Plot vertical lines
                if m == 1:
                    ax.axvline(x=float(median[n]),
                               ymin=(m/tick_values)-(height),
                               ymax=(m/tick_values)+(height),
                               color='black', linestyle='--',
                               linewidth=2, label='Median')
                    ax.axvline(x=float(lower_outlier_fence[n]),
                               ymin=(m/tick_values)-(height*0.5),
                               ymax=(m/tick_values)+(height*0.5),
                               color='black',
                               linestyle=':', label='Outlier fence')
                    ax.axvline(x=float(suspected_lower_outlier_fence[n]),
                               ymin=(m/tick_values)-(height*0.5),
                               ymax=(m/tick_values)+(height*0.5), color='black',
                               linestyle='-', label='Suspected outlier fence')
                else:
                    ax.axvline(x=float(median[n]), ymin=(m/tick_values)-(height),
                               ymax=(m/tick_values)+(height),
                               color='black', linestyle='--', linewidth=2)
                    ax.axvline(x=float(lower_outlier_fence[n]),
                               ymin=(m/tick_values)-(height*0.5),
                               ymax=(m/tick_values)+(height*0.5),
                               color='black', linestyle=':')
                    ax.axvline(x=float(suspected_lower_outlier_fence[n]),
                               ymin=(m/tick_values)-(height*0.5),
                               ymax=(m/tick_values)+(height*0.5),
                               color='black', linestyle='-')
                ax.axvline(x=float(suspected_upper_outlier_fence[n]),
                           ymin=(m/tick_values)-(height*0.5),
                           ymax=(m/tick_values)+(height*0.5),
                           color='black', linestyle='-')
                ax.axvline(x=float(upper_outlier_fence[n]),
                           ymin=(m/tick_values)-(height*0.5),
                           ymax=(m/tick_values)+(height*0.5),
                           color='black', linestyle=':')
                n += 1 #pylint: disable=invalid-name
                m += 1 #pylint: disable=invalid-name
            n = 0 #pylint: disable=invalid-name
            m = 1 #pylint: disable=invalid-name
            ax.set_ylim([0, 1])
            while n < len(values):
                # Plot horizontal lines
                ax.hlines(y=m/tick_values, xmin=suspected_lower_outlier_fence[n], xmax=lq[n])
                ax.hlines(y=m/tick_values, xmin=uq[n], xmax=suspected_upper_outlier_fence[n])
                # Add box
                ax.axvspan(xmin=lq[n], xmax=uq[n], ymin=(m/tick_values)-(height*0.75),
                           ymax=(m/tick_values)+(height*0.75))
                n += 1 #pylint: disable=invalid-name
                m += 1 #pylint: disable=invalid-name
            n = 0 #pylint: disable=invalid-name
            m = 1 #pylint: disable=invalid-name
            l = 0 #pylint: disable=invalid-name
            # Plot suspected outliers
            while n < len(values):
                data_points = suspected_outliers[n]
                for i in data_points:
                    if l == 0:
                        ax.plot(i, m/tick_values, 'o', color='grey', markersize=5, alpha=0.3,
                                label='Suspected outlier')
                        l += 1 #pylint: disable=invalid-name
                    else:
                        ax.plot(i, m/tick_values, 'o', color='grey', markersize=5, alpha=0.3)
                n += 1 #pylint: disable=invalid-name
                m += 1 #pylint: disable=invalid-name
            n = 0 #pylint: disable=invalid-name
            m = 1 #pylint: disable=invalid-name
            l = 0 #pylint: disable=invalid-name
            # Plot outliers
            while n < len(values):
                data_points = outliers[n]
                for i in data_points:
                    if l == 0:
                        ax.plot(i, m/tick_values, 'o', color='red', markersize=5,
                                alpha=0.3, label='Outlier')
                        l += 1 #pylint: disable=invalid-name
                    else:
                        ax.plot(i, m/tick_values, 'o', color='red', markersize=5, alpha=0.3)
                n += 1 #pylint: disable=invalid-name
                m += 1 #pylint: disable=invalid-name

            # Add legend
            ax.legend(loc='best', edgecolor='w')
            # Turn spines off
            for i in ['top', 'bottom', 'right', 'left']:
                ax.spines[i].set_visible(False)
            # Add gridlines
            ax.grid(which="major", axis="x", color='black', linestyle='-', linewidth=1, alpha=0.2)
            ax.set_axisbelow(True)
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
            if title is not None:
                ax.set_title(title)
        return ax, cont.collect()


class Profiler():
    """
    Class to build a HANA Profiler, including:
    - Variable descriptions
    - Missing values %
    - High cardinality %
    - Skewness
    - Numeric distributions
    - Categorical distributions
    - Correlations
    - High correlaton warnings

    """

    def description(self, data, key, bins=20, missing_threshold=10, card_threshold=100, #pylint: disable=too-many-locals, too-many-arguments, too-many-branches, too-many-statements, no-self-use
                    skew_threshold=0.5, figsize=None):
        """
        Returns a HANA profiler, including:
        - Variable descriptions
        - Missing values %
        - High cardinality %
        - Skewness
        - Numeric distributions
        - Categorical distributions
        - Correlations
        - High correlaton warnings

        Parameters
        ----------
        data : DataFrame
            DataFrame to use for the plot.
        key : str, optional
            Key in the DataFrame.
        bins : int, optional
            Number of bins for numeric distributions. Default value = 20.
        missing_threshold : float
            Percentage threshold to display missing values.
        card_threshold : int
            Threshold for column to be considered with high cardinality.
        skew_threshold : float
            Absolute value threshold for column to be considered as highly skewed.
        tight_layout : bool, optional
            Use matplotlib tight layout or not.
        figsize : tuple, optional
            Size of figure to be plotted. First element is width, second is height.

        Note: categorical columns with cardinality warnings are not plotted.

        Returns
        -------
        fig : Figure
            matplotlib axis of the profiler
        """
        conn_context = data.connection_context
        print("- Creating data description")
        number_variables = len(data.columns)
        number_observations = data.count()
        numeric = [i for i in data.columns if data.is_numeric(i)]
        categorical = [i[0] for i in data.dtypes() if (i[1] == 'NVARCHAR') or (i[1] == 'VARCHAR')]
        date = [i[0] for i in data.dtypes() if (i[1] == 'DATE') or (i[1] == 'TIMESTAMP')]
        print("- Counting missing values")
        # missing values
        warnings_missing = {}
        for i in data.columns:
            query = 'SELECT SUM(CASE WHEN {0} is NULL THEN 1 ELSE 0 END) AS "nulls" FROM ({1})'
            pct_missing = conn_context.sql(query.format(i, data.select_statement))
            pct_missing = pct_missing.collect().values[0][0]
            pct_missing = pct_missing/number_observations
            if pct_missing > missing_threshold/100:
                warnings_missing[i] = pct_missing
        print("- Judging high cardinality")
        # cardinality
        warnings_cardinality = {}
        warnings_constant = {}
        for i in data.columns:
            query = 'SELECT COUNT(DISTINCT {0}) AS "unique" FROM ({1})'
            cardinality = conn_context.sql(query.format(i, data.select_statement))
            cardinality = cardinality.collect().values[0][0]
            if cardinality > card_threshold:
                warnings_cardinality[i] = (cardinality/number_observations)*100
            elif cardinality == 1:
                warnings_constant[i] = data.collect()[i].unique()
        print("- Finding skewed variables")
        # Skewed
        warnings_skewness = {}
        cont, cat = stats.univariate_analysis(conn_context=conn_context, data=data, cols=numeric) #pylint: disable=unused-variable
        for i in numeric:
            skewness = cont.collect()['STAT_VALUE']
            stat = 'STAT_NAME'
            val = 'skewness'
            var = 'VARIABLE_NAME'
            skewness = skewness.loc[(cont.collect()[stat] == val) & (cont.collect()[var] == i)]
            skewness = skewness.values[0]
            if abs(skewness) > skew_threshold:
                warnings_skewness[i] = skewness
            else:
                pass
        if key:
            if key in numeric:
                numeric.remove(key)
            elif key in categorical:
                categorical.remove(key)
            elif key in date:
                date.remove(key)
        for i in warnings_cardinality:
            if i in categorical:
                categorical.remove(i)
            else:
                pass
        rows = 4
        m = 0 #pylint: disable=invalid-name
        o = 0 #pylint: disable=invalid-name
        while o < len(numeric):
            if m <= 4:
                m += 1 #pylint: disable=invalid-name
            elif m > 4:
                rows += 2
                m = 0 #pylint: disable=invalid-name
                m += 1 #pylint: disable=invalid-name
            o += 1 #pylint: disable=invalid-name
        rows += 2
        rows += 1
        m = 0 #pylint: disable=invalid-name
        o = 0 #pylint: disable=invalid-name
        while o < len(categorical):
            if m <= 4:
                m += 1 #pylint: disable=invalid-name
            elif m > 4:
                rows += 2
                m = 0 #pylint: disable=invalid-name
                m += 1 #pylint: disable=invalid-name
            o += 1 #pylint: disable=invalid-name
        rows += 2
        rows += 1
        rows += 4
        # Make figure
        fig = plt.figure(figsize=(20, 40))
        ax1 = plt.subplot2grid((rows, 5), (0, 0), rowspan=1, colspan=5) #pylint: disable=unused-variable
        alignment = {'horizontalalignment': 'left', 'verticalalignment': 'baseline'}
        t = plt.text(0, 0, "Data Description", fontdict={'size':30, 'fontweight':'bold'}, #pylint: disable=unused-variable, invalid-name
                     **alignment)
        plt.axis('off')
        # Data description
        ax2 = plt.subplot2grid((rows, 5), (1, 0), rowspan=2, colspan=1)
        labels = "Numeric", "Categorical", "Date"
        sizes = [len([i for i in data.columns if data.is_numeric(i)]),
                 len([i[0] for i in data.dtypes() if (i[1] == 'NVARCHAR') or (i[1] == 'VARCHAR')]),
                 len([i[0] for i in data.dtypes() if (i[1] == 'DATE') or (i[1] == 'TIMESTAMP')])]
        ax2.barh(labels[::-1], sizes[::-1])
        ax2.grid(which="major", axis="x", color='black', linestyle='-', linewidth=1, alpha=0.2)
        ax2.set_axisbelow(True)
        ax2.spines['right'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.xaxis.set_ticks_position('none')
        ax2.set_title("Variable Types")
        # Missing values
        ax3 = plt.subplot2grid((rows, 5), (1, 1), rowspan=2, colspan=1)
        labels = list(warnings_missing.keys())
        sizes = list(warnings_missing.values())
        ax3.barh(labels[::-1], sizes[::-1])
        ax3.grid(which="major", axis="x", color='black', linestyle='-', linewidth=1, alpha=0.2)
        ax3.set_axisbelow(True)
        ax3.spines['right'].set_visible(False)
        ax3.spines['bottom'].set_visible(False)
        ax3.spines['top'].set_visible(False)
        ax3.xaxis.set_ticks_position('none')
        ax3.set_title("Missing Values %")
        # High cardinality
        ax4 = plt.subplot2grid((rows, 5), (1, 2), rowspan=2, colspan=1)
        labels = list(warnings_cardinality.keys())
        sizes = list(warnings_cardinality.values())
        ax4.barh(labels[::-1], sizes[::-1])
        ax4.grid(which="major", axis="x", color='black', linestyle='-', linewidth=1, alpha=0.2)
        ax4.set_axisbelow(True)
        ax4.spines['right'].set_visible(False)
        ax4.spines['bottom'].set_visible(False)
        ax4.spines['top'].set_visible(False)
        ax4.xaxis.set_ticks_position('none')
        ax4.set_title("High cardinality %")
        # Skewed variables
        ax5 = plt.subplot2grid((rows, 5), (1, 3), rowspan=2, colspan=1)
        labels = list(warnings_skewness.keys())
        sizes = list(warnings_skewness.values())
        ax5.barh(labels[::-1], sizes[::-1])
        ax5.grid(which="major", axis="x", color='black', linestyle='-', linewidth=1, alpha=0.2)
        ax5.set_axisbelow(True)
        ax5.spines['right'].set_visible(False)
        ax5.spines['bottom'].set_visible(False)
        ax5.spines['top'].set_visible(False)
        ax5.xaxis.set_ticks_position('none')
        ax5.set_title("Highly skewed variables")
        # Description summary
        ax6 = plt.subplot2grid((rows, 5), (1, 4), rowspan=2, colspan=1)
        alignment = {'horizontalalignment': 'left', 'verticalalignment': 'baseline'}
        t = plt.text(0, 0.7, "Data description summary", fontweight='bold', #pylint: disable=invalid-name
                     size=12, **alignment)
        text = "-  There are {} variables and {} rows in this dataset"
        t = plt.text(0, 0.6, text.format(number_variables, number_observations), **alignment) #pylint: disable=invalid-name
        high_missing_values_pct = [i for i in warnings_missing.values() if i > 0.1]
        if warnings_missing:
            text = "-  There are {} variables with a high % of missing values"
            t = plt.text(0, 0.5, text.format(len(high_missing_values_pct)), **alignment) #pylint: disable=invalid-name
        else:
            t = plt.text(0, 0.5, "-  No missing values", **alignment) #pylint: disable=invalid-name
        if warnings_constant:
            text = "-  {} variables are constant: [{}]"
            t = plt.text(0, 0.4, text.format(len(warnings_constant), #pylint: disable=invalid-name
                                             list(warnings_constant.keys())), **alignment)
        else:
            text = "-  {} variables have high cardinality and 0 are constant"
            t = plt.text(0, 0.4, text.format(len(list(warnings_cardinality.keys()))), #pylint: disable=invalid-name
                         **alignment)
        if warnings_skewness:
            text = "-  {} variables are skewed, consider transformation"
            ax6.text(0, 0.3, text.format(len(warnings_skewness)), **alignment)
        else:
            ax6.text(0, 0.3, "-  No variables are skewed", **alignment)
        plt.axis('off')
        ax7 = plt.subplot2grid((rows, 5), (3, 0), rowspan=1, colspan=5) #pylint: disable=unused-variable
        alignment = {'horizontalalignment': 'left', 'verticalalignment': 'baseline'}
        t = plt.text(0, 0, "Numeric Distributions", #pylint: disable=invalid-name
                     fontdict={'size':30, 'fontweight':'bold'}, **alignment)
        plt.axis('off')
        # Numeric distributions
        print("- Calculating numeric distributions")
        n = 4 #pylint: disable=invalid-name
        m = 0 #pylint: disable=invalid-name
        o = 0 #pylint: disable=invalid-name
        for i in numeric:
            if m <= 4:
                ax = plt.subplot2grid((rows, 5), (n, m), rowspan=2, colspan=1) #pylint: disable=invalid-name
                eda = EDAVisualizer(ax)
                ax, dist_data = eda.distribution_plot(data=data, column=i, bins=bins, #pylint: disable=invalid-name, unused-variable
                                                      title="Distribution of {}".format(i),
                                                      debrief=False)
                m += 1 #pylint: disable=invalid-name
            elif m > 4:
                n += 2 #pylint: disable=invalid-name
                m = 0 #pylint: disable=invalid-name
                ax = plt.subplot2grid((rows, 5), (n, m), rowspan=2, colspan=1) #pylint: disable=invalid-name, unused-variable
                eda = EDAVisualizer(ax)
                ax, dist_data = eda.distribution_plot(data=data, column=i, #pylint: disable=invalid-name
                                                      bins=bins,
                                                      title="Distribution of {}".format(i),
                                                      debrief=False)
                m += 1 #pylint: disable=invalid-name
            o += 1 #pylint: disable=invalid-name
        n += 2 #pylint: disable=invalid-name
        ax8 = plt.subplot2grid((rows, 5), (n, 0), rowspan=1, colspan=5) #pylint: disable=unused-variable
        alignment = {'horizontalalignment': 'left', 'verticalalignment': 'baseline'}
        t = plt.text(0, 0, "Categorical Distributions", #pylint: disable=invalid-name
                     fontdict={'size':30, 'fontweight':'bold'}, **alignment)
        plt.axis('off')
        n += 1 #pylint: disable=invalid-name
        # Categorical distributions
        print("- Calculating categorical distributions")
        m = 0 #pylint: disable=invalid-name
        o = 0 #pylint: disable=invalid-name
        for i in categorical:
            if m <= 4:
                ax = plt.subplot2grid((rows, 5), (n, m), rowspan=2, colspan=1) #pylint: disable=invalid-name
                eda = EDAVisualizer(ax)
                ax, pie_data = eda.pie_plot(data=data, column=i, title="% of {}".format(i), #pylint: disable=invalid-name, unused-variable
                                            legend=False)
                m += 1 #pylint: disable=invalid-name
            elif m > 4:
                n += 2 #pylint: disable=invalid-name
                m = 0 #pylint: disable=invalid-name
                ax = plt.subplot2grid((rows, 5), (n, m), rowspan=2, colspan=1) #pylint: disable=invalid-name
                eda = EDAVisualizer(ax)
                ax, pie_data = eda.pie_plot(data=data, column=i, title="% of {}".format(i), #pylint: disable=invalid-name
                                            legend=False)
                m += 1 #pylint: disable=invalid-name
            o += 1 #pylint: disable=invalid-name
        n += 2 #pylint: disable=invalid-name
        ax9 = plt.subplot2grid((rows, 5), (n, 0), rowspan=1, colspan=5) #pylint: disable=unused-variable
        alignment = {'horizontalalignment': 'left', 'verticalalignment': 'baseline'}
        t = plt.text(0, 0, "Data Correlations", fontdict={'size':30, 'fontweight':'bold'}, #pylint: disable=invalid-name
                     **alignment)
        plt.axis('off')
        n += 1 #pylint: disable=invalid-name
        # Correlation plot
        print("- Calculating correlations")
        ax10 = plt.subplot2grid((rows, 5), (n, 0), rowspan=4, colspan=3)
        eda = EDAVisualizer(ax10)
        ax10, corr = eda.correlation_plot(data=data, corr_cols=numeric, label=True)
        warnings_correlation = {}
        if len(numeric) > 1:
            for i, col in enumerate(numeric): #pylint: disable=unused-variable
                for j in range(i+1, len(numeric)):
                    dfc = stats.pearsonr_matrix(conn_context=conn_context, data=data,
                                                cols=[numeric[i], numeric[j]]).collect()
                    dfc = dfc.iloc[1, 1]
                    if (i != j) and (abs(dfc) > 0.3):
                        warnings_correlation[numeric[i], numeric[j]] = dfc
                    else:
                        pass
        ax11 = plt.subplot2grid((rows, 5), (n, 3), rowspan=4, colspan=2) #pylint: disable=unused-variable
        alignment = {'horizontalalignment': 'left', 'verticalalignment': 'baseline'}
        t = plt.text(0, 0.8, "Data correlations summary", fontweight='bold', size=20, **alignment) #pylint: disable=invalid-name
        text = "There are {} pair(s) of variables that are show significant correlation"
        t = plt.text(0, 0.7, text.format(len(warnings_correlation), **alignment)) #pylint: disable=invalid-name
        n = 0.7 #pylint: disable=invalid-name
        m = 1 #pylint: disable=invalid-name
        for i in warnings_correlation:
            corr = warnings_correlation.get(i)
            if abs(corr) >= 0.5:
                v = n-(m*0.05) #pylint: disable=invalid-name
                text = "-  {} and {} are highly correlated, p = {:.2f}"
                t = plt.text(0, v, text.format(i[0], i[1], warnings_correlation.get(i)), #pylint: disable=invalid-name
                             **alignment)
                m += 1 #pylint: disable=invalid-name
            elif 0.3 <= abs(corr) < 0.5:
                v = n-(m*0.05) #pylint: disable=invalid-name
                text = "-  {} and {} are moderately correlated, p = {:.2f}"
                t = plt.text(0, v, text.format(i[0], i[1], warnings_correlation.get(i)), #pylint: disable=invalid-name
                             **alignment)
                m += 1 #pylint: disable=invalid-name
            else:
                pass
        plt.axis('off')
        if isinstance(figsize, tuple):
            a, b = figsize #pylint: disable=invalid-name
            plt.figure(figsize=(a, b))
        plt.tight_layout()
        plt.close()
        print("\n ---> Profiler is ready to plot, run the " +
              "returned figure to display the results.")
        return fig

    def set_size(self, fig, figsize): #pylint: disable=no-self-use
        """
        Set the size of the data description plot, in inches.

        Parameters
        ----------
        fig : ax
            The returned axes constructed by the description method.
        figsize : tuple
            Tuple of width and height for the plot.
        """
        while True:
            if isinstance(figsize, tuple):
                fig.set_figwidth(figsize[0])
                fig.set_figheight(figsize[1])
                print("Axes size set: width = {0}, height = {1} inches".format(figsize[0],
                                                                               figsize[1]))
                print("\n ---> Profiler is ready to plot, run the " +
                      "returned figure to display the results.")
            else:
                print("Please enter a tuple for figsize.")
            break
