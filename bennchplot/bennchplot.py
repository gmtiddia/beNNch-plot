"""
beNNch-plot - standardized plotting routines for performance benchmarks.
Copyright (C) 2021 Forschungszentrum Juelich GmbH, INM-6

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
this program. If not, see <https://www.gnu.org/licenses/>.

SPDX-License-Identifier: GPL-3.0-or-later
"""

"""
Class for benchmarking plots
"""
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec
import numpy as np
import yaml
import os
try:
    from . import plot_params as pp
except ImportError:
    import plot_params as pp

plt.rcParams.update(plt.rcParamsDefault)

class Plot():
    """
    Class organizing benchmarking plots.

    Attributes
    ----------
    x_axis : str or list
        variable to be plotted on x-axis
    x_ticks : str, optional

    data_file : str, optional
        path to data
    matplotlib_params : dict, optional
        parameters passed to matplotlib
    color_params : dict, optional
        unique colors for variables
    additional_params : dict, optional
        additional parameters used for plotting
    label_params : dict, optional
        labels used when plotting
    time_scaling : int, optional
        scaling parameter for simulation time
   """

    def __init__(self, x_axis,
                 x_ticks='data',
                 data_file='/path/to/data',
                 matplotlib_params=pp.matplotlib_params,
                 color_params=pp.color_params,
                 additional_params=pp.additional_params,
                 label_params=pp.label_params,
                 time_scaling=1):

        self.x_axis = x_axis
        self.x_ticks = x_ticks
        self.matplotlib_params = matplotlib_params
        self.additional_params = additional_params
        self.color_params = color_params
        self.label_params = label_params
        self.time_scaling = time_scaling

        self.load_data(data_file)
        self.compute_derived_quantities()

        print(self.matplotlib_params)
        for p in self.matplotlib_params.keys():
            plt.rcParams[p] = self.matplotlib_params[p]

        plt.rc('text.latex', preamble=r'\usepackage{gensymb}')


    def load_data(self, data_file):
        """
        Load data to dataframe, to be used later when plotting.

        Group the data by specified operations.

        Attributes
        ----------
        data_file : str
            data file to be loaded and later plotted

        Raises
        ------
        ValueError
        """
        try:
            self.df = pd.read_csv(data_file, delimiter=',', index_col=0)
        except FileNotFoundError:
            print('File could not be found')
            quit()

        dict_ = {'nodes': 'first',
                 'gpus_per_node': 'first',
                 'model_time_sim': 'first',
                 'time_configure': ['mean', 'std'],
                 'time_create_nodes': ['mean', 'std'],
                 'time_connect_local': ['mean', 'std'],
                 'time_network_local_tot': ['mean', 'std'],
                 'time_area_packing': ['mean', 'std'],
                 'time_connect_global': ['mean', 'std'],
                 'total_constr': ['mean', 'std'],
                 'time_calibrate': ['mean', 'std'],
                 'time_presimulate': ['mean', 'std'],
                 'time_simulate': ['mean', 'std'],
                 'sim_factor': ['mean', 'std']
                 }
        

        self.df = self.df.groupby(
            ['nodes',
             'gpus_per_node',
             'model_time_sim'], as_index=False).agg(dict_)


    def compute_derived_quantities(self):
        """
        Do computations to get parameters needed for plotting.
        """

        self.df['model_time_sim'] /= self.time_scaling
        
        
        """
        if self.detailed_timers:
            self.df['time_construction_create+time_construction_connect'] = (
                self.df['py_time_create'] + self.df['py_time_connect'])
            self.df['time_construction_create+time_construction_connect_std'] = (
                np.sqrt((self.df['time_construction_create_std']**2 +
                         self.df['time_construction_connect_std']**2)))

            self.df['time_phase_total'] = (
                # self.df['time_update_spike_data'] +
                self.df['time_communicate_spike_data'] +
                self.df['time_deliver_spike_data'] +
                self.df['time_collocate_spike_data'])
            self.df['time_phase_total_std'] = \
                np.sqrt(
                # self.df['time_update_spike_data_std']**2 +
                self.df['time_communicate_spike_data_std']**2 +
                self.df['time_deliver_spike_data_std']**2 +
                self.df['time_collocate_spike_data_std']**2
            )
            self.df['phase_total_factor'] = (
                self.df['time_phase_total'] /
                self.df['model_time_sim'])
            self.df['phase_total_factor_std'] = (
                self.df['time_phase_total_std'] /
                self.df['model_time_sim'])

            for phase in ['update', 'communicate', 'deliver', 'collocate']:
                self.df['phase_' + phase + '_factor'] = (
                    self.df['time_' + phase + '_spike_data'] /
                    self.df['model_time_sim'])

                self.df['phase_' + phase + '_factor' + '_std'] = (
                    self.df['time_' + phase + '_spike_data' + '_std'] /
                    self.df['model_time_sim'])

                self.df['frac_phase_' + phase] = (
                    100 * self.df['time_' + phase + '_spike_data'] /
                    self.df['time_phase_total'])

                self.df['frac_phase_' + phase + '_std'] = (
                    100 * self.df['time_' + phase + '_spike_data' + '_std'] /
                    self.df['time_phase_total'])
        
        self.df['total_memory_per_node'] = (self.df['total_memory']['mean'].values /
                                            self.df['num_nodes'].values.flatten())
        self.df['total_memory_per_node_std'] = (self.df['total_memory']['std'].values /
                                                self.df['num_nodes'].values.flatten())
        """
    
    """
    def plot_fractions(self, axis, fill_variables,
                       interpolate=False, step=None, log=False, alpha=1.,
                       error=False):
        """
        #Fill area between curves.

        #axis : Matplotlib axes object
        #fill_variables : list
        #    variables (e.g. timers) to be plotted as fill  between graph and
        #    x axis
        #interpolate : bool, default
        #    whether to interpolate between the curves
        #step : {'pre', 'post', 'mid'}, optional
        #    should the filling be a step function
        #log : bool, default
        #    whether the x-axes should have logarithmic scale
        #alpha, int, default
        #    alpha value of fill_between plot
        #error : bool
        #    whether plot should have error bars
    """

        fill_height = 0
        for fill in fill_variables:
            axis.fill_between(np.squeeze(self.df[self.x_axis]),
                              fill_height,
                              np.squeeze(self.df[fill]) + fill_height,
                              label=self.label_params[fill],
                              facecolor=self.color_params[fill],
                              interpolate=interpolate,
                              step=step,
                              alpha=alpha,
                              linewidth=0.5,
                              edgecolor='#444444')
            if error:
                axis.errorbar(np.squeeze(self.df[self.x_axis]),
                              np.squeeze(self.df[fill]) + fill_height,
                              yerr=np.squeeze(self.df[fill + '_std']),
                              capsize=3,
                              capthick=1,
                              color='k',
                              fmt='none'
                              )
            fill_height += self.df[fill].to_numpy()

        if self.x_ticks == 'data':
            axis.set_xticks(np.squeeze(self.df[self.x_axis]))
        else:
            axis.set_xticks(self.x_ticks)

        if log:
            axis.set_xscale('log')
            axis.tick_params(bottom=False, which='minor')
            axis.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())
    """

    def plot_main(self, quantities, axis, log=(False, False), labels=None, colors=None, ecolor=None,
                  error=False, fmt='none'):
        """
        Main plotting function.

        Attributes
        ----------
        quantities : list
            list with plotting quantities
        axis : axis object
            axis object used when plotting
        log : tuple of bools, default
            whether x and y axis should have logarithmic scale
        error : bool, default
            whether or not to plot error bars
        fmt : string
            matplotlib format string (fmt) for defining line style
        """

        print(self.df)

        for dumy, y in enumerate(quantities):
            label = self.label_params[y] if labels is None else labels[dumy]
            color = self.color_params[y] if colors is None else colors[dumy]
            ecolor = self.color_params[y] if ecolor is None else ecolor

            print(ecolor)
            
            axis.plot(self.df[self.x_axis],
                      self.df[y]['mean'],
                      marker=None,
                      label=label,
                      color=color,
                      linewidth=2)

            if error:
                axis.errorbar(
                    self.df[self.x_axis].values,
                    self.df[y]['mean'].values,
                    yerr=self.df[y]['std'].values,
                    marker=None,
                    capsize=3,
                    capthick=1,
                    color=color,
                    ecolor=ecolor,
                    fmt=fmt)

        if self.x_ticks == 'data':
            axis.set_xticks(self.df[self.x_axis].values.flatten())
        else:
            axis.set_xticks(self.x_ticks)

        if log[0]:
            axis.set_xscale('log')
        if log[1]:
            axis.tick_params(bottom=False, which='minor')
            axis.set_yscale('log')


    def plot_fractions(self, axis, fill_variables,
                       interpolate=False, step=None, log=False, alpha=1.,
                       error=False):
        """
        Fill area between curves.

        axis : Matplotlib axes object
        fill_variables : list
            variables (e.g. timers) to be plotted as fill  between graph and
            x axis
        interpolate : bool, default
            whether to interpolate between the curves
        step : {'pre', 'post', 'mid'}, optional
            should the filling be a step function
        log : bool, default
            whether the x-axes should have logarithmic scale
        alpha, int, default
            alpha value of fill_between plot
        error : bool
            whether plot should have error bars
        """

        fill_height = 0
        for fill in fill_variables:
            print(self.df[self.x_axis])
            axis.fill_between(np.squeeze(self.df[self.x_axis].values.flatten()),
                              fill_height,
                              np.squeeze(self.df[fill]['mean']) + fill_height,
                              label=self.label_params[fill],
                              facecolor=self.color_params[fill],
                              interpolate=interpolate,
                              step=step,
                              alpha=alpha,
                              linewidth=0.5,
                              edgecolor='#444444')
            if error:
                axis.errorbar(np.squeeze(self.df[self.x_axis].values.flatten()),
                              np.squeeze(self.df[fill]['mean']) + fill_height,
                              yerr=np.squeeze(self.df[fill]['std']),
                              capsize=3,
                              capthick=1,
                              color='k',
                              fmt='none'
                              )
            fill_height += self.df[fill]['mean'].to_numpy()

        if self.x_ticks == 'data':
            axis.set_xticks(np.squeeze(self.df[self.x_axis]))
        else:
            axis.set_xticks(self.x_ticks)

        if log:
            axis.set_xscale('log')
            axis.tick_params(bottom=False, which='minor')
            axis.get_xaxis().set_major_formatter(
                matplotlib.ticker.ScalarFormatter())

    def merge_legends(self, ax1, ax2):
        """
        Merge legends from two axes, display them in the first

        Attributes
        ----------
        ax1 : axes object
            first axis
        ax2 : axes object
            second axis
        """
        handles, labels = [(a + b) for a, b in zip(
            ax2.get_legend_handles_labels(),
            ax1.get_legend_handles_labels())]
        ax1.legend(handles, labels, loc='upper right')

    def simple_axis(self, ax):
        """
        Remove top and right spines.

        Attributes
        ----------
        ax : axes object
            axes object for which to adjust spines
        """
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        # Only show ticks on the left and bottom spines
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
