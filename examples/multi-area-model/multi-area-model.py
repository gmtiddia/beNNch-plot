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
import numpy as np
import bennchplot as bp
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import tol_colors



"""
define what to plot:
- data_file:
    Path to .csv file containing benchmarking measurements.
- x_axis:
    Giving a list of strings corresponding to the main scaling
    variable, typically 'num_nodes' or 'num_nvp'.
- time_scaling:
    Quotient between unit time of timing measurement and
    simulation. Usually, the former is given in s while
    the latter is given in ms. 
"""
args = {
    'data_file': 'processed_times_mean.csv',
    'x_axis': ['nodes'],
    'time_scaling': 1.0
}


light = tol_colors.tol_cset('light')


# Instantiate class
B = bp.Plot(**args)

# Figure layout
widths = [1, 1]
fig = plt.figure(figsize=(12, 6), constrained_layout=True)
spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig,
                         width_ratios=widths)

ax1 = fig.add_subplot(spec[:, 0])
ax2 = fig.add_subplot(spec[0, 1])



# Add plots
#B.plot_main(axis=ax1,
#            quantities=['total_constr', 'time_simulate'],
#            ecolor='k',
#            error=True)

B.plot_fractions(axis=ax1,
                 fill_variables=['total_constr',
                                 'time_simulate'],
                 interpolate=True,
                 step=None,
                 error=True)

B.plot_main(quantities=['sim_factor'], axis=ax2, error=True, fmt='-', ecolor=None)

# Set labels, limits etc.
#ax2.set_ylim(0, 300)
B.simple_axis(ax1)
B.simple_axis(ax2)

ax1.set_xlabel('Number of nodes')
ax1.set_ylabel(r'$T_{\mathrm{wall}}$ [s] for $T_{\mathrm{model}} =$'
               + f'{np.unique(B.df.model_time_sim.values)[0]} s')
ax2.set_xlabel('Number of nodes')
ax2.set_ylabel(r'real-time factor $T_{\mathrm{wall}}/$'
               r'$T_{\mathrm{model}}$')
#ax3.set_xlabel('Number of nodes')
#ax3.set_ylabel(r'relative $T_{\mathrm{wall}}$ [%]')

ax1.legend()
ax2.legend()

plt.show()
# Save figure
#plt.savefig('scaling.pdf')
