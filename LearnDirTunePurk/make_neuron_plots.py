import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c_look
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import LearnDirTunePurk.analyze_behavior as ab
import LearnDirTunePurk.analyze_neurons as an


# Set as global list for me to remember
seq_colormaps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']
def fr_to_colors(fr, use_map='Greys', vmin=None, vmax=None):
    """ Converts firing rate values to color map values according to input map
    'use_map'. """
    if vmin is None:
        vmin = np.amin(fr)
    if vmax is None:
        vmax = np.amax(fr)
    vmin = np.floor(vmin)
    vmax = np.ceil(vmax)
    cmap_norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_fun = getattr(cm, use_map)
    colors = cmap_fun(cmap_norm(fr))
    colorbar_sm = cm.ScalarMappable(cmap=use_map, norm=cmap_norm)

    return colors, colorbar_sm


def update_min_max_fr(fr, min_fr, max_fr):
    curr_max_fr = np.amax(fr)
    if curr_max_fr > max_fr:
        max_fr = curr_max_fr
    curr_min_fr = np.amin(fr)
    if curr_min_fr < min_fr:
        min_fr = curr_min_fr
    return min_fr, max_fr


def baseline_tuning_2D(ldp_sess, base_block, base_data, neuron_series,
                        use_map="Reds"):
    """ Plots the 4 direction 2D tuning for the block and data sepcified from
    the baseline sets stored in ldp_sess. """
    fix_trial_sets = [key for key in ldp_sess.trial_sets.keys() if key[0:4] == "fix_"]
    fix_time_window = [0, 600]
    fix_block = "FixTunePre"
    fig = plt.figure()
    ax = plt.axes()
    fr_by_set = {}
    max_fr = 0.
    min_fr = np.inf
    # Get firing rate for each trial to find max and min for colormap
    for curr_set in fix_trial_sets:
        fr_by_set[curr_set] = an.get_mean_firing_trace(ldp_sess,
                                                neuron_series,
                                                fix_time_window,
                                                fix_block, curr_set)
        min_fr, max_fr = update_min_max_fr(fr_by_set[curr_set], min_fr, max_fr)
    for curr_set in ldp_sess.four_dir_trial_sets:
        fr_by_set[curr_set] = an.get_mean_firing_trace(ldp_sess,
                                                neuron_series,
                                                ldp_sess.baseline_time_window,
                                                base_block, curr_set)
        min_fr, max_fr = update_min_max_fr(fr_by_set[curr_set], min_fr, max_fr)
    for curr_set in ldp_sess.four_dir_trial_sets:
        colors, colorbar_sm = fr_to_colors(fr_by_set[curr_set], use_map, min_fr, max_fr)
        s_plot = ax.scatter(ldp_sess.baseline_tuning[base_block][base_data][curr_set][0, :],
                   ldp_sess.baseline_tuning[base_block][base_data][curr_set][1, :],
                   color=colors)
    for curr_set in fix_trial_sets:
        colors, colorbar_sm = fr_to_colors(fr_by_set[curr_set], use_map, min_fr, max_fr)
        x, y = ab.get_mean_xy_traces(ldp_sess, base_data, fix_time_window, blocks=fix_block,
                                trial_sets=curr_set, rescale=False)
        s_plot = ax.scatter(x, y, color=colors)

    if "position" in base_data:
        units = " (deg)"
    elif "velocity" in base_data:
        units = " (deg/s)"
    else:
        print("Cannot find postion/velocity units for axis label with data", base_data)
    ax.set_ylabel("Learning axis " + base_data + units)
    ax.set_xlabel("Pursuit axis " + base_data + units)
    ax.set_title("Four direction baseline tuning trials")
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')
    plt.colorbar(colorbar_sm, ticks=np.linspace(np.floor(min_fr), np.ceil(max_fr), 10))

    return ax


def plot_instruction_position_xy(ldp_sess, bin_edges, time_window=None,
                                 base_block="StabTunePre", rescale=False):

    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    fig = plt.figure()
    learn_ax = plt.axes()
    bin_xy_out = ab.get_binned_mean_xy_traces(
                ldp_sess, bin_edges, "eye position", time_window,
                blocks="Learning", trial_sets="instruction",
                bin_basis="instructed", rescale=rescale)
    if rescale:
        bin_x_data, bin_y_data, alpha_bin = bin_xy_out
    else:
        bin_x_data, bin_y_data  = bin_xy_out
        alpha_bin = None
    bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(
                                ldp_sess, base_block, "instruction", "eye position",
                                bin_x_data, bin_y_data,
                                alpha_scale_factors=alpha_bin)
    learn_ax = binned_mean_traces_2D(bin_x_data, bin_y_data,
                            ax=learn_ax, color='k', saturation=None)

    learn_ax.set_ylabel("Learning axis eye position (deg)")
    learn_ax.set_xlabel("Pursuit axis eye position (deg)")
    learn_ax.set_title("Instruction trials")
    learn_ax.axvline(0, color='b')
    learn_ax.axhline(0, color='b')

    return learn_ax
