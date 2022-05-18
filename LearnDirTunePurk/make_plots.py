import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c_look
import LearnDirTunePurk.analyze_behavior as ab



def baseline_tuning_2D(ldp_sess, base_block, base_data, colors=None):
    """ Plots the 4 direction 2D tuning for the block and data sepcified from
    the baseline sets stored in ldp_sess. """

    colors = parse_colors(ldp_sess.four_dir_trial_sets, colors)
    fig = plt.figure()
    ax = plt.axes()
    for curr_set in ldp_sess.four_dir_trial_sets:
        ax.scatter(ldp_sess.baseline_tuning[base_block][base_data][curr_set][0, :],
                   ldp_sess.baseline_tuning[base_block][base_data][curr_set][1, :],
                   color=colors[curr_set])
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')

    return ax


def binned_mean_traces_2D(bin_x_data, bin_y_data, ax=None, color='k', saturation=None):
    """ """
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()
    if isinstance(color, str):
        color = c_look.to_rgb(color)
    if saturation is None:
        saturation = [.2, .8]
    color = np.array(color)
    if np.all(color == 0.):
        color[:] = 1.
    darkness = saturation[0]
    dark_step = (saturation[1] - saturation[0]) / len(bin_x_data)
    for x_trace, y_trace in zip(bin_x_data, bin_y_data):
        ax.scatter(x_trace, y_trace, color=(darkness * color))
        darkness += dark_step

    return ax


def mean_traces_2D(x, y, time_window, blocks,
                plot_ax=None, trial_sets=None, rotate=True):
    """ Makes xy plot of traces for the hardcoded directions for the data
    and info specified. """

    if trial_sets is None:
        trial_sets = ldp_sess.four_dir_trial_sets
    if plot_ax is None:
        plt.figure()
        plot_ax = plt.axes()





def scatter_2D(sess, data_name, time_window):
    pass


def single_traces():
    pass


def binned_traces():
    pass


def parse_colors(dict_names, colors):
    """ Checks colors against expected number of names and returns a dictionary
    with keys 'dict_names' corresponding to colors in 'colors'. """
    if colors is None:
        colors = {x: 'k' for x in dict_names}
    if isinstance(colors, str):
        colors = {x: colors for x in dict_names}
    if isinstance(colors, list):
        if isinstance(colors[0], str):
            if len(colors) == 1:
                colors = {x: 'k' for x in dict_names}
            elif len(colors) != len(dict_names):
                raise ValueError("Colors must be a single value or match the number of trial sets ({0}).".format(len(dict_names)))
            else:
                colors = {x: y for x,y in zip(dict_names, colors)}
        else:
            # Assume number specification
            if len(colors) == 3:
                colors = {x: colors for x in dict_names}
            elif len(colors) != len(dict_names):
                raise ValueError("Colors must be a single value or match the number of trial sets ({0}).".format(len(dict_names)))
            else:
                colors = {x: y for x,y in zip(dict_names, colors)}
    if not isinstance(colors, dict):
        raise ValueError("Colors input must be a string or list of strings or ints specifying valid matplotlib color values.")
    if len(colors) != len(dict_names):
        raise ValueError("Not enough colors specified for the {0} trial sets present.".format(len(dict_names)))

    return colors
