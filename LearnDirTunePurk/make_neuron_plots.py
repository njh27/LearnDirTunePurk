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
    if len(fr) == 0:
        # Empty rate so return input values
        return min_fr, max_fr
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
        if len(fr_by_set[curr_set]) > 0:
            fr_by_set[curr_set] = np.nanmean(fr_by_set[curr_set])
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
        if len(x) > 0:
            x = np.nanmean(x)
            y = np.nanmean(y)
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


def binned_mean_firing_traces_2D(bin_fr_data, bin_x_data, bin_y_data, ax=None,
                        min_fr=None, max_fr=None, use_map="Reds",
                        show_colorbar=True, return_last_plot=False):
    """ """
    if ax is None:
        fig = plt.figure()
        ax = plt.axes()

    if min_fr is None:
        min_fr = np.inf
    if max_fr is None:
        max_fr = 0.
    if (min_fr is None) or (max_fr is None):
        for fr_trace in bin_fr_data:
            min_fr, max_fr = update_min_max_fr(fr_trace, min_fr, max_fr)

    for bin_ind in range(0, len(bin_fr_data)):
        colors, colorbar_sm = fr_to_colors(bin_fr_data[bin_ind], use_map,
                                             min_fr, max_fr)
        last_s_plot = ax.scatter(bin_x_data[bin_ind], bin_y_data[bin_ind],
                                 color=colors)
    if show_colorbar:
        plt.colorbar(colorbar_sm, ticks=np.linspace(np.floor(min_fr), np.ceil(max_fr), 10))
    if return_last_plot:
        return ax, last_s_plot
    else:
        return ax


def plot_instruction_firing_position_xy(ldp_sess, bin_edges, neuron_series_name,
                                time_window=None, base_block="StabTunePre",
                                rescale=False):

    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    # Blocks are hard coded for learning instruction trials binned by instructed
    use_block = "Learning"
    use_trial_set = "instruction"
    use_bin_basis = "instructed"
    fig = plt.figure()
    learn_ax = plt.axes()
    bin_xy_out = ab.get_binned_mean_xy_traces(
                ldp_sess, bin_edges, "eye position", time_window,
                blocks=use_block, trial_sets=use_trial_set,
                bin_basis=use_bin_basis, rescale=rescale)
    if rescale:
        bin_x_data, bin_y_data, alpha_bin = bin_xy_out
    else:
        bin_x_data, bin_y_data  = bin_xy_out
        alpha_bin = None
    bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(
                                ldp_sess, base_block, use_trial_set, "eye position",
                                bin_x_data, bin_y_data,
                                alpha_scale_factors=alpha_bin)
    bin_fr_data = an.get_binned_mean_firing_trace(ldp_sess, bin_edges,
                                    neuron_series_name,
                                    time_window, blocks=use_block, trial_sets=use_trial_set,
                                    bin_basis=use_bin_basis, return_t_inds=False)

    learn_ax = binned_mean_firing_traces_2D(bin_fr_data, bin_x_data, bin_y_data,
                                            ax=learn_ax, use_map="Reds")

    learn_ax.set_ylabel("Learning axis eye position (deg)")
    learn_ax.set_xlabel("Pursuit axis eye position (deg)")
    learn_ax.set_title("Instruction trials")
    learn_ax.axvline(0, color='b')
    learn_ax.axhline(0, color='b')

    return learn_ax


def plot_tuning_probe_firing_position_xy(ldp_sess, bin_edges, neuron_series_name,
                                    time_window=None, base_block="StabTunePre",
                                    trial_sets=None, rescale=False):

    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    # Blocks are hard coded for learning instruction trials binned by instructed
    # Trial set is iterated over four dirs by default
    use_block = "Learning"
    use_bin_basis = "instructed"
    if trial_sets is None:
        trial_sets = ldp_sess.four_dir_trial_sets
    if not isinstance(trial_sets, list):
        trial_sets = [trial_sets]
    bin_fr_data = {}
    bin_x_data = {}
    bin_y_data = {}
    alpha_bin = {}
    fig = plt.figure()
    learn_ax = plt.axes()
    min_fr = np.inf
    max_fr = 0.
    for curr_set in trial_sets:
        bin_xy_out = ab.get_binned_mean_xy_traces(
                                ldp_sess, bin_edges, "eye position", time_window,
                                blocks=use_block, trial_sets=curr_set,
                                bin_basis=use_bin_basis, rescale=rescale)
        if rescale:
            bin_x_data[curr_set], bin_y_data[curr_set], alpha_bin[curr_set] = bin_xy_out
        else:
            bin_x_data[curr_set], bin_y_data[curr_set]  = bin_xy_out
            alpha_bin[curr_set] = None
        bin_x_data[curr_set], bin_y_data[curr_set] = ab.subtract_baseline_tuning_binned(
                                ldp_sess, base_block, curr_set, "eye position",
                                bin_x_data[curr_set], bin_y_data[curr_set],
                                alpha_scale_factors=alpha_bin[curr_set])

        bin_fr_data[curr_set] = an.get_binned_mean_firing_trace(ldp_sess, bin_edges,
                                        neuron_series_name,
                                        time_window, blocks=use_block, trial_sets=curr_set,
                                        bin_basis=use_bin_basis, return_t_inds=False)

        for fr_trace in bin_fr_data[curr_set]:
            min_fr, max_fr = update_min_max_fr(fr_trace, min_fr, max_fr)

    should_show = True
    for curr_set in trial_sets:
        learn_ax = binned_mean_firing_traces_2D(bin_fr_data[curr_set],
                                    bin_x_data[curr_set], bin_y_data[curr_set],
                                    ax=learn_ax, min_fr=min_fr,
                                    max_fr=max_fr, use_map="Reds",
                                    show_colorbar=should_show)
        should_show = False

    learn_ax.set_ylabel("Learning axis eye position (deg)")
    learn_ax.set_xlabel("Pursuit axis eye position (deg)")
    learn_ax.set_title("Direction tuning probe trials")
    learn_ax.axvline(0, color='b')
    learn_ax.axhline(0, color='b')

    # save_name = "/Users/nathanhall/onedrive - duke university/sync/LearnDirTunePurk/Data/Maestro/" + ldp_sess.session_name + ".pdf"
    # plt.savefig(save_name)

    return learn_ax


def plot_post_tuning_firing_position_xy(ldp_sess, neuron_series_name, time_window=None,
                                base_block="StabTunePre", trial_sets=None, rescale=False):
    """
    """
    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    blocks = ["StabTunePost", "StabTuneWash"]
    colormaps = {'StabTunePost': "Reds",
                 'StabTuneWash': "Greens"
                 }
    p_b_labels = {'StabTunePost': "Post", 'StabTuneWash': 'Washout'}
    if trial_sets is None:
        trial_sets = ldp_sess.four_dir_trial_sets
    if not isinstance(trial_sets, list):
        trial_sets = [trial_sets]
    fig = plt.figure()
    post_tune_ax = plt.axes()
    # Get firing rate min and max for colors
    min_fr = np.inf
    max_fr = 0.
    fr_by_block_set = {} # And just save rates now instead of computing twice
    for block in blocks:
        fr_by_block_set[block] = {}
        if ldp_sess.blocks[block] is None:
            continue
        for curr_set in trial_sets:
            fr_by_block_set[block][curr_set] = an.get_mean_firing_trace(ldp_sess,
                                                    neuron_series,
                                                    time_window,
                                                    block, curr_set)
            min_fr, max_fr = update_min_max_fr(fr_by_block_set[block][curr_set], min_fr, max_fr)


    for block in blocks:
        if ldp_sess.blocks[block] is None:
            continue
        for curr_set in trial_sets:
            xy_out = ab.get_mean_xy_traces(ldp_sess, "eye position", time_window,
                        blocks=block, trial_sets=curr_set,
                        rescale=rescale)
            if rescale:
                x, y, alpha = xy_out
            else:
                x, y = xy_out
                alpha = None
            x, y = ab.subtract_baseline_tuning(ldp_sess, base_block, curr_set,
                                               "eye position", x, y,
                                               alpha_scale_factors=alpha)
            colors, colorbar_sm = fr_to_colors(fr_by_block_set[block][curr_set],
                                            colormaps['block'], min_fr, max_fr)


            last_line = post_tune_ax.scatter(x, y, color=colors[block])
        last_line.set_label(p_b_labels[block])

    post_tune_ax.legend()
    post_tune_ax.set_ylabel("Learning axis eye position (deg)")
    post_tune_ax.set_xlabel("Pursuit axis eye position (deg)")
    post_tune_ax.set_title("Direction tuning after learning")
    post_tune_ax.axvline(0, color='k')
    post_tune_ax.axhline(0, color='k')

    # save_name = "/Users/nate/onedrive - duke university/sync/LearnDirTunePurk/Data/Maestro/" + ldp_sess.session_name + ".pdf"
    # plt.savefig(save_name)

    return post_tune_ax
