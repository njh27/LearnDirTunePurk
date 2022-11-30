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
                                                    neuron_series_name,
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
            # x, y = ab.subtract_baseline_tuning(ldp_sess, base_block, curr_set,
            #                                    "eye position", x, y,
            #                                    alpha_scale_factors=alpha)
        #     colors, colorbar_sm = fr_to_colors(fr_by_block_set[block][curr_set],
        #                                     colormaps[block], min_fr, max_fr)
        #
        #
        #     last_line = post_tune_ax.scatter(x, y, color=colors[block])
        # last_line.set_label(p_b_labels[block])
            colors, colorbar_sm = fr_to_colors(fr_by_block_set[block][curr_set],
                                                colormaps[block], min_fr, max_fr)
            s_plot = post_tune_ax.scatter(x, y, color=colors)
        s_plot.set_label(p_b_labels[block])

    post_tune_ax.legend()
    post_tune_ax.set_ylabel("Learning axis eye position (deg)")
    post_tune_ax.set_xlabel("Pursuit axis eye position (deg)")
    post_tune_ax.set_title("Direction tuning after learning")
    post_tune_ax.axvline(0, color='k')
    post_tune_ax.axhline(0, color='k')
    plt.colorbar(colorbar_sm, ticks=np.linspace(np.floor(min_fr), np.ceil(max_fr), 10))

    # save_name = "/Users/nate/onedrive - duke university/sync/LearnDirTunePurk/Data/Maestro/" + ldp_sess.session_name + ".pdf"
    # plt.savefig(save_name)

    return post_tune_ax


def baseline_firing_tuning(ldp_sess, neuron_series_name, base_block, base_data, colors=None):
    """ Plots the 4 direction 1D tuning vs. time for the block and data
    sepcified from the baseline sets stored in ldp_sess. """

    colors = parse_colors(ldp_sess.four_dir_trial_sets, colors)
    fig = plt.figure()
    pursuit_ax = plt.axes()
    fig = plt.figure()
    learn_ax = plt.axes()
    time = np.arange(ldp_sess.baseline_time_window[0], ldp_sess.baseline_time_window[1])
    for curr_set in ldp_sess.four_dir_trial_sets:
        # Need to plot the OPPOSITE of the orthogonal axis
        if ldp_sess.trial_set_base_axis[curr_set] == 0:
            plot_axis = 1
        elif ldp_sess.trial_set_base_axis[curr_set] == 1:
            plot_axis = 0
        else:
            raise ValueError("Unrecognized trial set baseline axis {0} for set {1}.".format(ldp_sess.trial_set_base_axis[curr_set], curr_set))

        fr = an.get_mean_firing_trace(ldp_sess,
                                                neuron_series_name,
                                                ldp_sess.baseline_time_window,
                                                base_block, curr_set)
        fix_fr = np.nanmean(fr[time < 75])
        fr -= fix_fr
        if "learn" in curr_set:
            if 'anti' in curr_set:
                line_label = "Anti-learning"
            else:
                line_label = "Learning"
            eye_data = ldp_sess.baseline_tuning[base_block][base_data][curr_set][plot_axis, :]
            use_ax = learn_ax
        elif "pursuit" in curr_set:
            if 'anti' in curr_set:
                line_label = "Anti-pursuit"
            else:
                line_label = "Pursuit"
            eye_data = ldp_sess.baseline_tuning[base_block][base_data][curr_set][plot_axis, :]
            use_ax = pursuit_ax
            # pursuit_ax.plot(time,
            #         ,
            #         color=colors[curr_set], label=line_label)
        else:
            raise ValueError("Unrecognized trial set {0}.".format(curr_set))

        use_ax.plot(time, fr, color=colors[curr_set], label=line_label)
        # use_ax.plot(time, eye_data, color=colors[curr_set], label=line_label)
        # eye_data[eye_data < 1.0] = np.inf
        # spk_per_deg = fr / eye_data
        # use_ax.plot(time, spk_per_deg, color=colors[curr_set], label=line_label)


    learn_ax.set_ylabel("Learning axis firing rate (spk/s)")
    learn_ax.set_xlabel("Time from target motion (ms)")
    learn_ax.set_title("Baseline tuning trials")
    learn_ax.legend()
    pursuit_ax.set_ylabel("Pursuit axis firing rate (spk/s)")
    pursuit_ax.set_xlabel("Time from target motion (ms)")
    pursuit_ax.set_title("Baseline tuning trials")
    pursuit_ax.legend()

    learn_ax.axvline(0, color='k')
    learn_ax.axhline(0, color='k')
    pursuit_ax.axvline(0, color='k')
    pursuit_ax.axhline(0, color='k')

    return learn_ax, pursuit_ax


def plot_instruction_firing_traces(ldp_sess, bin_edges, neuron_series_name,
                    time_window=None, base_block="StabTunePre"):
    """ """
    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    fig = plt.figure()
    inst_ax = plt.axes()
    time = np.arange(time_window[0], time_window[1])

    bin_fr_data = an.get_binned_mean_firing_trace(ldp_sess, bin_edges,
                                    neuron_series_name,
                                    time_window, blocks="Learning",
                                    trial_sets="instruction",
                                    bin_basis="instructed", return_t_inds=False)

    inst_ax = binned_mean_traces(bin_fr_data, t_vals=time,
                                ax=inst_ax, color='k', saturation=None)

    inst_ax.set_ylabel("Learning trial firing rate (spk/s)")
    inst_ax.set_xlabel("Time from target motion (ms)")
    inst_ax.set_title("Instruction trials")

    inst_ax.axvline(0, color='b')
    inst_ax.axhline(0, color='b')
    inst_ax.axvline(250, color='r')

    return inst_ax


def plot_tuning_probe_firing_traces(ldp_sess, bin_edges, neuron_series_name,
                                     time_window=None,
                                     base_block="StabTunePre"):
    """ """
    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    bin_fr_data = {}
    fig = plt.figure()
    learn_learn_ax = plt.axes()
    fig = plt.figure()
    learn_pursuit_ax = plt.axes()
    time = np.arange(time_window[0], time_window[1])
    p_col = {'pursuit': 'g', 'anti_pursuit': 'r', 'learning': 'g', 'anti_learning': 'r'}

    for curr_set in ldp_sess.four_dir_trial_sets:
        bin_fr_data[curr_set] = an.get_binned_mean_firing_trace(ldp_sess, bin_edges,
                                        neuron_series_name,
                                        time_window, blocks="Learning",
                                        trial_sets=curr_set,
                                        bin_basis="instructed", return_t_inds=False)

        plot_axis = ldp_sess.trial_set_base_axis[curr_set]
        if plot_axis == 0:
            learn_pursuit_ax = binned_mean_traces(bin_fr_data[curr_set], t_vals=time,
                                        ax=learn_pursuit_ax, color=p_col[curr_set], saturation=None)
        elif plot_axis == 1:
            learn_learn_ax = binned_mean_traces(bin_fr_data[curr_set], t_vals=time,
                                        ax=learn_learn_ax, color=p_col[curr_set], saturation=None)
        else:
            raise ValueError("Unrecognized trial set {0}.".format(curr_set))

    learn_learn_ax.set_ylabel("Learning axis firing rate (spk/s)")
    learn_learn_ax.set_xlabel("Time from target motion (ms)")
    learn_learn_ax.set_title("Pursuit axis probe tuning trials")
    learn_pursuit_ax.set_ylabel("Pursuit axis firing rate (spk/s)")
    learn_pursuit_ax.set_xlabel("Time from target motion (ms)")
    learn_pursuit_ax.set_title("Learning axis probe tuning trials")

    learn_pursuit_ax.axvline(0, color='b')
    learn_pursuit_ax.axhline(0, color='b')
    learn_pursuit_ax.axvline(250, color='r')
    learn_learn_ax.axvline(0, color='b')
    learn_learn_ax.axhline(0, color='b')
    learn_learn_ax.axvline(250, color='r')

    return learn_learn_ax, learn_pursuit_ax


def plot_post_tuning_firing_traces(ldp_sess, neuron_series_name, time_window=None,
                        base_block="StabTunePre"):
    """ """
    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    blocks = ["StabTunePost", "StabTuneWash"]
    p_col = {'StabTunePost': "r",
              'StabTuneWash': "g"
              }
    p_style = {'pursuit': '-', 'anti_pursuit': '--', 'learning': '-', 'anti_learning': '--'}
    p_b_labels = {'StabTunePost': "Post", 'StabTuneWash': 'Washout'}
    p_s_labels = {'pursuit': 'pursuit', 'anti_pursuit': 'anti-pursuit', 'learning': 'learning', 'anti_learning': 'anti-learning'}
    fig = plt.figure()
    post_learn_ax = plt.axes()
    fig = plt.figure()
    post_pursuit_ax = plt.axes()
    time = np.arange(time_window[0], time_window[1])

    for block in blocks:
        if ldp_sess.blocks[block] is None:
            continue
        for curr_set in ldp_sess.four_dir_trial_sets:
            fr = an.get_mean_firing_trace(ldp_sess,
                                                    neuron_series_name,
                                                    time_window,
                                                    block, curr_set)

            plot_axis = ldp_sess.trial_set_base_axis[curr_set]
            line_label = p_b_labels[block] + " " + p_s_labels[curr_set]
            if plot_axis == 0:
                post_pursuit_ax.plot(time, fr, color=p_col[block], linestyle=p_style[curr_set], label=line_label)
            elif plot_axis == 1:
                post_learn_ax.plot(time, fr, color=p_col[block], linestyle=p_style[curr_set], label=line_label)
            else:
                raise ValueError("Unrecognized trial set {0}.".format(curr_set))

    post_learn_ax.set_ylabel("Learning axis velocity (deg/s)")
    post_learn_ax.set_xlabel("Time from target motion (ms)")
    post_learn_ax.set_title("Pursuit axis tuning trials")
    post_learn_ax.legend()
    post_pursuit_ax.set_ylabel("Pursuit axis velocity (deg/s)")
    post_pursuit_ax.set_xlabel("Time from target motion (ms)")
    post_pursuit_ax.set_title("Learning axis tuning trials")
    post_pursuit_ax.legend()

    post_pursuit_ax.axvline(0, color='k')
    post_pursuit_ax.axhline(0, color='k')
    post_pursuit_ax.axvline(250, color='r')
    post_learn_ax.axvline(0, color='k')
    post_learn_ax.axhline(0, color='k')
    post_learn_ax.axvline(250, color='r')

    return post_learn_ax, post_pursuit_ax


def plot_washout_firing_traces(ldp_sess, bin_edges, neuron_series_name,
                    time_window=None, base_block="StabTunePre"):
    """ """
    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    fig = plt.figure()
    wash_ax = plt.axes()
    time = np.arange(time_window[0], time_window[1])

    bin_fr_data = an.get_binned_mean_firing_trace(ldp_sess, bin_edges,
                                    neuron_series_name,
                                    time_window, blocks="Washout",
                                    trial_sets="instruction",
                                    bin_basis="raw", return_t_inds=False)

    wash_ax = binned_mean_traces(bin_fr_data, t_vals=time,
                                ax=wash_ax, color='k', saturation=None)

    wash_ax.set_ylabel("Washout trial firing rate (spk/s)")
    wash_ax.set_xlabel("Time from target motion (ms)")
    wash_ax.set_title("Instruction trials")

    wash_ax.axvline(0, color='b')
    wash_ax.axhline(0, color='b')
    wash_ax.axvline(250, color='r')

    return wash_ax


def binned_mean_traces(bin_data, t_vals=None, ax=None, color='k',
                       linestyle='-', saturation=None, return_last_line=False):
    """
    """
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
    dark_step = (saturation[1] - saturation[0]) / len(bin_data)
    for data_trace in bin_data:
        if len(data_trace) == 0:
            darkness += dark_step
            continue
        if t_vals is None:
            last_line = ax.plot(data_trace, color=(darkness * color), linestyle=linestyle)
        else:
            last_line = ax.plot(t_vals, data_trace, color=(darkness * color), linestyle=linestyle)
        darkness += dark_step

    if return_last_line:
        return ax, last_line[-1]
    else:
        return ax


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
