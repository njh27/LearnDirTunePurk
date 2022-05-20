import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c_look
import LearnDirTunePurk.analyze_behavior as ab



def show_all_eye_plots(ldp_sess, base_block="StabTunePre", exp_bins=False):
    base_t_ax = baseline_tuning_2D(ldp_sess, base_block, "eye position", colors='k')

    t_max = 700
    step_size = 20
    bin_edges = np.arange(-1, t_max+step_size, step_size)
    if exp_bins:
        bin_edges = np.hstack((0, 2*np.exp(np.arange(1, np.log(t_max/2)+1, 1))))
    learn_ax = plot_instruction_position_xy(ldp_sess, bin_edges, time_window=None, base_block=base_block)

    t_max = 700
    step_size = 100
    bin_edges = np.arange(-1, t_max+step_size, step_size)
    if exp_bins:
        bin_edges = np.hstack((0, 2*np.exp(np.arange(1, np.log(t_max/2)+1, 1))))
    learn_ax = plot_tuning_probe_position_xy(ldp_sess, bin_edges, time_window=None,
                                      base_block=base_block, colors='k',
                                      saturation=None)

    post_tune_ax = plot_post_tuning_position_xy(ldp_sess, time_window=None, base_block=base_block)

    base_t_ax = baseline_tuning_2D(ldp_sess, base_block, "eye velocity", colors='k')

    p_col = {'pursuit': 'g', 'anti_pursuit': 'r', 'learning': 'g', 'anti_learning': 'r'}
    base_learn_ax, base_pursuit_ax = baseline_tuning(ldp_sess, base_block, "eye velocity", colors=p_col)

    t_max = 700
    step_size = 20
    bin_edges = np.arange(-1, t_max+step_size, step_size)
    if exp_bins:
        bin_edges = np.hstack((0, 2*np.exp(np.arange(1, np.log(t_max/2)+1, 1))))
    inst_learn_ax, instr_pursuit_ax = plot_instruction_velocity_traces(ldp_sess, bin_edges, time_window=None,
                                                                                      base_block=base_block)

    t_max = 700
    step_size = 100
    bin_edges = np.arange(-1, t_max+step_size, step_size)
    if exp_bins:
        bin_edges = np.hstack((0, 2*np.exp(np.arange(1, np.log(t_max/2)+1, 1))))
    learn_learn_ax, learn_pursuit_ax = plot_tuning_probe_velocity_traces(ldp_sess, bin_edges, time_window=None,
                                                                                        base_block=base_block)

    post_learn_ax, post_pursuit_ax = plot_post_tuning_velocity_traces(ldp_sess, time_window=None, base_block=base_block)

    if ldp_sess.blocks['Washout'] is not None:
        t_max = ldp_sess.blocks['Washout'][1]
        step_size = 10
        bin_edges = np.arange(ldp_sess.blocks['Washout'][0], t_max+step_size, step_size)
        if exp_bins:
            bin_edges = np.hstack((0, 2*np.exp(np.arange(1, np.log(t_max/2)+1, 1))))
        learn_ax = plot_washout_position_xy(ldp_sess, bin_edges, time_window=None, base_block=base_block)

        t_max = ldp_sess.blocks['Washout'][1]
        step_size = 10
        bin_edges = np.arange(ldp_sess.blocks['Washout'][0], t_max+step_size, step_size)
        if exp_bins:
            bin_edges = np.hstack((0, 2*np.exp(np.arange(1, np.log(t_max/2)+1, 1))))
        inst_learn_ax, instr_pursuit_ax = plot_washout_velocity_traces(ldp_sess, bin_edges, time_window=None,
                                                                                  base_block=base_block)

    return None


def plot_instruction_position_xy(ldp_sess, bin_edges, time_window=None,
                                 base_block="StabTunePre"):

    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    fig = plt.figure()
    learn_ax = plt.axes()
    bin_x_data, bin_y_data = ab.get_binned_mean_xy_traces(
                ldp_sess, bin_edges, "eye position", time_window,
                blocks="Learning", trial_sets="instruction", rotate=True,
                bin_by_instructed=True)
    bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(
                                ldp_sess, base_block, "instruction", "eye position",
                                bin_x_data, bin_y_data)
    learn_ax = binned_mean_traces_2D(bin_x_data, bin_y_data,
                            ax=learn_ax, color='k', saturation=None)

    learn_ax.set_ylabel("Learning axis eye position (deg)")
    learn_ax.set_xlabel("Pursuit axis eye position (deg)")
    learn_ax.set_title("Instruction trials")
    learn_ax.axvline(0, color='b')
    learn_ax.axhline(0, color='b')

    return learn_ax


def plot_tuning_probe_position_xy(ldp_sess, bin_edges, time_window=None,
                                  base_block="StabTunePre", colors='k',
                                  saturation=None):

    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    bin_x_data = {}
    bin_y_data = {}
    fig = plt.figure()
    learn_ax = plt.axes()
    for curr_set in ldp_sess.four_dir_trial_sets:
        bin_x_data[curr_set], bin_y_data[curr_set] = ab.get_binned_mean_xy_traces(
                                ldp_sess, bin_edges, "eye position", time_window,
                                blocks="Learning", trial_sets=curr_set,
                                rotate=True, bin_by_instructed=True)
        bin_x_data[curr_set], bin_y_data[curr_set] = ab.subtract_baseline_tuning_binned(
                                ldp_sess, base_block, curr_set, "eye position",
                                bin_x_data[curr_set], bin_y_data[curr_set])
        learn_ax = binned_mean_traces_2D(bin_x_data[curr_set], bin_y_data[curr_set],
                                    ax=learn_ax, color=colors, saturation=saturation)

    learn_ax.set_ylabel("Learning axis eye position (deg)")
    learn_ax.set_xlabel("Pursuit axis eye position (deg)")
    learn_ax.set_title("Direction tuning probe trials")
    learn_ax.axvline(0, color='b')
    learn_ax.axhline(0, color='b')

    save_name = "/Users/nathanhall/onedrive - duke university/sync/LearnDirTunePurk/Data/Maestro/" + ldp_sess.session_name + ".pdf"
    plt.savefig(save_name)

    return learn_ax


def plot_post_tuning_position_xy(ldp_sess, time_window=None, base_block="StabTunePre"):
    """
    """
    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    blocks = ["StabTunePost", "StabTuneWash"]
    colors = {'StabTunePost': "r",
              'StabTuneWash': "g"
              }
    p_b_labels = {'StabTunePost': "Post", 'StabTuneWash': 'Washout'}
    fig = plt.figure()
    post_tune_ax = plt.axes()
    for block in blocks:
        if ldp_sess.blocks[block] is None:
            continue
        for curr_set in ldp_sess.four_dir_trial_sets:
            x, y = ab.get_mean_xy_traces(ldp_sess, "eye position", time_window,
                        blocks=block, trial_sets=curr_set, rotate=True)
            x, y = ab.subtract_baseline_tuning(ldp_sess, base_block, curr_set,
                                               "eye position", x, y)
            last_line = post_tune_ax.scatter(x, y, color=colors[block])
        last_line.set_label(p_b_labels[block])

    post_tune_ax.legend()
    post_tune_ax.set_ylabel("Learning axis eye position (deg)")
    post_tune_ax.set_xlabel("Pursuit axis eye position (deg)")
    post_tune_ax.set_title("Direction tuning after learning")
    post_tune_ax.axvline(0, color='k')
    post_tune_ax.axhline(0, color='k')

    return post_tune_ax


def plot_instruction_velocity_traces(ldp_sess, bin_edges, time_window=None,
                                     base_block="StabTunePre"):
    """ """
    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    fig = plt.figure()
    inst_learn_ax = plt.axes()
    fig = plt.figure()
    instr_pursuit_ax = plt.axes()
    time = np.arange(time_window[0], time_window[1])

    bin_x_data, bin_y_data = ab.get_binned_mean_xy_traces(ldp_sess, bin_edges,
                                "eye velocity", time_window, blocks="Learning",
                                trial_sets="instruction", rotate=True,
                                bin_by_instructed=True)
    bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(ldp_sess,
                                base_block, "instruction", "eye velocity",
                                bin_x_data, bin_y_data)

    inst_learn_ax = binned_mean_traces(bin_y_data, t_vals=time,
                                ax=inst_learn_ax, color='k', saturation=None)
    instr_pursuit_ax = binned_mean_traces(bin_x_data, t_vals=time,
                                ax=instr_pursuit_ax, color='k', saturation=None)

    inst_learn_ax.set_ylabel("Learning axis velocity (deg/s)")
    inst_learn_ax.set_xlabel("Time from target motion (ms)")
    inst_learn_ax.set_title("Instruction trials")
    instr_pursuit_ax.set_ylabel("Pursuit axis velocity (deg/s)")
    instr_pursuit_ax.set_xlabel("Time from target motion (ms)")
    instr_pursuit_ax.set_title("Instruction trials")

    inst_learn_ax.axvline(0, color='b')
    inst_learn_ax.axhline(0, color='b')
    inst_learn_ax.axvline(250, color='r')
    instr_pursuit_ax.axvline(0, color='b')
    instr_pursuit_ax.axhline(0, color='b')
    instr_pursuit_ax.axvline(250, color='r')

    return inst_learn_ax, instr_pursuit_ax


def plot_tuning_probe_velocity_traces(ldp_sess, bin_edges, time_window=None,
                                     base_block="StabTunePre"):
    """ """
    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    bin_x_data = {}
    bin_y_data = {}
    fig = plt.figure()
    learn_learn_ax = plt.axes()
    fig = plt.figure()
    learn_pursuit_ax = plt.axes()
    time = np.arange(time_window[0], time_window[1])
    p_col = {'pursuit': 'g', 'anti_pursuit': 'r', 'learning': 'g', 'anti_learning': 'r'}

    for curr_set in ldp_sess.four_dir_trial_sets:
        bin_x_data[curr_set], bin_y_data[curr_set] = ab.get_binned_mean_xy_traces(
                                ldp_sess, bin_edges, "eye velocity", time_window,
                                blocks="Learning", trial_sets=curr_set,
                                rotate=True, bin_by_instructed=True)
        bin_x_data[curr_set], bin_y_data[curr_set] = ab.subtract_baseline_tuning_binned(
                                ldp_sess, base_block, curr_set, "eye velocity",
                                bin_x_data[curr_set], bin_y_data[curr_set])
        plot_axis = ldp_sess.trial_set_base_axis[curr_set]
        if plot_axis == 0:
            learn_pursuit_ax = binned_mean_traces(bin_x_data[curr_set], time,
                    ax=learn_pursuit_ax, color=p_col[curr_set], saturation=None)
        elif plot_axis == 1:
            learn_learn_ax = binned_mean_traces(bin_y_data[curr_set], time,
                    ax=learn_learn_ax, color=p_col[curr_set], saturation=None)
        else:
            raise ValueError("Unrecognized trial set {0}.".format(curr_set))

    learn_learn_ax.set_ylabel("Learning axis velocity (deg/s)")
    learn_learn_ax.set_xlabel("Time from target motion (ms)")
    learn_learn_ax.set_title("Pursuit axis probe tuning trials")
    learn_pursuit_ax.set_ylabel("Pursuit axis velocity (deg/s)")
    learn_pursuit_ax.set_xlabel("Time from target motion (ms)")
    learn_pursuit_ax.set_title("Learning axis probe tuning trials")

    learn_pursuit_ax.axvline(0, color='b')
    learn_pursuit_ax.axhline(0, color='b')
    learn_pursuit_ax.axvline(250, color='r')
    learn_learn_ax.axvline(0, color='b')
    learn_learn_ax.axhline(0, color='b')
    learn_learn_ax.axvline(250, color='r')

    return learn_learn_ax, learn_pursuit_ax


def plot_post_tuning_velocity_traces(ldp_sess, time_window=None, base_block="StabTunePre"):
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
            x, y = ab.get_mean_xy_traces(ldp_sess, "eye velocity", time_window,
                                blocks=block, trial_sets=curr_set, rotate=True)
            x, y = ab.subtract_baseline_tuning(ldp_sess, base_block, curr_set,
                                               "eye velocity", x, y)

            plot_axis = ldp_sess.trial_set_base_axis[curr_set]
            line_label = p_b_labels[block] + " " + p_s_labels[curr_set]
            if plot_axis == 0:
                post_pursuit_ax.plot(time, x, color=p_col[block], linestyle=p_style[curr_set], label=line_label)
            elif plot_axis == 1:
                post_learn_ax.plot(time, y, color=p_col[block], linestyle=p_style[curr_set], label=line_label)
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


def plot_washout_position_xy(ldp_sess, bin_edges, time_window=None,
                                 base_block="StabTunePre"):

    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    fig = plt.figure()
    learn_ax = plt.axes()
    bin_x_data, bin_y_data = ab.get_binned_mean_xy_traces(
                ldp_sess, bin_edges, "eye position", time_window,
                blocks="Washout", trial_sets="instruction", rotate=True,
                bin_by_instructed=False)
    bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(
                                ldp_sess, base_block, "instruction", "eye position",
                                bin_x_data, bin_y_data)
    learn_ax = binned_mean_traces_2D(bin_x_data, bin_y_data,
                            ax=learn_ax, color='k', saturation=None)

    learn_ax.set_ylabel("Learning axis eye position (deg)")
    learn_ax.set_xlabel("Pursuit axis eye position (deg)")
    learn_ax.set_title("Instruction trials")
    learn_ax.axvline(0, color='b')
    learn_ax.axhline(0, color='b')

    return learn_ax


def plot_washout_velocity_traces(ldp_sess, bin_edges, time_window=None,
                                     base_block="StabTunePre"):
    """ """
    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    fig = plt.figure()
    inst_learn_ax = plt.axes()
    fig = plt.figure()
    instr_pursuit_ax = plt.axes()
    time = np.arange(time_window[0], time_window[1])

    bin_x_data, bin_y_data = ab.get_binned_mean_xy_traces(ldp_sess, bin_edges,
                                "eye velocity", time_window, blocks="Washout",
                                trial_sets="instruction", rotate=True,
                                bin_by_instructed=False)
    bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(ldp_sess,
                                base_block, "instruction", "eye velocity",
                                bin_x_data, bin_y_data)

    inst_learn_ax = binned_mean_traces(bin_y_data, t_vals=time,
                                ax=inst_learn_ax, color='k', saturation=None)
    instr_pursuit_ax = binned_mean_traces(bin_x_data, t_vals=time,
                                ax=instr_pursuit_ax, color='k', saturation=None)

    inst_learn_ax.set_ylabel("Learning axis velocity (deg/s)")
    inst_learn_ax.set_xlabel("Time from target motion (ms)")
    inst_learn_ax.set_title("Washout trials")
    instr_pursuit_ax.set_ylabel("Pursuit axis velocity (deg/s)")
    instr_pursuit_ax.set_xlabel("Time from target motion (ms)")
    instr_pursuit_ax.set_title("Washout trials")

    inst_learn_ax.axvline(0, color='b')
    inst_learn_ax.axhline(0, color='b')
    inst_learn_ax.axvline(250, color='r')
    instr_pursuit_ax.axvline(0, color='b')
    instr_pursuit_ax.axhline(0, color='b')
    instr_pursuit_ax.axvline(250, color='r')

    return inst_learn_ax, instr_pursuit_ax


def baseline_tuning(ldp_sess, base_block, base_data, colors=None):
    """ Plots the 4 direction 2D tuning for the block and data sepcified from
    the baseline sets stored in ldp_sess. """

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

        if "learn" in curr_set:
            if 'anti' in curr_set:
                line_label = "Anti-learning"
            else:
                line_label = "Learning"
            learn_ax.plot(time,
                    ldp_sess.baseline_tuning[base_block][base_data][curr_set][plot_axis, :],
                    color=colors[curr_set], label=line_label)
        elif "pursuit" in curr_set:
            if 'anti' in curr_set:
                line_label = "Anti-pursuit"
            else:
                line_label = "Pursuit"
            pursuit_ax.plot(time,
                    ldp_sess.baseline_tuning[base_block][base_data][curr_set][plot_axis, :],
                    color=colors[curr_set], label=line_label)
        else:
            raise ValueError("Unrecognized trial set {0}.".format(curr_set))

    learn_ax.set_ylabel("Learning axis velocity (deg/s)")
    learn_ax.set_xlabel("Time from target motion (ms)")
    learn_ax.set_title("Baseline tuning trials")
    learn_ax.legend()
    pursuit_ax.set_ylabel("Pursuit axis velocity (deg/s)")
    pursuit_ax.set_xlabel("Time from target motion (ms)")
    pursuit_ax.set_title("Baseline tuning trials")
    pursuit_ax.legend()

    learn_ax.axvline(0, color='k')
    learn_ax.axhline(0, color='k')
    pursuit_ax.axvline(0, color='k')
    pursuit_ax.axhline(0, color='k')

    return learn_ax, pursuit_ax


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

    return ax


def binned_mean_traces_2D(bin_x_data, bin_y_data, ax=None, color='k',
                          saturation=None):
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


def binned_mean_traces(bin_data, t_vals=None, ax=None, color='k',
                       saturation=None):
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
            ax.plot(data_trace, color=(darkness * color))
        else:
            ax.plot(t_vals, data_trace, color=(darkness * color))
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
