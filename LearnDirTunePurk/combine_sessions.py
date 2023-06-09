import os
import re
import numpy as np
import matplotlib.pyplot as plt
from LearnDirTunePurk.build_session import create_behavior_session
import LearnDirTunePurk.analyze_behavior as ab
import LearnDirTunePurk.make_plots as plots



def get_all_mean_data(f_regex, directory, time_window, base_block,
        learn_bin_edges, probe_bin_edges, tuning_bin_edges,
        n_min_trials, n_bin_min_trials, rescale=False):
    """

    e.g f_regex = 'LearnDirTunePurk_Dandy_[0-9][0-9]_maestro.pickle'
        f_regex = 'LearnDirTunePurk_[A-Za-z]*_[0-9][0-9]_maestro.pickle'
    """

    """ NEED TO COUNT NUMBER OF SAMPLES PER AVERAGE/BIN !!! """
    if "Dandy" in f_regex:
        skip_file_nums = ["32", "36", "47"]
    elif "Yoda" in f_regex:
        if base_block != "StandTunePre":
            raise ValueError("Yoda only has StandTunePre tuning blocks!")
        skip_file_nums = ["06", "07", "08", "09", "10", "11", "12", "13", "14",
                          "15", "16", "17", "18", "19", "23", "31", "32", "35",
                          "37", "42"]
    else:
        raise ValueError("Unrecognized monkey in f_regex.")
    verbose = True
    four_dir_trial_sets = ["learning", "anti_learning", "pursuit", "anti_pursuit"]
    post_blocks = ["StabTunePost", "StabTuneWash"]
    # Generic bins for initializing below. Overwritten as needed
    wash_bin_edges = learn_bin_edges[learn_bin_edges <= 101]

    # Set output, starting with trial indices
    output_dict = {}
    output_dict['bin_inds_learn'] = learn_bin_edges[0:-1] + (np.diff(learn_bin_edges) / 2)
    output_dict['bin_inds_probe'] = probe_bin_edges[0:-1] + (np.diff(probe_bin_edges) / 2)
    output_dict['bin_inds_tune'] = tuning_bin_edges[0:-1] + (np.diff(tuning_bin_edges) / 2)
    output_dict['bin_inds_wash'] = wash_bin_edges[0:-1] + (np.diff(wash_bin_edges) / 2)

    pattern = re.compile(f_regex)
    filenames = [f for f in os.listdir(directory) if pattern.search(f) is not None]

    all_learning_bins_x_pos = [[] for x in range(0, len(learn_bin_edges)-1)]
    all_learning_bins_y_pos = [[] for x in range(0, len(learn_bin_edges)-1)]
    all_probe_bins_x_pos = {curr_set: [[] for x in range(0, len(probe_bin_edges)-1)] for curr_set in four_dir_trial_sets}
    all_probe_bins_y_pos = {curr_set: [[] for x in range(0, len(probe_bin_edges)-1)] for curr_set in four_dir_trial_sets}

    all_post_tuning_x_pos = {block: {curr_set: [[] for x in range(0, len(tuning_bin_edges)-1)] for curr_set in four_dir_trial_sets} for block in post_blocks}
    all_post_tuning_y_pos = {block: {curr_set: [[] for x in range(0, len(tuning_bin_edges)-1)] for curr_set in four_dir_trial_sets} for block in post_blocks}

    all_learning_bins_x_vel = [[] for x in range(0, len(learn_bin_edges)-1)]
    all_learning_bins_y_vel = [[] for x in range(0, len(learn_bin_edges)-1)]
    all_probe_bins_x_vel = {curr_set: [[] for x in range(0, len(probe_bin_edges)-1)] for curr_set in four_dir_trial_sets}
    all_probe_bins_y_vel = {curr_set: [[] for x in range(0, len(probe_bin_edges)-1)] for curr_set in four_dir_trial_sets}

    all_post_tuning_x_vel = {block: {curr_set: [[] for x in range(0, len(tuning_bin_edges)-1)] for curr_set in four_dir_trial_sets} for block in post_blocks}
    all_post_tuning_y_vel = {block: {curr_set: [[] for x in range(0, len(tuning_bin_edges)-1)] for curr_set in four_dir_trial_sets} for block in post_blocks}

    all_washout_bins_x_pos = [[] for x in range(0, len(wash_bin_edges)-1)]
    all_washout_bins_y_pos = [[] for x in range(0, len(wash_bin_edges)-1)]
    all_wash_bins_x_vel = [[] for x in range(0, len(wash_bin_edges)-1)]
    all_wash_bins_y_vel = [[] for x in range(0, len(wash_bin_edges)-1)]

    n_post_tune = []
    n_washout = []
    n_wash_tune = []
    n_files = 0
    for f in filenames:
        skip = False
        for skip_n in skip_file_nums:
            if skip_n in f:
                skip = True
                break
        if skip:
            print("SKIPPIN' file: ", f)
            continue
        print("Loading file:", f)

        ldp_sess = create_behavior_session(f, session_name=f.split("_maestro")[0],
                    existing_dir=directory, verbose=True, rotate=True)
        ldp_sess.set_baseline_averages(time_window)
        n_files += 1

        if verbose: print("Getting bin learning position data")
        bin_x_data, bin_y_data, bin_t_inds = ab.get_binned_mean_xy_traces(
                    ldp_sess, learn_bin_edges, "eye position", time_window,
                    blocks="Learning", trial_sets="instruction",
                    bin_basis="instructed", return_t_inds=True)
        bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(
                                    ldp_sess, base_block, "instruction", "eye position",
                                    bin_x_data, bin_y_data)
        # Append all the bin data
        for inds_ind, inds in enumerate(bin_t_inds):
            if len(inds) < n_bin_min_trials:
                # Not enough trials
                continue
            # Otherwise add it
            all_learning_bins_x_pos[inds_ind].append(bin_x_data[inds_ind])
            all_learning_bins_y_pos[inds_ind].append(bin_y_data[inds_ind])

        if verbose: print("Getting bin learning probe position data")
        for curr_set in four_dir_trial_sets:
            bin_x_data, bin_y_data, bin_t_inds = ab.get_binned_mean_xy_traces(
                                    ldp_sess, probe_bin_edges, "eye position", time_window,
                                    blocks="Learning", trial_sets=curr_set,
                                    bin_basis="instructed",
                                    return_t_inds=True)
            bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(
                                    ldp_sess, base_block, curr_set, "eye position",
                                    bin_x_data, bin_y_data)
            # Append all the bin data
            for inds_ind, inds in enumerate(bin_t_inds):
                if len(inds) < n_bin_min_trials:
                    # Not enough trials
                    continue
                # Otherwise add it
                all_probe_bins_x_pos[curr_set][inds_ind].append(bin_x_data[inds_ind])
                all_probe_bins_y_pos[curr_set][inds_ind].append(bin_y_data[inds_ind])

        if verbose: print("Getting post tuning position data")
        for block in post_blocks:
            if ldp_sess.blocks[block] is None:
                continue
            for curr_set in four_dir_trial_sets:
                bin_x_data, bin_y_data, bin_t_inds = ab.get_binned_mean_xy_traces(
                                        ldp_sess, tuning_bin_edges, "eye position", time_window,
                                        blocks=block, trial_sets=curr_set,
                                        bin_basis="order",
                                        return_t_inds=True)
                bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(
                                        ldp_sess, base_block, curr_set, "eye position",
                                        bin_x_data, bin_y_data)

                # Append all the bin data
                for inds_ind, inds in enumerate(bin_t_inds):
                    if len(inds) < n_bin_min_trials:
                        # Not enough trials
                        continue
                    # Otherwise add it
                    all_post_tuning_x_pos[block][curr_set][inds_ind].append(bin_x_data[inds_ind])
                    all_post_tuning_y_pos[block][curr_set][inds_ind].append(bin_y_data[inds_ind])

        if verbose: print("Getting bin learning velocity data")
        bin_xy_out = ab.get_binned_mean_xy_traces(ldp_sess,
                                    learn_bin_edges,
                                    "eye velocity", time_window, blocks="Learning",
                                    trial_sets="instruction",
                                    bin_basis="instructed", return_t_inds=True,
                                    rescale=rescale)
        if rescale:
            bin_x_data, bin_y_data, bin_t_inds, alpha_bin = bin_xy_out
        else:
            bin_x_data, bin_y_data, bin_t_inds = bin_xy_out
            alpha_bin = None
        bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(ldp_sess,
                                    base_block, "instruction", "eye velocity",
                                    bin_x_data, bin_y_data,
                                    alpha_scale_factors=alpha_bin)
        # Append all the bin data
        for inds_ind, inds in enumerate(bin_t_inds):
            if len(inds) < n_bin_min_trials:
                # Not enough trials
                continue
            # Otherwise add it
            all_learning_bins_x_vel[inds_ind].append(bin_x_data[inds_ind])
            all_learning_bins_y_vel[inds_ind].append(bin_y_data[inds_ind])

        if verbose: print("Getting bin learning probe velocity traces")
        for curr_set in four_dir_trial_sets:
            bin_xy_out = ab.get_binned_mean_xy_traces(
                                    ldp_sess, probe_bin_edges, "eye velocity", time_window,
                                    blocks="Learning", trial_sets=curr_set,
                                    bin_basis="instructed",
                                    return_t_inds=True, rescale=rescale)
            if rescale:
                bin_x_data, bin_y_data, bin_t_inds, alpha_bin = bin_xy_out
            else:
                bin_x_data, bin_y_data, bin_t_inds = bin_xy_out
                alpha_bin = None
            bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(
                                    ldp_sess, base_block, curr_set, "eye velocity",
                                    bin_x_data, bin_y_data,
                                    alpha_scale_factors=alpha_bin)
            # Append all the bin data
            for inds_ind, inds in enumerate(bin_t_inds):
                if len(inds) < n_bin_min_trials:
                    # Not enough trials
                    continue
                # Otherwise add it
                all_probe_bins_x_vel[curr_set][inds_ind].append(bin_x_data[inds_ind])
                all_probe_bins_y_vel[curr_set][inds_ind].append(bin_y_data[inds_ind])

        if verbose: print("Getting post tuning velocity traces")
        for block in post_blocks:
            if ldp_sess.blocks[block] is None:
                continue
            for curr_set in four_dir_trial_sets:
                bin_xy_out= ab.get_binned_mean_xy_traces(
                                        ldp_sess, tuning_bin_edges, "eye velocity", time_window,
                                        blocks=block, trial_sets=curr_set,
                                        bin_basis="order",
                                        return_t_inds=True, rescale=rescale)
                if rescale:
                    bin_x_data, bin_y_data, bin_t_inds, alpha_bin = bin_xy_out
                else:
                    bin_x_data, bin_y_data, bin_t_inds = bin_xy_out
                    alpha_bin = None
                bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(
                                        ldp_sess, base_block, curr_set, "eye velocity",
                                        bin_x_data, bin_y_data,
                                        alpha_scale_factors=alpha_bin)
                # Append all the bin data
                for inds_ind, inds in enumerate(bin_t_inds):
                    if len(inds) < n_bin_min_trials:
                        # Not enough trials
                        continue
                    # Otherwise add it
                    all_post_tuning_x_vel[block][curr_set][inds_ind].append(bin_x_data[inds_ind])
                    all_post_tuning_y_vel[block][curr_set][inds_ind].append(bin_y_data[inds_ind])

        if verbose: print("Getting bin washout position data")
        if ldp_sess.blocks['Washout'] is not None:
            bin_x_data, bin_y_data, bin_t_inds = ab.get_binned_mean_xy_traces(
                        ldp_sess, wash_bin_edges, "eye position", time_window,
                        blocks="Washout", trial_sets="instruction",
                        bin_basis="order", return_t_inds=True)
            bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(
                                        ldp_sess, base_block, "instruction", "eye position",
                                        bin_x_data, bin_y_data)
            # Append all the bin data
            for inds_ind, inds in enumerate(bin_t_inds):
                if len(inds) < n_bin_min_trials:
                    # Not enough trials
                    continue
                # Otherwise add it
                all_washout_bins_x_pos[inds_ind].append(bin_x_data[inds_ind])
                all_washout_bins_y_pos[inds_ind].append(bin_y_data[inds_ind])

        if verbose: print("Getting bin washout velocity traces")
        if ldp_sess.blocks['Washout'] is not None:
            bin_xy_out = ab.get_binned_mean_xy_traces(ldp_sess,
                                        wash_bin_edges,
                                        "eye velocity", time_window, blocks="Washout",
                                        trial_sets="instruction",
                                        bin_basis="order", return_t_inds=True,
                                        rescale=rescale)
            if rescale:
                bin_x_data, bin_y_data, bin_t_inds, alpha_bin = bin_xy_out
            else:
                bin_x_data, bin_y_data, bin_t_inds = bin_xy_out
                alpha_bin = None
            bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(ldp_sess,
                                        base_block, "instruction", "eye velocity",
                                        bin_x_data, bin_y_data,
                                        alpha_scale_factors=alpha_bin)
            # Append all the bin data
            for inds_ind, inds in enumerate(bin_t_inds):
                if len(inds) < n_bin_min_trials:
                    # Not enough trials
                    continue
                # Otherwise add it
                all_wash_bins_x_vel[inds_ind].append(bin_x_data[inds_ind])
                all_wash_bins_y_vel[inds_ind].append(bin_y_data[inds_ind])


        # if n_files > 2:
        #     break



    # Concatenate learning arrays
    for bin in range(0, len(all_learning_bins_x_pos)):
        if len(all_learning_bins_x_pos[bin]) == 0:
            continue
        all_learning_bins_x_pos[bin] = np.vstack(all_learning_bins_x_pos[bin])
        all_learning_bins_y_pos[bin] = np.vstack(all_learning_bins_y_pos[bin])
    output_dict['learn_bin_x_pos'] = all_learning_bins_x_pos
    output_dict['learn_bin_y_pos'] = all_learning_bins_y_pos

    # Concatenate probe arrays
    for curr_set in four_dir_trial_sets:
        for bin in range(0, len(all_probe_bins_x_pos[curr_set])):
            if len(all_probe_bins_x_pos[curr_set][bin]) == 0:
                continue
            all_probe_bins_x_pos[curr_set][bin] = np.vstack(all_probe_bins_x_pos[curr_set][bin])
            all_probe_bins_y_pos[curr_set][bin] = np.vstack(all_probe_bins_y_pos[curr_set][bin])
    output_dict['probe_bin_x_pos'] = all_probe_bins_x_pos
    output_dict['probe_bin_y_pos'] = all_probe_bins_y_pos

    # Concatenate post tuning arrays
    for block in post_blocks:
        for curr_set in four_dir_trial_sets:
            for bin in range(0, len(all_post_tuning_x_pos[block][curr_set])):
                if len(all_post_tuning_x_pos[block][curr_set][bin]) == 0:
                    continue
                all_post_tuning_x_pos[block][curr_set][bin] = np.vstack(all_post_tuning_x_pos[block][curr_set][bin])
                all_post_tuning_y_pos[block][curr_set][bin] = np.vstack(all_post_tuning_y_pos[block][curr_set][bin])
    output_dict['post_tuning_x_pos'] = all_post_tuning_x_pos
    output_dict['post_tuning_y_pos'] = all_post_tuning_y_pos

    # Concatenate learning arrays velocity
    for bin in range(0, len(all_learning_bins_x_vel)):
        if len(all_learning_bins_x_vel[bin]) == 0:
            continue
        all_learning_bins_x_vel[bin] = np.vstack(all_learning_bins_x_vel[bin])
        all_learning_bins_y_vel[bin] = np.vstack(all_learning_bins_y_vel[bin])
    output_dict['learn_bin_x_vel'] = all_learning_bins_x_vel
    output_dict['learn_bin_y_vel'] = all_learning_bins_y_vel

    # Concatenate probe arrays
    for curr_set in four_dir_trial_sets:
        for bin in range(0, len(all_probe_bins_x_vel[curr_set])):
            if len(all_probe_bins_x_vel[curr_set][bin]) == 0:
                continue
            all_probe_bins_x_vel[curr_set][bin] = np.vstack(all_probe_bins_x_vel[curr_set][bin])
            all_probe_bins_y_vel[curr_set][bin] = np.vstack(all_probe_bins_y_vel[curr_set][bin])
    output_dict['probe_bin_x_vel'] = all_probe_bins_x_vel
    output_dict['probe_bin_y_vel'] = all_probe_bins_y_vel

    # Concatenate post tuning arrays
    for block in post_blocks:
        for curr_set in four_dir_trial_sets:
            for bin in range(0, len(all_post_tuning_x_vel[block][curr_set])):
                if len(all_post_tuning_x_vel[block][curr_set][bin]) == 0:
                    continue
                all_post_tuning_x_vel[block][curr_set][bin] = np.vstack(all_post_tuning_x_vel[block][curr_set][bin])
                all_post_tuning_y_vel[block][curr_set][bin] = np.vstack(all_post_tuning_y_vel[block][curr_set][bin])
    output_dict['post_tuning_x_vel'] = all_post_tuning_x_vel
    output_dict['post_tuning_y_vel'] = all_post_tuning_y_vel

    # Concatenate washout arrays
    for bin in range(0, len(all_washout_bins_x_pos)):
        if len(all_washout_bins_x_pos[bin]) == 0:
            continue
        all_washout_bins_x_pos[bin] = np.vstack(all_washout_bins_x_pos[bin])
        all_washout_bins_y_pos[bin] = np.vstack(all_washout_bins_y_pos[bin])
    output_dict['wash_bin_x_pos'] = all_washout_bins_x_pos
    output_dict['wash_bin_y_pos'] = all_washout_bins_y_pos

    # Concatenate washout arrays
    for bin in range(0, len(all_wash_bins_x_vel)):
        if len(all_wash_bins_x_vel[bin]) == 0:
            continue
        all_wash_bins_x_vel[bin] = np.vstack(all_wash_bins_x_vel[bin])
        all_wash_bins_y_vel[bin] = np.vstack(all_wash_bins_y_vel[bin])
    output_dict['wash_bin_x_vel'] = all_wash_bins_x_vel
    output_dict['wash_bin_y_vel'] = all_wash_bins_y_vel


    return output_dict


""" **************** START PLOTTING FUNCTIONS***************************** """

def plot_comb_instruction_position_xy(ldp_sess, bin_edges, time_window=None,
                                 base_block="StabTunePre"):

    if time_window is None:
        time_window = ldp_sess.baseline_time_window
    fig = plt.figure()
    learn_ax = plt.axes()
    bin_x_data, bin_y_data = ab.get_binned_mean_xy_traces(
                ldp_sess, bin_edges, "eye position", time_window,
                blocks="Learning", trial_sets="instruction",
                bin_basis="instructed")
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


def plot_comb_tuning_probe_position_xy(data):
    four_dir_trial_sets = ["learning", "anti_learning", "pursuit", "anti_pursuit"]
    bin_means_x = {}
    bin_means_y = {}
    for curr_set in four_dir_trial_sets:
        bin_means_x[curr_set] = []
        bin_means_y[curr_set] = []
        for bin in range(0, len(data['probe_bin_x_pos'][curr_set])):
            if len(data['probe_bin_x_pos'][curr_set][bin]) == 0:
                continue
            bin_means_x[curr_set].append(np.nanmean(data['probe_bin_x_pos'][curr_set][bin], axis=0))
            bin_means_y[curr_set].append(np.nanmean(data['probe_bin_y_pos'][curr_set][bin], axis=0))

    p_col = {'pursuit': 'g', 'anti_pursuit': 'r', 'learning': 'g', 'anti_learning': 'r'}
    fig = plt.figure(figsize=(10,10))
    learn_ax = plt.axes()
    for curr_set in four_dir_trial_sets:
        learn_ax = plots.binned_mean_traces_2D(bin_means_x[curr_set], bin_means_y[curr_set],
                                        ax=learn_ax, color='k', saturation=None)
    learn_ax.set_ylabel("Learning axis eye position (deg)")
    learn_ax.set_xlabel("Pursuit axis eye position (deg)")
    learn_ax.set_title("Direction tuning probe trials")
    learn_ax.axvline(0, color='b')
    learn_ax.axhline(0, color='b')

    return learn_ax, fig


def plot_comb_post_tuning_position_xy(data):
    post_blocks = ["StabTunePost", "StabTuneWash"]
    four_dir_trial_sets = ["learning", "anti_learning", "pursuit", "anti_pursuit"]
    bin_means_x = {}
    bin_means_y = {}
    for block in post_blocks:
        bin_means_x[block] = {}
        bin_means_y[block] = {}
        for curr_set in four_dir_trial_sets:
            bin_means_x[block][curr_set] = []
            bin_means_y[block][curr_set] = []
            for bin in range(0, len(data['post_tuning_x_pos'][block][curr_set])):
                if len(data['post_tuning_x_pos'][block][curr_set][bin]) == 0:
                    continue
                bin_means_x[block][curr_set].append(np.nanmean(data['post_tuning_x_pos'][block][curr_set][bin], axis=0))
                bin_means_y[block][curr_set].append(np.nanmean(data['post_tuning_y_pos'][block][curr_set][bin], axis=0))

    p_b_labels = {'StabTunePost': "Post", 'StabTuneWash': 'Washout'}
    colors = {"StabTunePost": 'r', "StabTuneWash": 'g'}
    fig = plt.figure(figsize=(10, 10))
    post_tune_ax = plt.axes()
    for block in post_blocks:
        plotted_block = False
        for curr_set in four_dir_trial_sets:
            if len(data['post_tuning_x_pos'][block][curr_set][0]) == 0:
                continue
            plotted_block = True
            post_tune_ax, last_line = plots.binned_mean_traces_2D(bin_means_x[block][curr_set],
                                    bin_means_y[block][curr_set],
                                    ax=post_tune_ax, color=colors[block],
                                    saturation=None, return_last_line=True)
        if plotted_block:
            # Only mark the line if we plotted something for this block
            last_line.set_label(p_b_labels[block])

    post_tune_ax.legend()
    post_tune_ax.set_ylabel("Learning axis eye position (deg)")
    post_tune_ax.set_xlabel("Pursuit axis eye position (deg)")
    post_tune_ax.set_title("Direction tuning after learning")
    post_tune_ax.axvline(0, color='k')
    post_tune_ax.axhline(0, color='k')

    return post_tune_ax, fig


def plot_comb_instruction_velocity_traces(data, time_window):
    time = np.arange(time_window[0], time_window[1])
    bin_means_x = []
    bin_means_y = []
    for bin in range(0, len(data['learn_bin_x_vel'])):
        if len(data['learn_bin_x_vel'][bin]) == 0:
            continue
        bin_means_x.append(np.nanmean(data['learn_bin_x_vel'][bin], axis=0))
        bin_means_y.append(np.nanmean(data['learn_bin_y_vel'][bin], axis=0))
    inst_learn_fig = plt.figure(figsize=(12, 8))
    inst_learn_ax = plt.axes()
    instr_pursuit_fig = plt.figure(figsize=(12, 8))
    instr_pursuit_ax = plt.axes()

    inst_learn_ax = plots.binned_mean_traces(bin_means_y, t_vals=time,
                                ax=inst_learn_ax, color='k', saturation=None)
    instr_pursuit_ax = plots.binned_mean_traces(bin_means_x, t_vals=time,
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

    return inst_learn_ax, inst_learn_fig, instr_pursuit_ax, instr_pursuit_fig


def plot_comb_tuning_probe_velocity_traces(data, time_window):
    four_dir_trial_sets = ["learning", "anti_learning", "pursuit", "anti_pursuit"]
    trial_set_base_axis = {'learning': 0,
                       'anti_learning': 0,
                       'pursuit': 1,
                       'anti_pursuit': 1,
                       'instruction': 1
                        }
    time = np.arange(time_window[0], time_window[1])
    bin_means_x = {}
    bin_means_y = {}
    for curr_set in four_dir_trial_sets:
        bin_means_x[curr_set] = []
        bin_means_y[curr_set] = []
        for bin in range(0, len(data['probe_bin_x_vel'][curr_set])):
            if len(data['probe_bin_x_vel'][curr_set][bin]) == 0:
                continue
            bin_means_x[curr_set].append(np.nanmean(data['probe_bin_x_vel'][curr_set][bin], axis=0))
            bin_means_y[curr_set].append(np.nanmean(data['probe_bin_y_vel'][curr_set][bin], axis=0))
    learn_learn_fig = plt.figure(1, figsize=(12, 8))
    learn_learn_ax = plt.axes()
    learn_pursuit_fig = plt.figure(2, figsize=(12, 8))
    learn_pursuit_ax = plt.axes()
    time = np.arange(time_window[0], time_window[1])
    p_col = {'pursuit': 'g', 'anti_pursuit': 'r', 'learning': 'g', 'anti_learning': 'r'}
    p_b_labels = {'learning': "Learning", 'anti_learning': 'Anti-learning',
                  'pursuit': 'Pursuit', 'anti_pursuit': 'Anti-pursuit'}
    for curr_set in four_dir_trial_sets:
        # if curr_set != "anti_pursuit":
        #     continue
        plot_axis = trial_set_base_axis[curr_set]
        if plot_axis == 0:
            learn_pursuit_ax, learn_last_line = plots.binned_mean_traces(bin_means_x[curr_set], time,
                    ax=learn_pursuit_ax, color=p_col[curr_set], saturation=None,
                    return_last_line=True)
            learn_last_line.set_label(p_b_labels[curr_set])
        elif plot_axis == 1:
            learn_learn_ax, pursuit_last_line = plots.binned_mean_traces(bin_means_y[curr_set], time,
                    ax=learn_learn_ax, color=p_col[curr_set], saturation=None,
                    return_last_line=True)
            pursuit_last_line.set_label(p_b_labels[curr_set])
        else:
            raise ValueError("Unrecognized trial set {0}.".format(curr_set))

    learn_learn_ax.legend()
    learn_learn_ax.set_ylabel("Learning axis velocity (deg/s)")
    learn_learn_ax.set_xlabel("Time from target motion (ms)")
    learn_learn_ax.set_title("Pursuit axis probe tuning trials")
    learn_pursuit_ax.legend()
    learn_pursuit_ax.set_ylabel("Pursuit axis velocity (deg/s)")
    learn_pursuit_ax.set_xlabel("Time from target motion (ms)")
    learn_pursuit_ax.set_title("Learning axis probe tuning trials")

    learn_pursuit_ax.axvline(0, color='b')
    learn_pursuit_ax.axhline(0, color='b')
    learn_pursuit_ax.axvline(250, color='r')
    learn_learn_ax.axvline(0, color='b')
    learn_learn_ax.axhline(0, color='b')
    learn_learn_ax.axvline(250, color='r')

    return learn_learn_ax, learn_learn_fig, learn_pursuit_ax, learn_pursuit_fig


def plot_comb_post_tuning_velocity_traces(data, time_window):
    four_dir_trial_sets = ["learning", "anti_learning", "pursuit", "anti_pursuit"]
    post_blocks = ["StabTunePost", "StabTuneWash"]
    means_x = {}
    means_y = {}
    for block in post_blocks:
        means_x[block] = {}
        means_y[block] = {}
        for curr_set in four_dir_trial_sets:
            means_x[block][curr_set] = []
            means_y[block][curr_set] = []
            for bin in range(0, len(data['post_tuning_x_vel'][block][curr_set])):
                if len(data['post_tuning_x_vel'][block][curr_set][bin]) == 0:
                    continue
                means_x[block][curr_set].append(np.nanmean(data['post_tuning_x_vel'][block][curr_set][bin], axis=0))
                means_y[block][curr_set].append(np.nanmean(data['post_tuning_y_vel'][block][curr_set][bin], axis=0))
    trial_set_base_axis = {'learning': 0,
                       'anti_learning': 0,
                       'pursuit': 1,
                       'anti_pursuit': 1,
                       'instruction': 1
                        }
    p_col = {'StabTunePost': "r",
              'StabTuneWash': "g"
              }
    p_style = {'pursuit': '-', 'anti_pursuit': '--', 'learning': '-', 'anti_learning': '--'}
    p_b_labels = {'StabTunePost': "Post", 'StabTuneWash': 'Washout'}
    p_s_labels = {'pursuit': 'pursuit', 'anti_pursuit': 'anti-pursuit', 'learning': 'learning', 'anti_learning': 'anti-learning'}
    post_learn_fig = plt.figure(figsize=(12, 8))
    post_learn_ax = plt.axes()
    post_pursuit_fig = plt.figure(figsize=(12, 8))
    post_pursuit_ax = plt.axes()
    time = np.arange(time_window[0], time_window[1])
    for block in post_blocks:
        for curr_set in four_dir_trial_sets:
            if len(means_x[block][curr_set]) == 0:
                continue
            plot_axis = trial_set_base_axis[curr_set]
            line_label = p_b_labels[block] + " " + p_s_labels[curr_set]
            if plot_axis == 0:
                post_pursuit_ax, pp_last_line = plots.binned_mean_traces(means_x[block][curr_set],
                        time, ax=post_pursuit_ax, color=p_col[block],
                        linestyle=p_style[curr_set], saturation=None,
                        return_last_line=True)
                pp_last_line.set_label(line_label)
            elif plot_axis == 1:
                post_learn_ax, pl_last_line = plots.binned_mean_traces(means_y[block][curr_set],
                        time, ax=post_learn_ax, color=p_col[block],
                        linestyle=p_style[curr_set], saturation=None,
                        return_last_line=True)
                pl_last_line.set_label(line_label)
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

    return post_learn_ax, post_learn_fig, post_pursuit_ax, post_pursuit_fig


def plot_comb_washout_position_xy(data, time_window):
    wash_fig = plt.figure(figsize=(10,10))
    bin_means_x = []
    bin_means_y = []
    for bin in range(0, len(data['wash_bin_x_pos'])):
        if len(data['wash_bin_x_pos'][bin]) == 0:
            continue
        bin_means_x.append(np.nanmean(data['wash_bin_x_pos'][bin], axis=0))
        bin_means_y.append(np.nanmean(data['wash_bin_y_pos'][bin], axis=0))
    wash_ax = plt.axes()
    wash_ax = plots.binned_mean_traces_2D(bin_means_x, bin_means_y,
                            ax=wash_ax, color='k', saturation=None)

    wash_ax.set_ylabel("Learning axis eye position (deg)")
    wash_ax.set_xlabel("Pursuit axis eye position (deg)")
    wash_ax.set_title("Washout instruction trials")
    wash_ax.axvline(0, color='b')
    wash_ax.axhline(0, color='b')

    return wash_ax, wash_fig


def plot_comb_washout_velocity_traces(data, time_window):
    time = np.arange(time_window[0], time_window[1])
    bin_means_x = []
    bin_means_y = []
    for bin in range(0, len(data['wash_bin_x_vel'])):
        if len(data['wash_bin_x_vel'][bin]) == 0:
            continue
        bin_means_x.append(np.nanmean(data['wash_bin_x_vel'][bin], axis=0))
        bin_means_y.append(np.nanmean(data['wash_bin_y_vel'][bin], axis=0))
    wash_learn_fig = plt.figure(figsize=(12, 8))
    wash_learn_ax = plt.axes()
    wash_pursuit_fig = plt.figure(figsize=(12, 8))
    wash_pursuit_ax = plt.axes()

    wash_learn_ax = plots.binned_mean_traces(bin_means_y, t_vals=time,
                                ax=wash_learn_ax, color='k', saturation=None)
    wash_pursuit_ax = plots.binned_mean_traces(bin_means_x, t_vals=time,
                                ax=wash_pursuit_ax, color='k', saturation=None)

    wash_learn_ax.set_ylabel("Learning axis velocity (deg/s)")
    wash_learn_ax.set_xlabel("Time from target motion (ms)")
    wash_learn_ax.set_title("Washout trials")
    wash_pursuit_ax.set_ylabel("Pursuit axis velocity (deg/s)")
    wash_pursuit_ax.set_xlabel("Time from target motion (ms)")
    wash_pursuit_ax.set_title("Washout trials")

    wash_learn_ax.axvline(0, color='b')
    wash_learn_ax.axhline(0, color='b')
    wash_learn_ax.axvline(250, color='r')
    wash_pursuit_ax.axvline(0, color='b')
    wash_pursuit_ax.axhline(0, color='b')
    wash_pursuit_ax.axvline(250, color='r')

    return wash_learn_ax, wash_learn_fig, wash_pursuit_ax, wash_pursuit_fig


def plot_combined_learning_curve(data, time_window, learn_window, trial_set,
                                 extra_bin_spacing=1):
    four_dir_trial_sets = ["learning", "anti_learning", "pursuit", "anti_pursuit"]
    trial_set_base_axis = {'learning': 0,
                       'anti_learning': 0,
                       'pursuit': 1,
                       'anti_pursuit': 1,
                       'instruction': 1
                        }
    colors = {"LearnProbes": 'g',
              "StabTunePost": 'b',
              "Washout": 'r',
              "StabTuneWash": 'b'}
    ln_labels = {"LearnProbes": 'Learning block',
                 "StabTunePost": 'Post-learning tuning block',
                 "Washout": 'Washout block',
                 "StabTuneWash": 'Post-washout tuning block'}
    marker_size = 100
    lines = {key: None for key in colors.keys()}
    block_ends = {key: None for key in colors.keys()}
    avg_indices = [learn_window[0] - time_window[0], learn_window[1] - time_window[0]]
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes()

    data_axis = trial_set_base_axis[trial_set]
    # START learning trial probe data
    data_name = "probe_bin_x_vel" if data_axis == 0 else "probe_bin_y_vel"
    bin_means = []
    for bin in range(0, len(data[data_name][trial_set])):
        if len(data[data_name][trial_set][bin]) == 0:
            bin_means.append(np.nan)
        else:
            avg_trace = np.nanmean(data[data_name][trial_set][bin], axis=0)
            bin_means.append(np.nanmean(avg_trace[avg_indices[0]:avg_indices[1]]))
    lines['LearnProbes'] = ax.scatter(data['bin_inds_probe'], bin_means,
                            color=colors['LearnProbes'], s=marker_size)
    last_t_ind = data['bin_inds_probe'][-1] + ((data['bin_inds_probe'][-1] - data['bin_inds_probe'][-2])/2)
    block_ends['LearnProbes'] = last_t_ind

    # START post learning stab tuning data
    data_name = "post_tuning_x_vel" if data_axis == 0 else "post_tuning_y_vel"
    block_name = "StabTunePost"
    bin_means = []
    for bin in range(0, len(data[data_name][block_name][trial_set])):
        if len(data[data_name][block_name][trial_set][bin]) == 0:
            bin_means.append(np.nan)
        else:
            avg_trace = np.nanmean(data[data_name][block_name][trial_set][bin], axis=0)
            bin_means.append(np.nanmean(avg_trace[avg_indices[0]:avg_indices[1]]))
    curr_t_inds = extra_bin_spacing*data['bin_inds_tune'] + last_t_ind
    lines['StabTunePost'] = ax.scatter(curr_t_inds,
                    bin_means, color=colors['StabTunePost'], s=marker_size)
    last_t_ind = (curr_t_inds[-1]) + extra_bin_spacing*(
                    (data['bin_inds_tune'][-1] - data['bin_inds_tune'][-2])/2)
    block_ends['StabTunePost'] = last_t_ind

    # START washout trial instruction data
    # Since there are no probes, we need our own data axis selection
    curr_t_inds = extra_bin_spacing*data['bin_inds_wash'] + last_t_ind
    if trial_set == "pursuit":
        data_name = "wash_bin_x_vel" if data_axis == 0 else "wash_bin_y_vel"
        bin_means = []
        for bin in range(0, len(data[data_name])):
            if len(data[data_name][bin]) == 0:
                bin_means.append(np.nan)
            else:
                avg_trace = np.nanmean(data[data_name][bin], axis=0)
                bin_means.append(np.nanmean(avg_trace[avg_indices[0]:avg_indices[1]]))
        lines['Washout'] = ax.scatter(curr_t_inds,
                            bin_means, color=colors['Washout'], s=marker_size)
    last_t_ind = (curr_t_inds[-1]) + extra_bin_spacing*(
                    (data['bin_inds_wash'][-1] - data['bin_inds_wash'][-2])/2)
    block_ends['Washout'] = last_t_ind

    # START post washout stab tuning data
    data_name = "post_tuning_x_vel" if data_axis == 0 else "post_tuning_y_vel"
    block_name = "StabTuneWash"
    bin_means = []
    for bin in range(0, len(data[data_name][block_name][trial_set])):
        if len(data[data_name][block_name][trial_set][bin]) == 0:
            bin_means.append(np.nan)
        else:
            avg_trace = np.nanmean(data[data_name][block_name][trial_set][bin], axis=0)
            bin_means.append(np.nanmean(avg_trace[avg_indices[0]:avg_indices[1]]))
    curr_t_inds = extra_bin_spacing*data['bin_inds_tune'] + last_t_ind
    lines['StabTuneWash'] = ax.scatter(curr_t_inds,
                                bin_means, color=colors['StabTuneWash'], s=marker_size)
    last_t_ind = (curr_t_inds[-1]) + extra_bin_spacing*(
                    (data['bin_inds_tune'][-1] - data['bin_inds_tune'][-2])/2)
    block_ends['StabTuneWash'] = last_t_ind

    y_name = "Learning" if trial_set in ["pursuit", "anti_pursuit"] else "Pursuit"
    yl_string = y_name + " axis velocity (deg/s)"
    ax.set_ylabel(yl_string)
    ax.set_xlabel("Trial number")
    ax.set_title("Learned eye velocity over trials")

    for ln in lines.keys():
        if lines[ln] is not None:
            lines[ln].set_label(ln_labels[ln])
        ax.axvline(block_ends[ln], color=colors[ln])
    fig.legend()
    ax.axhline(0, color=[.5, .5, .5])

    return ax, fig


def plot_tan_combined_learning_curve(data, time_window, learn_window, trial_set,
                                 extra_bin_spacing=1, multiplier=1):
    four_dir_trial_sets = ["learning", "anti_learning", "pursuit", "anti_pursuit"]
    trial_set_base_axis = {'learning': 0,
                       'anti_learning': 0,
                       'pursuit': 1,
                       'anti_pursuit': 1,
                       'instruction': 1
                        }
    colors = {"LearnProbes": 'g',
              "StabTunePost": 'b',
              "Washout": 'r',
              "StabTuneWash": 'b'}
    ln_labels = {"LearnProbes": 'Learning block',
                 "StabTunePost": 'Post-learning tuning block',
                 "Washout": 'Washout block',
                 "StabTuneWash": 'Post-washout tuning block'}
    marker_size = 100
    lines = {key: None for key in colors.keys()}
    block_ends = {key: None for key in colors.keys()}
    avg_indices = [learn_window[0] - time_window[0], learn_window[1] - time_window[0]]
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes()

    data_axis = trial_set_base_axis[trial_set]
    # START learning trial probe data
    if data_axis == 0:
        num_data = "probe_bin_x_pos"
        den_data = "probe_bin_y_pos"
    else:
        num_data = "probe_bin_y_pos"
        den_data = "probe_bin_x_pos"
    bin_means = []
    for bin in range(0, len(data[num_data][trial_set])):
        if len(data[num_data][trial_set][bin]) == 0:
            bin_means.append(np.nan)
        else:
            # tan_data = data[num_data][trial_set][bin] / np.abs(data[den_data][trial_set][bin])
            # tan_data[~np.isfinite(tan_data)] = np.nan
            # tan_data[np.abs(tan_data) > 5] = np.nan
            # tan_data = np.arctan(tan_data)
            # avg_trace = multiplier * np.nanmean(tan_data, axis=0)
            # bin_means.append(np.nanmean(avg_trace[avg_indices[0]:avg_indices[1]]))
            num_avg_trace = np.nanmean(data[num_data][trial_set][bin], axis=0)
            den_avg_trace = np.nanmean(np.abs(data[den_data][trial_set][bin]), axis=0)
            avg_tan = ( (num_avg_trace[avg_indices[1]] - num_avg_trace[avg_indices[0]]) /
                        (den_avg_trace[avg_indices[1]] - den_avg_trace[avg_indices[0]]) )
            bin_means.append(avg_tan)
    lines['LearnProbes'] = ax.scatter(data['bin_inds_probe'], bin_means,
                            color=colors['LearnProbes'], s=marker_size)
    last_t_ind = data['bin_inds_probe'][-1] + ((data['bin_inds_probe'][-1] - data['bin_inds_probe'][-2])/2)
    block_ends['LearnProbes'] = last_t_ind

    # START post learning stab tuning data
    if data_axis == 0:
        num_data = "post_tuning_x_pos"
        den_data = "post_tuning_y_pos"
    else:
        num_data = "post_tuning_y_pos"
        den_data = "post_tuning_x_pos"
    block_name = "StabTunePost"
    bin_means = []
    for bin in range(0, len(data[num_data][block_name][trial_set])):
        if len(data[num_data][block_name][trial_set][bin]) == 0:
            bin_means.append(np.nan)
        else:
            # tan_data = data[num_data][block_name][trial_set][bin] / np.abs(data[den_data][block_name][trial_set][bin])
            # tan_data[~np.isfinite(tan_data)] = np.nan
            # tan_data[np.abs(tan_data) > 5] = np.nan
            # tan_data = np.arctan(tan_data)
            # avg_trace = multiplier * np.nanmean(tan_data, axis=0)
            # bin_means.append(np.nanmean(avg_trace[avg_indices[0]:avg_indices[1]]))
            num_avg_trace = np.nanmean(data[num_data][block_name][trial_set][bin], axis=0)
            den_avg_trace = np.nanmean(np.abs(data[den_data][block_name][trial_set][bin]), axis=0)
            avg_tan = ( (num_avg_trace[avg_indices[1]] - num_avg_trace[avg_indices[0]]) /
                        (den_avg_trace[avg_indices[1]] - den_avg_trace[avg_indices[0]]) )
            bin_means.append(avg_tan)
    curr_t_inds = extra_bin_spacing*data['bin_inds_tune'] + last_t_ind
    lines['StabTunePost'] = ax.scatter(curr_t_inds,
                    bin_means, color=colors['StabTunePost'], s=marker_size)
    last_t_ind = (curr_t_inds[-1]) + extra_bin_spacing*(
                    (data['bin_inds_tune'][-1] - data['bin_inds_tune'][-2])/2)
    block_ends['StabTunePost'] = last_t_ind

    # START washout trial instruction data
    # Since there are no probes, we need our own data axis selection
    curr_t_inds = extra_bin_spacing*data['bin_inds_wash'] + last_t_ind
    if trial_set == "pursuit":
        if data_axis == 0:
            num_data = "wash_bin_x_pos"
            den_data = "wash_bin_y_pos"
        else:
            num_data = "wash_bin_y_pos"
            den_data = "wash_bin_x_pos"
        bin_means = []
        for bin in range(0, len(data[num_data])):
            if len(data[num_data][bin]) == 0:
                bin_means.append(np.nan)
            else:
                # tan_data = data[num_data][bin] / np.abs(data[den_data][bin])
                # tan_data[~np.isfinite(tan_data)] = np.nan
                # tan_data[np.abs(tan_data) > 5] = np.nan
                # tan_data = np.arctan(tan_data)
                # avg_trace = multiplier * np.nanmean(tan_data, axis=0)
                # bin_means.append(np.nanmean(avg_trace[avg_indices[0]:avg_indices[1]]))
                num_avg_trace = np.nanmean(data[num_data][bin], axis=0)
                den_avg_trace = np.nanmean(np.abs(data[den_data][bin]), axis=0)
                avg_tan = ( (num_avg_trace[avg_indices[1]] - num_avg_trace[avg_indices[0]]) /
                            (den_avg_trace[avg_indices[1]] - den_avg_trace[avg_indices[0]]) )
                bin_means.append(avg_tan)
        lines['Washout'] = ax.scatter(curr_t_inds,
                            bin_means, color=colors['Washout'], s=marker_size)
    last_t_ind = (curr_t_inds[-1]) + extra_bin_spacing*(
                    (data['bin_inds_wash'][-1] - data['bin_inds_wash'][-2])/2)
    block_ends['Washout'] = last_t_ind

    # START post washout stab tuning data
    if data_axis == 0:
        num_data = "post_tuning_x_pos"
        den_data = "post_tuning_y_pos"
    else:
        num_data = "post_tuning_y_pos"
        den_data = "post_tuning_x_pos"
    block_name = "StabTuneWash"
    bin_means = []
    for bin in range(0, len(data[num_data][block_name][trial_set])):
        if len(data[num_data][block_name][trial_set][bin]) == 0:
            bin_means.append(np.nan)
        else:
            # tan_data = data[num_data][block_name][trial_set][bin] / np.abs(data[den_data][block_name][trial_set][bin])
            # tan_data[~np.isfinite(tan_data)] = np.nan
            # tan_data[np.abs(tan_data) > 5] = np.nan
            # tan_data = np.arctan(tan_data)
            # avg_trace = multiplier * np.nanmean(tan_data, axis=0)
            # bin_means.append(np.nanmean(avg_trace[avg_indices[0]:avg_indices[1]]))
            num_avg_trace = np.nanmean(data[num_data][block_name][trial_set][bin], axis=0)
            den_avg_trace = np.nanmean(np.abs(data[den_data][block_name][trial_set][bin]), axis=0)
            avg_tan = ( (num_avg_trace[avg_indices[1]] - num_avg_trace[avg_indices[0]]) /
                        (den_avg_trace[avg_indices[1]] - den_avg_trace[avg_indices[0]]) )
            bin_means.append(avg_tan)
    curr_t_inds = extra_bin_spacing*data['bin_inds_tune'] + last_t_ind
    lines['StabTuneWash'] = ax.scatter(curr_t_inds,
                                bin_means, color=colors['StabTuneWash'], s=marker_size)
    last_t_ind = (curr_t_inds[-1]) + extra_bin_spacing*(
                    (data['bin_inds_tune'][-1] - data['bin_inds_tune'][-2])/2)
    block_ends['StabTuneWash'] = last_t_ind

    y_name = "Learning" if trial_set in ["pursuit", "anti_pursuit"] else "Pursuit"
    yl_string = y_name + " axis velocity (deg/s)"
    ax.set_ylabel(yl_string)
    ax.set_xlabel("Trial number")
    ax.set_title("Learned eye velocity over trials")

    for ln in lines.keys():
        if lines[ln] is not None:
            lines[ln].set_label(ln_labels[ln])
        ax.axvline(block_ends[ln], color=colors[ln])
    fig.legend()
    ax.axhline(0, color=[.5, .5, .5])

    return ax, fig
