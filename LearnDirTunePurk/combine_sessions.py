import os
import re
import numpy as np
from LearnDirTunePurk.build_session import create_behavior_session
import LearnDirTunePurk.analyze_behavior as ab



def get_all_mean_data(f_regex, directory, time_window, base_block,
        learn_bin_edges, probe_bin_edges, n_min_trials, n_bin_min_trials):
    """

    e.g f_regex = 'LearnDirTunePurk_Dandy_[0-9][0-9]_maestro.pickle'
        f_regex = 'LearnDirTunePurk_[A-Za-z]*_[0-9][0-9]_maestro.pickle'
    """

    """ NEED TO COUNT NUMBER OF SAMPLES PER AVERAGE/BIN !!! """
    skip_file_nums = ["14", "32", "36", "39", "47"]
    verbose = True
    four_dir_trial_sets = ["learning", "anti_learning", "pursuit", "anti_pursuit"]
    post_blocks = ["StabTunePost", "StabTuneWash"]

    pattern = re.compile(f_regex)
    filenames = [f for f in os.listdir(directory) if pattern.search(f) is not None]

    all_learning_bins_x_pos = [[] for x in range(0, len(learn_bin_edges)-1)]
    all_learning_bins_y_pos = [[] for x in range(0, len(learn_bin_edges)-1)]
    all_probe_bins_x_pos = {curr_set: [[] for x in range(0, len(probe_bin_edges)-1)] for curr_set in four_dir_trial_sets}
    all_probe_bins_y_pos = {curr_set: [[] for x in range(0, len(probe_bin_edges)-1)] for curr_set in four_dir_trial_sets}
    all_post_tuning_x_pos = {block: {curr_set: [] for curr_set in four_dir_trial_sets} for block in post_blocks}
    all_post_tuning_y_pos = {block: {curr_set: [] for curr_set in four_dir_trial_sets} for block in post_blocks}
    all_learning_bins_x_vel = [[] for x in range(0, len(learn_bin_edges)-1)]
    all_learning_bins_y_vel = [[] for x in range(0, len(learn_bin_edges)-1)]

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
                    existing_dir=directory, verbose=False)
        ldp_sess.set_baseline_averages(time_window)
        n_files += 1

        if verbose: print("Getting bin learning position data")
        bin_x_data, bin_y_data, bin_t_inds = ab.get_binned_mean_xy_traces(
                    ldp_sess, learn_bin_edges, "eye position", time_window,
                    blocks="Learning", trial_sets="instruction", rotate=True,
                    bin_by_instructed=True, return_t_inds=True)
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
                                    rotate=True, bin_by_instructed=True,
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
                x, y = ab.get_mean_xy_traces(ldp_sess, "eye position", time_window,
                            blocks=block, trial_sets=curr_set, rotate=True)
                x, y = ab.subtract_baseline_tuning(ldp_sess, base_block, curr_set,
                                                   "eye position", x, y)
                # Append the data
                all_post_tuning_x_pos[block][curr_set].append(x)
                all_post_tuning_y_pos[block][curr_set].append(y)

        if verbose: print("Getting bin learning velocity data")
        bin_x_data, bin_y_data, bin_t_inds = ab.get_binned_mean_xy_traces(ldp_sess,
                                    learn_bin_edges,
                                    "eye velocity", time_window, blocks="Learning",
                                    trial_sets="instruction", rotate=True,
                                    bin_by_instructed=True, return_t_inds=True)
        bin_x_data, bin_y_data = ab.subtract_baseline_tuning_binned(ldp_sess,
                                    base_block, "instruction", "eye velocity",
                                    bin_x_data, bin_y_data)
        # Append all the bin data
        for inds_ind, inds in enumerate(bin_t_inds):
            if len(inds) < n_bin_min_trials:
                # Not enough trials
                continue
            # Otherwise add it
            all_learning_bins_x_vel[inds_ind].append(bin_x_data[inds_ind])
            all_learning_bins_y_vel[inds_ind].append(bin_y_data[inds_ind])


        if n_files > 1:
            break

    output_dict = {}
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
            all_post_tuning_x_pos[block][curr_set] = np.vstack(all_post_tuning_x_pos[block][curr_set])
            all_post_tuning_y_pos[block][curr_set] = np.vstack(all_post_tuning_y_pos[block][curr_set])
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


    return output_dict
