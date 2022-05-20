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

    pattern = re.compile(f_regex)
    filenames = [f for f in os.listdir(directory) if pattern.search(f) is not None]

    data_x_all_learning_bins = [[] for x in range(0, len(learn_bin_edges)-1)]
    data_y_all_learning_bins = [[] for x in range(0, len(learn_bin_edges)-1)]
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
            print("SKIPPIN file: ", f)
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
            try:
                if len(inds) < n_bin_min_trials:
                    # Not enough trials
                    continue
            except:
                print(inds)
                raise
            # Otherwise add it
            data_x_all_learning_bins[inds_ind].append(bin_x_data[inds_ind])
            data_y_all_learning_bins[inds_ind].append(bin_y_data[inds_ind])

        if verbose: print("Getting bin learning probe position data")
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

        if n_files > 1:
            return data_x_all_learning_bins, data_y_all_learning_bins

    # Concatenate these arrays
    for bin in range(0, len(data_x_all_learning_bins)):
        if len(data_x_all_learning_bins[bin]) == 0:
            continue
        data_x_all_learning_bins[bin] = np.vstack(data_x_all_learning_bins[bin])
        data_y_all_learning_bins[bin] = np.vstack(data_y_all_learning_bins[bin])

    return data_x_all_learning_bins, data_y_all_learning_bins
