import numpy as np



ax_inds_to_names = {0: "Learn_ax",
                    1: "Linear_model",
                    2: "Fixation",
                    3: "Washout",
                    }


def setup_axes():
    plot_handles = {}
    plot_handles['fig'], ax_handles = plt.subplots(2, 2, figsize=(8, 11))
    ax_handles = ax_handles.ravel()
    for ax_ind in range(0, ax_handles.size):
        plot_handles[ax_inds_to_names[ax_ind]] = ax_handles[ax_ind]

    
    return plot_handles


def select_trials_by_win(trial_inds, trial_win):
    """ Will return the indices in "trial_inds" that falls within the two element window "trial win". Input
    trial_inds MUST BE SORTED! or this won't work (output of 'session._parse_blocks_trial_sets' is sorted)
    """
    selected_trials = []
    for ti in trial_inds:
        if ti >= trial_win[1]:
            break
        if ti >= trial_win[0]:
            selected_trials.append(ti)
    return np.array(selected_trials, dtype=np.int64)


def scatterplot_neuron_learning(neuron, fix_win, learn_win, sigma=12.5, 
                                cutoff_sigma=4, show_fig=False):
    """
    """
    # In case we want to switch this later
    fr_learn_win = learn_win

    # Some currently hard coded variables
    use_smooth_fix = True
    use_baseline_block = "StabTunePre"
    n_t_instructed_win = [80, 100] # Need to hit at least this many learning trials to get counted
    n_t_baseline_min = 15 # Need at least 15 baseline trials to get counted
    # Append valid neuron trials to input trial_sets
    trial_sets = neuron.append_valid_trial_set(trial_sets)

    # Setup figure layout
    plot_handles = setup_axes()
    plot_handles['fig'].suptitle(f"Firing rate changes as a function of tuning", fontsize=12, y=.95)
    
    # Get learning window tuning block rates
    fr_baseline_means = {}
    fr_baseline_fix_means = {}
    for tune_trial in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
        fr = neuron.get_firing_traces_fix_adj(fr_learn_win, use_baseline_block, tune_trial, 
                                              fix_time_window=fix_win, sigma=sigma, 
                                              cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                              rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                              return_inds=False)
        fr_fix = neuron.get_firing_traces(fix_win, use_baseline_block, tune_trial)
        if len(fr) == 0:
            print(f"No tuning trials found for {tune_trial} in blocks {tune_adjust_block}", flush=True)
            return plot_handles
        elif fr.shape[0] < n_t_baseline_min:
            print(f"Not enough tuning trials found for {tune_trial} in blocks {tune_adjust_block}", flush=True)
            return plot_handles
        else:
            fr_baseline_means[tune_trial] = np.nanmean(np.nanmean(fr, axis=1), axis=0)
            fr_baseline_fix_means[tune_trial] = np.nanmean(np.nanmean(fr_fix, axis=1), axis=0)
    
    # Now get the "learning" responses
    fr_learn_window = {}
    for block in ["Learning", "Washout"]:
        for trial_types in [["instruction", "pursuit"]]:
            valid_b_inds = neuron.session._parse_blocks_trial_sets([block], trial_types)
            select_inds = select_trials_by_win(valid_b_inds, n_t_instructed_win)
            fr = neuron.get_firing_traces_fix_adj(fr_learn_win, block, select_inds, 
                                                fix_time_window=fix_win, sigma=sigma, 
                                                cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                                return_inds=False)
            if len(fr) == 0:
                print(f"No trials found for {trial_types} in blocks {block}", flush=True)
            else:
                fr = np.nanmean(fr, axis=1) - fr_baseline_means['pursuit']
                fr_learn_window[block] = np.nanmean(fr)

