import numpy as np



def get_neuron_scatter_data(neuron, fix_win, learn_win, sigma=12.5, 
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

    # adjusted the n_instructed win to absolute trial number
    raw_instructed_inds = []
    for t_ind in range(neuron.session.blocks['Learning'][0], neuron.session.blocks['Learning'][1]):
        if n_t_instructed_win[0] <= neuron.session.n_instructed[t_ind] < n_t_instructed_win[1]:
            raw_instructed_inds.append(t_ind)
    for rel_ind, t_ind in enumerate(range(neuron.session.blocks['Washout'][0], neuron.session.blocks['Washout'][1])):
        if n_t_instructed_win[0] <= rel_ind < n_t_instructed_win[1]:
            raw_instructed_inds.append(t_ind)
    raw_instructed_inds = np.array(raw_instructed_inds)
    
    # Get learning window tuning block rates
    fr_win_means = {}
    for tune_trial in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
        fr = neuron.get_firing_traces_fix_adj(fr_learn_win, use_baseline_block, tune_trial, 
                                              fix_time_window=fix_win, sigma=sigma, 
                                              cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                              rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                              return_inds=False)
        if len(fr) == 0:
            print(f"No tuning trials found for {tune_trial} in blocks {tune_adjust_block}", flush=True)
            return fr_win_means
        elif fr.shape[0] < n_t_baseline_min:
            print(f"Not enough tuning trials found for {tune_trial} in blocks {tune_adjust_block}", flush=True)
            return fr_win_means
        else:
            fr_win_means[tune_trial + "_tune"] = np.nanmean(np.nanmean(fr, axis=1), axis=0)
    fr_fix = neuron.get_firing_traces(fix_win, use_baseline_block, None)
    fr_win_means["base_fix"] = np.nanmean(np.nanmean(fr_fix, axis=1), axis=0)
    
    # Now get the "learning" responses
    for block in ["Learning", "Washout"]:
        for trial_types in [["instruction", "pursuit"]]:
            t_set_inds = neuron.session.union_trial_sets_to_indices(trial_types)
            select_inds = neuron.session._parse_blocks_trial_sets([block], [t_set_inds, raw_instructed_inds])
            fr = neuron.get_firing_traces_fix_adj(fr_learn_win, block, select_inds, 
                                                fix_time_window=fix_win, sigma=sigma, 
                                                cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                                return_inds=False)
            if len(fr) == 0:
                print(f"No trials found for {trial_types} in blocks {block}", flush=True)
            else:
                fr = np.nanmean(fr, axis=1) - fr_win_means['pursuit_tune']
                fr_win_means[block + "_learn"] = np.nanmean(fr)

    return fr_win_means

