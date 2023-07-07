import numpy as np



def get_vel_pos_data(neuron, pos_win, vel_win):
    """ Gets the xy position in pos win and velocity in vel_win
    """
    data_block = "RandVPTunePre"
    norm_block = "StandTunePre"

    full_win = [pos_win[0], vel_win[1]]
    full_win_t_inds = np.arange(full_win[0], full_win[1])
    pos_win_t_inds = (full_win_t_inds >= pos_win[0]) & (full_win_t_inds < pos_win[1])
    vel_win_t_inds = (full_win_t_inds >= vel_win[0]) & (full_win_t_inds < vel_win[1])
    # Get position and velocity data for each trial type
    response_vs_pos = {}
    for trial_type in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
        fr, fr_t_inds = neuron.get_firing_traces(full_win, data_block, 
                                                 trial_type, return_inds=True)
        eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", full_win, blocks=data_block,
                                                        trial_sets=fr_t_inds, return_inds=False)
        # Select appropriate velocity dimension and get velocity response
        use_eye = eyev_p if "pursuit" in trial_type else eyev_l
        fr_mod = np.nanmean(fr[:, vel_win_t_inds], axis=1) - np.nanmean(fr[:, pos_win_t_inds], axis=1)
        # print(f"Random fixation mean mod {np.nanmean(fr_mod)} for {trial_type}")
        eyev_mod = np.nanmean(use_eye[:, vel_win_t_inds], axis=1) - np.nanmean(use_eye[:, pos_win_t_inds], axis=1)
        # Velocity sign is arbitrary here so remove
        eyev_mod = np.abs(eyev_mod)
        eyev_mod[eyev_mod < 2.] = np.nan
        spk_per_deg = fr_mod / eyev_mod
        print(f"Random fixation mean {trial_type} spikes per deg mean {np.nanmean(spk_per_deg)}")
        # Now we need to get the initial fixation xy position
        eyep_p, eyep_l = neuron.session.get_xy_traces("eye position", full_win, blocks=data_block,
                                                        trial_sets=fr_t_inds, return_inds=False)
        eyep_p = np.nanmean(eyep_p[:, pos_win_t_inds], axis=1)
        eyep_l = np.nanmean(eyep_l[:, pos_win_t_inds], axis=1)

        # Now repeat for the mean of the norm_block for normalization
        fr, fr_t_inds = neuron.get_firing_traces(full_win, norm_block, 
                                                 trial_type, return_inds=True)
        eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", full_win, blocks=norm_block,
                                                        trial_sets=fr_t_inds, return_inds=False)
        # Select appropriate velocity dimension and get velocity response
        fr_mod = np.nanmean(fr[:, vel_win_t_inds], axis=1) - np.nanmean(fr[:, pos_win_t_inds], axis=1)
        eyev_mod = np.nanmean(use_eye[:, vel_win_t_inds], axis=1) - np.nanmean(use_eye[:, pos_win_t_inds], axis=1)
        # Velocity sign is arbitrary here so remove
        eyev_mod = np.abs(eyev_mod)
        eyev_mod[eyev_mod < 2.] = np.nan
        eyev_mod = np.nanmean(eyev_mod)
        spk_per_deg_norm = fr_mod / eyev_mod
        print(f"Standard TUNING mean {trial_type} spikes per deg mean {np.nanmean(spk_per_deg_norm)}")
        spk_per_deg_norm = np.nanmean(spk_per_deg_norm)
        spk_per_deg_norm = 1.

        # Then normalize the spikes per degree and save numpy array output
        normed_spk_per_deg = spk_per_deg / spk_per_deg_norm
        response_vs_pos[trial_type] = np.hstack((eyep_p[:, None], eyep_l[:, None], normed_spk_per_deg[:, None]))

    return response_vs_pos