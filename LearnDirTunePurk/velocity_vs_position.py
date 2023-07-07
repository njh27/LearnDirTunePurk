import numpy as np



def get_vel_pos_data(neuron, pos_win, vel_win):
    """ Gets the xy position in pos win and velocity in vel_win
    """
    data_block = "RandVPTunePre"
    norm_block = "StandTunePre"

    full_win = [pos_win[0], vel_win[1]]
    full_win_t_inds = np.arange(full_win[0], full_win[1])
    pos_win_t_inds = pos_win[0] <= full_win_t_inds < pos_win[1]
    vel_win_t_inds = vel_win[0] <= full_win_t_inds < vel_win[1]
    # Get position and velocity data for each trial type
    for trial_type in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
        fr, fr_t_inds = neuron.get_firing_traces(full_win, data_block, 
                                                 trial_type, return_inds=True)
        eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", full_win, blocks=data_block,
                                                        trial_sets=fr_t_inds, return_inds=False)
        # Select appropriate velocity dimension
        use_eye = eyev_p if "pursuit" in trial_type else eyev_l
        fr_mod = fr[:, vel_win_t_inds] - fr[:, pos_win_t_inds]
        eyev_mod = use_eye[:, vel_win_t_inds] - use_eye[:, pos_win_t_inds]
        spk_per_deg = fr_mod / eyev_mod

    return 