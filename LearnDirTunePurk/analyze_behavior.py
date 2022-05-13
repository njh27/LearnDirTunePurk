import numpy as np


def get_position(ldp_sess):
    pass


def bin_by_trial(t_inds, edges, inc_last_edge=True):
    """ Bins the trial indices (or any numbers) in t_inds within the edges
    specified by the intervals in 'edges'. If edges is a scalor, equally
    spaced 'edges' number of bins are created. Output is a list for each
    bin specified by edges, containing a list of the indices of t_inds that
    fall within the bin. """
    t_args = np.argsort(t_inds)
    if len(edges) == 1:
        edges = np.linspace(t_inds[0], t_inds[-1]+.001, edges)
    # Initialize each output bin
    bin_out = [[] for x in range(0, len(edges)-1)]
    curr_bin_out = []
    curr_bin = 0
    curr_t_arg = 0
    while (curr_bin < (len(bin_out))):
        if t_inds[t_args[curr_t_arg]] < edges[curr_bin]:
            # This t_ind is below specified bin range so skip
            curr_t_arg += 1
        elif ( (t_inds[t_args[curr_t_arg]] >= edges[curr_bin]) and
               (t_inds[t_args[curr_t_arg]] < edges[curr_bin+1]) ):
            # This t_ind falls within the current bin
            curr_bin_out.append(t_args[curr_t_arg])
            curr_t_arg += 1
        elif t_inds[t_args[curr_t_arg]] >= edges[curr_bin+1]:
            if ( ((curr_bin+1) == len(edges)) and inc_last_edge):
                # This t_ind is on the final bin edge and include last edge is on
                curr_bin_out.append(t_args[curr_t_arg])
                curr_t_arg += 1
            else:
                # This t_ind is beyond current bin edge so save and increment bin
                bin_out[curr_bin] = curr_bin_out
                curr_bin_out = []
                curr_bin += 1
        else:
            # Shouldn't be possible, but just in case
            raise RuntimeError("Can't figure out how to handle t_ind {0} in edges {1}.".format(t_inds[t_args[curr_t_arg]], edges[curr_bin:curr_bin+1]))
        if curr_t_arg >= len(t_args):
            # We are done with all t_inds so save and exit
            bin_out[curr_bin] = curr_bin_out
            break
            
    return bin_out


def get_xy_traces(ldp_sess, series_name, time_window, blocks=None,
                 trial_sets=None, return_inds=False, rotate=False):
    """ If rotated, the "x/horizontal" axis of output will be the "pursuit"
    axis and the "y/vertical" axis will be the learning axis.
    """
    # Parse data name to type and series xy
    if "eye" in series_name:
        if "position" in series_name:
            x_name = "horizontal_eye_position"
            y_name = "vertical_eye_position"
        elif "velocity" in series_name:
            x_name = "horizontal_eye_velocity"
            y_name = "vertical_eye_velocity"
        else:
            raise InputError("Data name for 'eye' must also include either 'position' or 'velocity' to specify data type.")
        data_name = "eye"
    elif "target" in series_name:
        if "position" in series_name:
            x_name = "horizontal_target_position"
            y_name = "vertical_target_position"
        elif "velocity" in series_name:
            x_name = "horizontal_target_velocity"
            y_name = "vertical_target_velocity"
        else:
            raise InputError("Data name for 'eye' must also include either 'position' or 'velocity' to specify data type.")
        data_name = "target0"
    else:
        raise InputError("Data name must include either 'eye' or 'target' to specify data type.")

    data_out_x = []
    data_out_y = []
    t_inds_out = []
    t_inds = ldp_sess._parse_blocks_trial_sets(blocks, trial_sets)
    for t in t_inds:
        if not ldp_sess._session_trial_data[t]['incl_align']:
            # Trial is not aligned with others due to missing event
            continue
        trial_obj = ldp_sess._trial_lists[data_name][t]
        ldp_sess._set_t_win(t, time_window)
        valid_tinds = ldp_sess._session_trial_data[t]['curr_t_win']['valid_tinds']
        out_inds = ldp_sess._session_trial_data[t]['curr_t_win']['out_inds']

        t_data_x = np.full(out_inds.shape[0], np.nan)
        t_data_y = np.full(out_inds.shape[0], np.nan)
        t_data_x[out_inds] = trial_obj['data'][x_name][valid_tinds]
        t_data_y[out_inds] = trial_obj['data'][y_name][valid_tinds]
        if rotate:
            rot_data = ldp_sess.rotation_matrix @ np.vstack((t_data_x, t_data_y))
            t_data_x = rot_data[0, :]
            t_data_y = rot_data[1, :]

        data_out_x.append(t_data_x)
        data_out_y.append(t_data_y)
        t_inds_out.append(t)

    if return_inds:
        return np.vstack(data_out_x), np.vstack(data_out_y), np.hstack(t_inds_out)
    else:
        return np.vstack(data_out_x), np.vstack(data_out_y)
