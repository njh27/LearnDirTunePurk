import numpy as np


def get_position(ldp_sess):
    pass


def get_data_array(ldp_sess, series_name, time_window, blocks=None,
                   trial_sets=None, return_inds=False):
    """ Returns a n trials by m time points numpy array of the requested
    timeseries data. Missing data points are filled in with np.nan.
    Call "data_names()" to get a list of available data names. """
    data_out = []
    t_inds_out = []
    data_name = self.__series_names[series_name]
    t_inds = self._parse_blocks_trial_sets(blocks, trial_sets)
    for t in t_inds:
        if not self._session_trial_data[t]['incl_align']:
            # Trial is not aligned with others due to missing event
            continue
        trial_obj = self._trial_lists[data_name][t]
        self._set_t_win(t, time_window)
        valid_tinds = self._session_trial_data[t]['curr_t_win']['valid_tinds']
        out_inds = self._session_trial_data[t]['curr_t_win']['out_inds']
        t_data = np.full(out_inds.shape[0], np.nan)
        t_data[out_inds] = trial_obj['data'][series_name][valid_tinds]
        data_out.append(t_data)
        t_inds_out.append(t)
    if return_inds:
        return np.vstack(data_out), np.hstack(t_inds_out)
    else:
        return np.vstack(data_out)

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
