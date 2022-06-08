import numpy as np
import warnings



def subtract_baseline_tuning(ldp_sess, base_block, base_set, base_data, x, y):
    """ Subtracts the baseline data on the orthogonal axis relative to the
    4 axes defined by ldp_sess.trial_set_base_axis.
    Relatively simple function but cleans up other code since this is clumsy.
    Data are subtracted from x and y IN PLACE!"""
    if (len(x) == 0) or (len(y) == 0):
        return x, y
    if ldp_sess.trial_set_base_axis[base_set] == 0:
        x = x - ldp_sess.baseline_tuning[base_block][base_data][base_set][0, :]
    elif ldp_sess.trial_set_base_axis[base_set] == 1:
        y = y - ldp_sess.baseline_tuning[base_block][base_data][base_set][1, :]
    else:
        raise ValueError("Could not match baseline for subtraction for block '{0}', set '{1}', and data '{2}'.".format(base_block, base_set, base_data))

    return x, y

def subtract_baseline_tuning_binned(ldp_sess, base_block, base_set, base_data, x, y):
    """ Subtracts the baseline data on the orthogonal axis relative to the
    4 axes defined by ldp_sess.trial_set_base_axis.
    Relatively simple function but cleans up other code since this is clumsy.
    Data are subtracted from x and y IN PLACE!"""
    if len(x) != len(y):
        raise ValueError("x and y data must have the same number of bins (length).")
    for bin_ind in range(0, len(x)):
        x[bin_ind], y[bin_ind] = subtract_baseline_tuning(ldp_sess, base_block,
                                    base_set, base_data, x[bin_ind], y[bin_ind])

    return x, y


def get_mean_xy_traces(ldp_sess, series_name, time_window, blocks=None,
                        trial_sets=None):
    """ Calls get_xy_traces below and takes the mean over rows of the output. """

    x, y = get_xy_traces(ldp_sess, series_name, time_window, blocks=blocks,
                     trial_sets=trial_sets, return_inds=False)
    if x.shape[0] == 0:
        # Found no matching data
        return x, y
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        x = np.nanmean(x, axis=0)
        y = np.nanmean(y, axis=0)

    return x, y


def get_binned_mean_xy_traces(ldp_sess, edges, series_name, time_window,
                              blocks=None, trial_sets=None,
                              bin_basis="raw",
                              return_t_inds=False):
    """ Calls get_xy_traces and then bin_by_trial. Returns the mean of each bin
    corresponding to 'edges'. """
    x, y, t = get_xy_traces(ldp_sess, series_name, time_window, blocks=blocks,
                     trial_sets=trial_sets, return_inds=True)
    if bin_basis.lower() == "raw":
        # Do nothing
        pass
    elif bin_basis.lower() == "order":
        t_order = np.argsort(t)
        t[t_order] = np.arange(0, len(t))
    elif bin_basis.lower() == "instructed":
        t = ldp_sess.n_instructed[t]
    elif bin_basis.lower() == "block":
        # If blocks is None, then we do nothing it's same as raw
        if blocks is not None:
            if isinstance(blocks, list):
                if len(blocks) > 1:
                    raise ValueError("Block bin basis is not defined over multiple blocks because it is ambiguous.")
                else:
                    blocks = blocks[0]
            t = t - ldp_sess[blocks][0]
    bin_inds = bin_by_trial(t, edges, inc_last_edge=True)
    x_binned_traces = []
    y_binned_traces = []
    t_binned_inds = []
    for inds in bin_inds:
        if len(inds) == 0:
            x_binned_traces.append([])
            y_binned_traces.append([])
            t_binned_inds.append([])
        elif len(inds) == 1:
            x_binned_traces.append(x[inds[0], :])
            y_binned_traces.append(y[inds[0], :])
            t_binned_inds.append(np.array([t[inds[0]]]))
        else:
            numpy_inds = np.array(inds)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
                x_binned_traces.append(np.nanmean(x[numpy_inds, :], axis=0))
                y_binned_traces.append(np.nanmean(y[numpy_inds, :], axis=0))
            t_binned_inds.append(t[numpy_inds])

    if return_t_inds:
        return x_binned_traces, y_binned_traces, t_binned_inds
    else:
        return x_binned_traces, y_binned_traces


def get_binned_xy_traces(ldp_sess, edges, series_name, time_window,
                         blocks=None, trial_sets=None,
                         bin_basis=False):
    """ Calls get_xy_traces and then bin_by_trial. Returns the mean of each bin
    corresponding to 'edges'. """
    x, y, t = get_xy_traces(ldp_sess, series_name, time_window, blocks=blocks,
                     trial_sets=trial_sets, return_inds=True)
    if bin_basis:
        t = ldp_sess.n_instructed[t]
    bin_inds = bin_by_trial(t, edges, inc_last_edge=True)
    x_binned_traces = []
    y_binned_traces = []
    for inds in bin_inds:
        if len(inds) == 0:
            x_binned_traces.append([])
            y_binned_traces.append([])
        else:
            numpy_inds = np.array(inds)
            x_binned_traces.append(x[numpy_inds, :])
            y_binned_traces.append(y[numpy_inds, :])

    return x_binned_traces, y_binned_traces


def bin_by_trial(t_inds, edges, inc_last_edge=True):
    """ Bins the trial indices (or any numbers) in t_inds within the edges
    specified by the intervals in 'edges'. If edges is a scalor, equally
    spaced 'edges' number of bins are created. Output is a list for each
    bin specified by edges, containing a list of the indices of t_inds that
    fall within the bin. """
    t_args = np.argsort(t_inds)
    if len(edges) == 1:
        edges = np.linspace(np.amin(t_inds), np.amax(t_inds)+.001, edges)
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


def rescale_velocity(ldp_sess, x, y, t):
    pass


def get_xy_traces(ldp_sess, series_name, time_window, blocks=None,
                 trial_sets=None, return_inds=False):
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
        if ldp_sess.rotate:
            rot_data = ldp_sess.rotation_matrix @ np.vstack((t_data_x, t_data_y))
            t_data_x = rot_data[0, :]
            t_data_y = rot_data[1, :]

        data_out_x.append(t_data_x)
        data_out_y.append(t_data_y)
        t_inds_out.append(t)

    if return_inds:
        if len(data_out_x) > 0:
            # We found data to concatenate
            return np.vstack(data_out_x), np.vstack(data_out_y), np.hstack(t_inds_out)
        else:
            return np.zeros((0, time_window[1]-time_window[0])), np.zeros((0, time_window[1]-time_window[0])), np.array([], dtype=np.int32)
    else:
        if len(data_out_x) > 0:
            # We found data to concatenate
            return np.vstack(data_out_x), np.vstack(data_out_y)
        else:
            return np.zeros((0, time_window[1]-time_window[0])), np.zeros((0, time_window[1]-time_window[0]))
