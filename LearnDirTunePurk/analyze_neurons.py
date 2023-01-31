import numpy as np
import warnings
import LearnDirTunePurk.analyze_behavior as ab



def subtract_baseline_firing(ldp_sess, base_block, base_set, base_data, x, y,
                             alpha_scale_factors=None):
    """ Subtracts the baseline data on the orthogonal axis relative to the
    4 axes defined by ldp_sess.trial_set_base_axis.
    Relatively simple function but cleans up other code since this is clumsy.
    Data are subtracted from x and y IN PLACE!"""
    if (len(x) == 0) or (len(y) == 0):
        return x, y
    # alpha_scale_factors = None
    # print("NO SCALING IN BASELINE OFF!!!!!!")
    if ldp_sess.trial_set_base_axis[base_set] == 0:
        if alpha_scale_factors is None:
            x = x - ldp_sess.baseline_tuning[base_block][base_data][base_set][0, :]
        else:
            x = x - (alpha_scale_factors * ldp_sess.baseline_tuning[base_block][base_data][base_set][0, :])
    elif ldp_sess.trial_set_base_axis[base_set] == 1:
        if alpha_scale_factors is None:
            y = y - ldp_sess.baseline_tuning[base_block][base_data][base_set][1, :]
        else:
            y = y - (alpha_scale_factors * ldp_sess.baseline_tuning[base_block][base_data][base_set][1, :])
    else:
        raise ValueError("Could not match baseline for subtraction for block '{0}', set '{1}', and data '{2}'.".format(base_block, base_set, base_data))

    return x, y


def subtract_baseline_firing_binned(ldp_sess, base_block, base_set, base_data,
                                    x, y, alpha_scale_factors=None):
    """ Subtracts the baseline data on the orthogonal axis relative to the
    4 axes defined by ldp_sess.trial_set_base_axis.
    Relatively simple function but cleans up other code since this is clumsy.
    Data are subtracted from x and y IN PLACE!"""
    if len(x) != len(y):
        raise ValueError("x and y data must have the same number of bins (length).")

    if alpha_scale_factors is None:
        for bin_ind in range(0, len(x)):
            x[bin_ind], y[bin_ind] = subtract_baseline_tuning(ldp_sess, base_block,
                                        base_set, base_data, x[bin_ind], y[bin_ind],
                                        alpha_scale_factors=None)
    else:
        for bin_ind in range(0, len(x)):
            x[bin_ind], y[bin_ind] = subtract_baseline_tuning(ldp_sess, base_block,
                                        base_set, base_data, x[bin_ind], y[bin_ind],
                                        alpha_scale_factors=alpha_scale_factors[bin_ind])

    return x, y


def get_mean_firing_trace(ldp_sess, series_name, time_window, blocks=None,
                        trial_sets=None, return_inds=False):
    """ Calls ldp_sess.get_data_array and takes the mean over rows of the output. """
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    fr, t = ldp_sess.get_data_array(series_name, time_window, blocks=blocks,
                        trial_sets=trial_sets, return_inds=True)
    if len(fr) == 0:
        # Found no matching data
        if return_inds:
            return fr, t
        else:
            return fr
    with warnings.catch_warnings():
        fr = np.nanmean(fr, axis=0)

    if return_inds:
        return fr, t
    else:
        return fr


def get_binned_mean_firing_trace(ldp_sess, edges, series_name, time_window,
                              blocks=None, trial_sets=None,
                              bin_basis="raw", return_t_inds=False):
    """ Calls get_data_array and then bin_by_trial. Returns the mean of each bin
    corresponding to 'edges'. """
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
    fr, t = ldp_sess.get_data_array(series_name, time_window, blocks=blocks,
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
    else:
        raise ValueError("Unrecognized bin_basis {0}.".format(bin_basis))
    bin_inds = ab.bin_by_trial(t, edges, inc_last_edge=True)
    fr_binned_traces = []
    t_binned_inds = []
    for inds in bin_inds:
        if len(inds) == 0:
            fr_binned_traces.append([])
            t_binned_inds.append([])
        elif len(inds) == 1:
            fr_binned_traces.append(fr[inds[0], :])
            t_binned_inds.append(np.array([t[inds[0]]]))
        else:
            numpy_inds = np.array(inds)
            with warnings.catch_warnings():
                fr_binned_traces.append(np.nanmean(fr[numpy_inds, :], axis=0))
            t_binned_inds.append(t[numpy_inds])

    if return_t_inds:
        return fr_binned_traces, t_binned_inds
    else:
        return fr_binned_traces
