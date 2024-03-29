import numpy as np
import scipy.stats as stats
import pickle
import matplotlib.pyplot as plt
from NeuronAnalysis.fit_neuron_to_eye import FitNeuronToEye, piece_wise_eye_data
from SessionAnalysis.utils import eye_data_series



ax_inds_to_names = {0: "scatter_raw",
                    1: "scatter_learned",
                    2: "learn_80_fr",
                    3: "learn_480_fr",
                    4: "pursuit_base_fr",
                    5: "learn_base_fr",
                    }
# Hard coded used globally rightnow
t_title_pad = 0
tuning_block = "StandTunePre"

def round_to_nearest_five_greatest(n, round_n=5):
    if n > 0:
        return n + (round_n - n % round_n) if n % round_n != 0 else n
    else:
        return n - (n % round_n) if n % round_n != 0 else n

def setup_axes():
    plot_handles = {}
    plot_handles['fig'], ax_handles = plt.subplots(3, 2, figsize=(8, 11))
    ax_handles = ax_handles.ravel()
    for ax_ind in range(0, ax_handles.size):
        plot_handles[ax_inds_to_names[ax_ind]] = ax_handles[ax_ind]

    return plot_handles

def setup_figcomp_figs():
    fig_handles = {}
    plot_handles = {}
    for ax_ind in range(0, len(ax_inds_to_names)):
        fig_handles[ax_inds_to_names[ax_ind]], plot_handles[ax_inds_to_names[ax_ind]] = plt.subplots()

    return fig_handles, plot_handles


def set_all_axis_same(ax_list, round_n=5):
    lims_x = [np.inf, -np.inf]
    lims_y = [np.inf, -np.inf]
    for ax in ax_list:
        xlims = ax.get_xlim()
        if xlims[0] < lims_x[0]:
            lims_x[0] = xlims[0]
        if xlims[1] > lims_x[1]:
            lims_x[1] = xlims[1]
        ylims = ax.get_ylim()
        if ylims[0] < lims_y[0]:
            lims_y[0] = ylims[0]
        if ylims[1] > lims_y[1]:
            lims_y[1] = ylims[1]
    lims_x = [round_to_nearest_five_greatest(x, round_n) for x in lims_x]
    lims_y = [round_to_nearest_five_greatest(x, round_n) for x in lims_y]
    for ax in ax_list:
        ax.set_xlim(lims_x)
        ax.set_ylim(lims_y)

def nansem(data, axis=None):
    """ Compute SEM while ignoring nan values
    """
    std_dev = np.nanstd(data, axis=axis)
    count = np.sum(~np.isnan(data), axis=axis)  # Counts non-nan values along the axis
    return std_dev / np.sqrt(count)

def plot_95_CI(ax, xvals, mean_trace, sem_trace, mean_color="black", ci_color=[.75, .75, .75], **kwargs):
    """ Plots the mean trace with shaded 95% confidence bounds computed from the SEM of the 
    input trace from sem_trace on the axes handle suppled by ax. """
    plot_params = {'mean_color': mean_color,
                   'mean_label': None,
                   'ci_color': ci_color,
                   'ci_linecolor': mean_color,
                   'ci_label': None,
                   'linewidth': 2.,
                   'linestyle': "-",
                   'alpha': 0.6,
                   'ci_linewidth': None,
                   }
    for key, value in kwargs.items():
        if key in plot_params:
            plot_params[key] = value
    # calculate confidence intervals (95%)
    ci = sem_trace * stats.t.ppf((1 + 0.95) / 2., mean_trace.shape[0]-1)
    # plot mean line
    ax.plot(xvals, mean_trace, color=mean_color, linewidth=plot_params['linewidth'], linestyle=plot_params['linestyle'], 
            label=plot_params['mean_label'])

    # # plot confidence interval
    # if plot_params['ci_linewidth'] is not None:
    #     # Plot the bounds of the CI
    #     ax.plot(xvals, mean_trace + ci, color=plot_params['ci_linecolor'], linewidth=plot_params['ci_linewidth'], linestyle=plot_params['linestyle'], 
    #             label=plot_params['ci_label'], alpha=plot_params['alpha'])
    #     ax.plot(xvals, mean_trace - ci, color=plot_params['ci_linecolor'], linewidth=plot_params['ci_linewidth'], linestyle=plot_params['linestyle'], 
    #             alpha=plot_params['alpha'])
    # ax.fill_between(xvals, mean_trace-ci, mean_trace+ci, color=plot_params['ci_color'], alpha=plot_params['alpha'])

def find_block_start_t_ind(neuron_dict, blockname):
    """ Iterates through the data in blockname to find the earliest t_ind to indicate block start
    """
    b_start_ind = np.inf
    for t_set in neuron_dict[blockname].keys():
        if neuron_dict[blockname][t_set]['t_inds'].size == 0:
            continue
        if neuron_dict[blockname][t_set]['t_inds'][0] < b_start_ind:
            b_start_ind = neuron_dict[blockname][t_set]['t_inds'][0]
    if not np.isfinite(b_start_ind):
        b_start_ind = 0
    return b_start_ind

def get_traces_win(neuron_dict, blockname, trial_type, data_type, win, trial_win=None, nansac=False):
    """
    """
    if trial_win is None:
        trial_inds = np.ones(neuron_dict[blockname][trial_type][data_type].shape[0], dtype='bool')
    else:
        if blockname == "Learning":
            trial_inds = np.copy(neuron_dict[blockname][trial_type]['n_inst'])
        else:
            trial_inds = np.copy(neuron_dict[blockname][trial_type]['t_inds'])
            b_start_ind = find_block_start_t_ind(neuron_dict, blockname)
            trial_inds -= b_start_ind
        trial_inds = np.logical_and(trial_inds >= trial_win[0], trial_inds < trial_win[1])
    t_inds = np.arange(neuron_dict['trace_win'][0], neuron_dict['trace_win'][1])
    time_inds = np.logical_and(t_inds >= win[0], t_inds < win[1])
    if neuron_dict[blockname][trial_type][data_type].shape[0] == 0:
        return np.full((1, np.count_nonzero(time_inds)), np.nan)
    traces_win = neuron_dict[blockname][trial_type][data_type][trial_inds, :][:, time_inds]
    if traces_win.shape[0] == 0:
        return np.full((1, np.count_nonzero(time_inds)), np.nan)
    if nansac:
        # Get saccade nans from eye velocity data and apply to current
        sac_nans = get_traces_win(neuron_dict, blockname, trial_type, "eyev_l", 
                                  win, trial_win=trial_win, nansac=False)
        sac_nans = np.isnan(sac_nans)
        traces_win[sac_nans] = np.nan

    return traces_win

def get_mean_win(neuron_dict, blockname, trial_type, data_type, win, trial_win=None, min_trials=10, nansac=False):
    traces_win = get_traces_win(neuron_dict, blockname, trial_type, data_type, win, trial_win=trial_win, nansac=nansac)
    if traces_win.shape[0] < min_trials:
        return np.nan
    mean_trace = np.nanmean(traces_win, axis=0)
    mean_win = np.nanmean(mean_trace)

    return mean_win

def get_learn_response_inst_probe_traces_win(neuron_dict, data_type, win=None, trial_win=None, nansac=False):
    if win is None:
        win = neuron_dict['trace_win']
    learn_80_trace = get_traces_win(neuron_dict, "Learning", "pursuit", data_type, win, trial_win=trial_win, nansac=nansac)
    learn_80_inst_trace = get_traces_win(neuron_dict, "Learning", "instruction", data_type, win, trial_win=trial_win, nansac=nansac)
    all_learn = np.vstack((learn_80_trace, learn_80_inst_trace))

    return all_learn

def get_learn_response_inst_probe_mean_win(neuron_dict, data_type, win, trial_win=None, min_trials=10, nansac=False):
    learn_probe = get_traces_win(neuron_dict, "Learning", "pursuit", data_type, win, trial_win=trial_win, nansac=nansac)
    learn_inst = get_traces_win(neuron_dict, "Learning", "instruction", data_type, win, trial_win=trial_win, nansac=nansac)
    all_learn = np.vstack((learn_probe, learn_inst))
    if all_learn.shape[0] < min_trials:
        return np.nan
    learn_resp = np.nanmean(np.nanmean(all_learn, axis=0, keepdims=True))

    return learn_resp

def get_y_hat_from_avg(neuron_dict, blockname, trial_type, trial_win=None, filter_win=1):
    """ Gets behavioral data from blocks and trial sets and formats in a
    way that it can be used to predict firing rate according to the linear
    eye kinematic model using predict_lin_eye_kinematics. """
    lagged_trace_win = [neuron_dict['trace_win'][0] + neuron_dict['lin_model']['eye_lag'],
                        neuron_dict['trace_win'][1] + neuron_dict['lin_model']['eye_lag']
                        ]
    t_inds = np.arange(neuron_dict['trace_win'][0], neuron_dict['trace_win'][1])
    n_eye_data_points = np.count_nonzero(np.logical_and(t_inds >= lagged_trace_win[0], t_inds < lagged_trace_win[1]))
    X = np.zeros((n_eye_data_points, 6))
    if (blockname == "Learning") and (trial_type == "inst_learn"):
        if neuron_dict['Learning']['learning']['fr'].shape[0] == 0:
            # No data
            return np.full((neuron_dict['trace_win'][1] - neuron_dict['trace_win'][0], ), np.nan)
        X[:, 0] = np.nanmean(get_learn_response_inst_probe_traces_win(neuron_dict, "eyep_p", lagged_trace_win, trial_win=trial_win), axis=0)
        X[:, 1] = np.nanmean(get_learn_response_inst_probe_traces_win(neuron_dict, "eyep_l", lagged_trace_win, trial_win=trial_win), axis=0)
        X[:, 2] = np.nanmean(get_learn_response_inst_probe_traces_win(neuron_dict, "eyev_p", lagged_trace_win, trial_win=trial_win), axis=0)
        X[:, 3] = np.nanmean(get_learn_response_inst_probe_traces_win(neuron_dict, "eyev_l", lagged_trace_win, trial_win=trial_win), axis=0)
    else:
        if neuron_dict[blockname][trial_type]['fr'].shape[0] == 0:
            # No data
            return np.full((neuron_dict['trace_win'][1] - neuron_dict['trace_win'][0], ), np.nan)
        X[:, 0] = np.nanmean(get_traces_win(neuron_dict, blockname, trial_type, "eyep_p", lagged_trace_win, trial_win=trial_win), axis=0)
        X[:, 1] = np.nanmean(get_traces_win(neuron_dict, blockname, trial_type, "eyep_l", lagged_trace_win, trial_win=trial_win), axis=0)
        X[:, 2] = np.nanmean(get_traces_win(neuron_dict, blockname, trial_type, "eyev_p", lagged_trace_win, trial_win=trial_win), axis=0)
        X[:, 3] = np.nanmean(get_traces_win(neuron_dict, blockname, trial_type, "eyev_l", lagged_trace_win, trial_win=trial_win), axis=0)

    X[:, 4:6] = eye_data_series.acc_from_vel(X[:,2:4], filter_win=filter_win, axis=0)
    X = piece_wise_eye_data(X[:, 0:6], add_constant=False)
    # Now compute the interaction terms
    X = np.hstack((X, np.zeros((X.shape[0], 8))))
    X[:, 12] = X[:, 4] * X[:, 8]
    X[:, 13] = X[:, 4] * X[:, 9]
    X[:, 14] = X[:, 5] * X[:, 8]
    X[:, 15] = X[:, 5] * X[:, 9]
    X[:, 16] = X[:, 6] * X[:, 10]
    X[:, 17] = X[:, 6] * X[:, 11]
    X[:, 18] = X[:, 7] * X[:, 10]
    X[:, 19] = X[:, 7] * X[:, 11]

    # Now compute y_hat from X and coefficients
    y_hat = np.full((neuron_dict['trace_win'][1] - neuron_dict['trace_win'][0], ), np.nan)
    trace_hat = X @ neuron_dict['lin_model']['coeffs']
    if neuron_dict['lin_model']['eye_lag'] <= 0:
        y_hat[-1 * neuron_dict['lin_model']['eye_lag']:] = trace_hat
    else:
        y_hat[0:-1 * neuron_dict['lin_model']['eye_lag']] = trace_hat
    return y_hat

def get_scatter_data(all_traces, trial_win=None, nansac=False, tuning_win=[200, 300]):
    scatter_data = {}
    scatter_xy = []
    for f_ind, fname in enumerate(all_traces.keys()):
        f_data = all_traces[fname][0]
        # First get tuning and classification data in FIXED TRIAL WINDOW [80, 100]
        base_learn = get_mean_win(f_data, tuning_block, "learning", "fr", tuning_win, nansac=nansac)
        base_anti_learn = get_mean_win(f_data, tuning_block, "anti_learning", "fr", tuning_win, nansac=nansac)
        # base_learn -= base_anti_learn


        base_pursuit = get_mean_win(f_data, tuning_block, "pursuit", "fr", tuning_win, nansac=nansac)
        learn_resp_80 = get_learn_response_inst_probe_mean_win(f_data, "fr", tuning_win, trial_win=[80, 100], nansac=nansac)
        learn_resp_80 -= base_pursuit
        # Now get predicted learning response minus observed during the INPUT TRIAL WINDOW!
        base_pursuit_hat = get_mean_win(f_data, tuning_block, "pursuit", "y_hat", tuning_win, nansac=nansac)
        learn_resp_hat = get_learn_response_inst_probe_mean_win(f_data, "y_hat", tuning_win, trial_win=trial_win, nansac=nansac)
        learn_resp_hat -= base_pursuit_hat
        learn_resp_trial = get_learn_response_inst_probe_mean_win(f_data, "fr", tuning_win, trial_win=trial_win, nansac=nansac)
        learn_resp_trial -= base_pursuit
        learn_obs_minus_act = learn_resp_trial - learn_resp_hat

        scatter_data[fname] = (np.array([base_learn, learn_resp_80, learn_obs_minus_act]), int(f_ind))
        scatter_xy.append(scatter_data[fname][0])
    scatter_xy = np.vstack(scatter_xy)

    return scatter_data, scatter_xy

def get_mean_traces(all_traces, nansac=False, y_hat_from_avg=True, trial_win=None, trial_win_late=None):
    # Build our dictionary of mean traces for all the units
    mean_traces = {}
    for fname in all_traces.keys():
        f_data = all_traces[fname][0]
        mean_traces[fname] = {}

        # Get the raw learning response eye velocity
        use_dtype = "eyev_l"
        mean_traces[fname]['learn_eye_raw'] = np.nanmean(get_learn_response_inst_probe_traces_win(f_data, use_dtype, trial_win=trial_win), axis=0)
        mean_traces[fname]['base_pursuit_eye'] = np.nanmean(get_traces_win(f_data, tuning_block, "pursuit", use_dtype, f_data['trace_win']), axis=0)
        mean_traces[fname]['base_learn_eye'] = np.nanmean(get_traces_win(f_data, tuning_block, "learning", use_dtype, f_data['trace_win']), axis=0)
        # Get the raw learning response firing rate
        use_dtype = "fr"
        mean_traces[fname]['learn_fr_raw'] = np.nanmean(get_learn_response_inst_probe_traces_win(f_data, use_dtype, 
                                                                            trial_win=trial_win, nansac=nansac), axis=0)
        mean_traces[fname]['base_pursuit_fr'] = np.nanmean(get_traces_win(f_data, tuning_block, "pursuit", use_dtype, 
                                                                          f_data['trace_win'], nansac=nansac), axis=0)
        mean_traces[fname]['base_learn_fr'] = np.nanmean(get_traces_win(f_data, tuning_block, "learning", use_dtype, 
                                                                        f_data['trace_win'], nansac=nansac), axis=0)
        if trial_win_late is None:
            mean_traces[fname]['learn_fr_raw_late'] = np.full(mean_traces[fname]['learn_fr_raw'].shape, np.nan)
        else:
            mean_traces[fname]['learn_fr_raw_late'] = np.nanmean(get_learn_response_inst_probe_traces_win(f_data, use_dtype, 
                                                                            trial_win=trial_win_late, nansac=nansac), axis=0)
        # Get the actual RAW RAW rates
        # Get the raw learning response firing rate
        use_dtype = "fr_raw"
        mean_traces[fname]['learn_fr_raw_raw'] = np.nanmean(get_learn_response_inst_probe_traces_win(f_data, use_dtype, 
                                                                            trial_win=trial_win, nansac=nansac), axis=0)
        mean_traces[fname]['base_pursuit_fr_raw'] = np.nanmean(get_traces_win(f_data, tuning_block, "pursuit", use_dtype, 
                                                                              f_data['trace_win'], nansac=nansac), axis=0)
        mean_traces[fname]['base_learn_fr_raw'] = np.nanmean(get_traces_win(f_data, tuning_block, "learning", use_dtype, 
                                                                            f_data['trace_win'], nansac=nansac), axis=0)
        if trial_win_late is None:
            mean_traces[fname]['learn_fr_raw_raw_late'] = np.full(mean_traces[fname]['learn_fr_raw_raw'].shape, np.nan)
        else:
            mean_traces[fname]['learn_fr_raw_raw_late'] = np.nanmean(get_learn_response_inst_probe_traces_win(f_data, use_dtype, 
                                                                            trial_win=trial_win_late, nansac=nansac), axis=0)
        # then for model prediction
        if y_hat_from_avg:
            mean_traces[fname]['learn_fr_hat'] = get_y_hat_from_avg(f_data, "Learning", "inst_learn", trial_win=trial_win, filter_win=101)
            mean_traces[fname]['base_pursuit_hat'] = get_y_hat_from_avg(f_data, tuning_block, "pursuit", trial_win=None, filter_win=101)
            mean_traces[fname]['base_learn_hat'] = get_y_hat_from_avg(f_data, tuning_block, "learning", trial_win=None, filter_win=101)
            if trial_win_late is None:
                mean_traces[fname]['learn_fr_hat_late'] = np.full(mean_traces[fname]['learn_fr_hat'].shape, np.nan)
            else:
                mean_traces[fname]['learn_fr_hat_late'] = get_y_hat_from_avg(f_data, "Learning", "inst_learn", trial_win=trial_win_late, filter_win=101)
        else:
            use_dtype = "y_hat"
            mean_traces[fname]['learn_fr_hat'] = np.nanmean(get_learn_response_inst_probe_traces_win(f_data, use_dtype, 
                                                                            trial_win=trial_win, nansac=nansac), axis=0)
            mean_traces[fname]['base_pursuit_hat'] = np.nanmean(get_traces_win(f_data, tuning_block, "pursuit", use_dtype, 
                                                                               f_data['trace_win'], nansac=nansac), axis=0)
            mean_traces[fname]['base_learn_hat'] = np.nanmean(get_traces_win(f_data, tuning_block, "learning", use_dtype, 
                                                                             f_data['trace_win'], nansac=nansac), axis=0)
            if trial_win_late is None:
                mean_traces[fname]['learn_fr_hat_late'] = np.full(mean_traces[fname]['learn_fr_hat'].shape, np.nan)
            else:
                mean_traces[fname]['learn_fr_hat_late'] = np.nanmean(get_learn_response_inst_probe_traces_win(f_data, use_dtype, 
                                                                            trial_win=trial_win_late, nansac=nansac), axis=0)

    return mean_traces

def gather_traces(mean_traces, fname, subtract_baseline=False):
    traces = {}
    # Get response change from baseline
    if mean_traces[fname]['learn_eye_raw'].size == 0:
        nan_out = np.full((mean_traces[fname]['base_learn_eye'].shape[0], ), np.nan)
        traces['sel_learn_eye'] = nan_out
        traces['sel_learn_fr'] = nan_out
        traces['sel_learn_fr_raw'] = nan_out
        traces['sel_learn_hat'] = nan_out

        traces['sel_learn_fr_late'] = nan_out
        traces['sel_learn_fr_raw_late'] = nan_out
        traces['sel_learn_hat_late'] = nan_out
    else:
        traces['sel_learn_eye'] = mean_traces[fname]['learn_eye_raw']
        if subtract_baseline:
            traces['sel_learn_eye'] -= mean_traces[fname]['base_pursuit_eye']
        traces['sel_learn_fr'] = mean_traces[fname]['learn_fr_raw']
        if subtract_baseline:
            traces['sel_learn_fr'] -= mean_traces[fname]['base_pursuit_fr']
        traces['sel_learn_fr_raw'] = mean_traces[fname]['learn_fr_raw_raw']
        if subtract_baseline:
            traces['sel_learn_fr_raw'] -= mean_traces[fname]['base_pursuit_fr_raw']
        traces['sel_learn_hat'] = mean_traces[fname]['learn_fr_hat']
        if subtract_baseline:
            traces['sel_learn_hat'] -= mean_traces[fname]['base_pursuit_hat']

        traces['sel_learn_fr_late'] = mean_traces[fname]['learn_fr_raw_late']
        if subtract_baseline:
            traces['sel_learn_fr_late'] -= mean_traces[fname]['base_pursuit_fr']
        traces['sel_learn_fr_raw_late'] = mean_traces[fname]['learn_fr_raw_raw_late']
        if subtract_baseline:
            traces['sel_learn_fr_raw_late'] -= mean_traces[fname]['base_pursuit_fr_raw']
        traces['sel_learn_hat_late'] = mean_traces[fname]['learn_fr_hat_late']
        if subtract_baseline:
            traces['sel_learn_hat_late'] -= mean_traces[fname]['base_pursuit_hat']

    traces['sel_base_learn_eye'] = mean_traces[fname]['base_learn_eye']
    traces['sel_base_learn_fr'] = mean_traces[fname]['base_learn_fr']
    traces['sel_base_learn_fr_raw'] = mean_traces[fname]['base_learn_fr_raw']
    traces['sel_base_learn_hat'] = mean_traces[fname]['base_learn_hat']
    traces['sel_base_pursuit_fr'] = mean_traces[fname]['base_pursuit_fr']
    traces['sel_base_pursuit_fr_raw'] = mean_traces[fname]['base_pursuit_fr_raw']
    traces['sel_base_pursuit_hat'] = mean_traces[fname]['base_pursuit_hat']

    return traces

def copy_sel_traces(traces, sel_traces):
    sel_traces['sel_learn_eye'].append(traces['sel_learn_eye'])
    sel_traces['sel_base_learn_eye'].append(traces['sel_base_learn_eye'])
    sel_traces['sel_learn_fr'].append(traces['sel_learn_fr'])
    sel_traces['sel_learn_hat'].append(traces['sel_learn_hat'])
    sel_traces['sel_base_learn_fr'].append(traces['sel_base_learn_fr'])
    sel_traces['sel_base_learn_hat'].append(traces['sel_base_learn_hat'])
    sel_traces['sel_base_pursuit_fr'].append(traces['sel_base_pursuit_fr'])
    sel_traces['sel_base_pursuit_hat'].append(traces['sel_base_pursuit_hat'])
    sel_traces['sel_learn_fr_raw'].append(traces['sel_learn_fr_raw'])
    sel_traces['sel_base_learn_fr_raw'].append(traces['sel_base_learn_fr_raw'])
    sel_traces['sel_base_pursuit_fr_raw'].append(traces['sel_base_pursuit_fr_raw'])
    sel_traces['sel_learn_fr_late'].append(traces['sel_learn_fr_late'])
    sel_traces['sel_learn_fr_raw_late'].append(traces['sel_learn_fr_raw_late'])
    sel_traces['sel_learn_hat_late'].append(traces['sel_learn_hat_late'])

def select_neuron_traces(all_traces, scatter_data, modulation_threshold, way, trial_win=None, nansac=True, trial_win_late=None,
                         subtract_baseline=False):
    """ Selects and returns the mean traces for each neuron in "all_traces" that satisfy the input selection
    criteria according to the data in "scatter_data".
    """
    mean_traces = get_mean_traces(all_traces, nansac=nansac, y_hat_from_avg=True, trial_win=trial_win, trial_win_late=trial_win_late)
    
    learn_dir = "off" if modulation_threshold <= 0. else "on"
    sel_traces = {
        'sel_learn_eye': [],
        'sel_base_learn_eye': [],
        'sel_learn_fr': [],
        'sel_learn_fr_late': [],
        'sel_learn_hat': [],
        'sel_base_learn_fr': [],
        'sel_base_learn_hat': [],
        'sel_base_pursuit_fr': [],
        'sel_base_pursuit_hat': [],
        'plotted_fnames': [],
        'sel_learn_fr_raw': [],
        'sel_learn_fr_raw_late': [],
        'sel_base_learn_fr_raw': [],
        'sel_base_pursuit_fr_raw': [],
        'sel_learn_hat_late': [],
    }

    for fname in mean_traces.keys():
        if learn_dir == "off":
            if scatter_data[fname][0][0] <= modulation_threshold:
                # Learning off direction
                if way == "right":
                    # Look for right-way learners
                    if np.sign(scatter_data[fname][0][1]) <= 0:
                        # Get response change from baseline
                        traces = gather_traces(mean_traces, fname, subtract_baseline=subtract_baseline)
                        copy_sel_traces(traces, sel_traces)
                        sel_traces['plotted_fnames'].append(fname)
                else:
                    # Look for wrong-way learners
                    if np.sign(scatter_data[fname][0][1]) > 0:
                        # Get response change from baseline
                        traces = gather_traces(mean_traces, fname, subtract_baseline=subtract_baseline)
                        copy_sel_traces(traces, sel_traces)
                        sel_traces['plotted_fnames'].append(fname)
        if learn_dir == "on":
            if scatter_data[fname][0][0] >= modulation_threshold:
                # Learning on direction
                if way == "right":
                    # Look for right-way learners
                    if np.sign(scatter_data[fname][0][1]) >= 0:
                        # Get response change from baseline
                        traces = gather_traces(mean_traces, fname, subtract_baseline=subtract_baseline)
                        copy_sel_traces(traces, sel_traces)
                        sel_traces['plotted_fnames'].append(fname)
                else:
                    # Look for wrong-way learners
                    if np.sign(scatter_data[fname][0][1]) < 0:
                        # Get response change from baseline
                        traces = gather_traces(mean_traces, fname, subtract_baseline=subtract_baseline)
                        copy_sel_traces(traces, sel_traces)
                        sel_traces['plotted_fnames'].append(fname)
    return sel_traces

def make_trace_data_figs(traces_fname, savename, modulation_threshold, way="right", trial_win=[80, 100], 
                         tuning_win=[200, 300], trial_win_late=[480, 500], save_fyp=False):
    """ Loads the all traces data file "fname" and makes plots for all the different neuron conditions
    and saves as a PDF.
    """
    plot_raw = False # This plots raw traces but currently DOES NOT SELECT BASED ON RAW SCATTERPLOTS
    subtract_baseline = True
    light_gray = [.75, .75, .75]
    dark_gray = [.25, .25, .25]
    plotted_col = "green"
    obs_trace_col = "black"
    obs_trace_col_ci = light_gray
    pred_trace_col = "red"
    pred_trace_col_ci = [0.8, 0.2, 0.2]
    base_dot_size = 15

    way = way.lower()
    learn_dir = "off" if modulation_threshold <= 0. else "on"
    if save_fyp:
        fig_handles, plot_handles = setup_figcomp_figs()
    else:
        plot_handles = setup_axes()
    with open(traces_fname, 'rb') as fp:
        all_traces = pickle.load(fp)
    scatter_data, scatter_xy = get_scatter_data(all_traces, trial_win=trial_win, tuning_win=tuning_win)
    sel_traces = select_neuron_traces(all_traces, scatter_data, modulation_threshold, way, 
                                      trial_win=trial_win, trial_win_late=trial_win_late,
                                      subtract_baseline=subtract_baseline)
    # sel_traces_late = select_neuron_traces(all_traces, scatter_data, modulation_threshold, way, trial_win=trial_win_late)

    # Make plots
    if not save_fyp:
        plot_handles['fig'].suptitle(f"{way.capitalize()} way learning SS-{learn_dir.upper()} >= {modulation_threshold} spk/s baseline pursuit learning axis modulation({len(sel_traces['plotted_fnames'])} PCs)", 
                                    fontsize=11, y=.99)
    # Get scatterplot indices for the files we kept
    plotted_inds = np.zeros(scatter_xy.shape[0], dtype='bool')
    for fname in sel_traces['plotted_fnames']:
        plotted_inds[scatter_data[fname][1]] = True
        trace_win = all_traces[fname][0]['trace_win']
    plot_handles['scatter_raw'].scatter(scatter_xy[~plotted_inds, 0], scatter_xy[~plotted_inds, 1],
                                           color=light_gray, s=base_dot_size, zorder=1)
    plot_handles['scatter_raw'].scatter(scatter_xy[plotted_inds, 0], scatter_xy[plotted_inds, 1],
                                           edgecolors=plotted_col, facecolors='none', s=base_dot_size, zorder=1)
    if not save_fyp:
        plot_handles['scatter_raw'].axvline(0, color=dark_gray, zorder=0)
        plot_handles['scatter_raw'].axhline(0, color=dark_gray, zorder=0)
    plot_handles['scatter_raw'].set_xticks(np.arange(-75, 76, 25))
    plot_handles['scatter_raw'].set_yticks(np.arange(-30, 31, 10))
    plot_handles['scatter_raw'].set_xlim([-80, 80])
    plot_handles['scatter_raw'].set_ylim([-30, 30])
    plot_handles['scatter_raw'].set_xlabel("Baseline learning axis respone (spk/s)", fontsize=8)
    plot_handles['scatter_raw'].set_ylabel("Learning response (spk/s) \n [instruction trial - baseline pursuit]", fontsize=8)
    plot_handles['scatter_raw'].tick_params(axis='both', which='major', labelsize=9)
    plot_handles['scatter_raw'].set_title(f"Learning vs. tuning PC firing rates from \n {tuning_win[0]}-{tuning_win[1]} ms after target onset", 
                                          fontsize=9, y=1.01)
    

    plot_handles['scatter_learned'].scatter(scatter_xy[~plotted_inds, 0], scatter_xy[~plotted_inds, 2],
                                           color=light_gray, s=base_dot_size, zorder=1)
    plot_handles['scatter_learned'].scatter(scatter_xy[plotted_inds, 0], scatter_xy[plotted_inds, 2],
                                           edgecolors=plotted_col, facecolors='none', s=base_dot_size, zorder=1)
    if not save_fyp:
        plot_handles['scatter_learned'].axvline(0, color=dark_gray, zorder=0)
        plot_handles['scatter_learned'].axhline(0, color=dark_gray, zorder=0)
    plot_handles['scatter_learned'].set_xticks(np.arange(-75, 76, 25))
    plot_handles['scatter_learned'].set_yticks(np.arange(-30, 31, 10))
    plot_handles['scatter_learned'].set_xlim([-80, 80])
    plot_handles['scatter_learned'].set_ylim([-30, 30])
    plot_handles['scatter_learned'].set_xlabel("Baseline learning axis respone (spk/s)", fontsize=8)
    plot_handles['scatter_learned'].set_ylabel("Observed minus expected \n learning response (spk/s)", fontsize=8)
    plot_handles['scatter_learned'].tick_params(axis='both', which='major', labelsize=9)
    plot_handles['scatter_learned'].set_yticklabels([])
    plot_handles['scatter_learned'].set_title(f"Observed - predicted learning vs. tuning PC firing rates \n from {tuning_win[0]}-{tuning_win[1]} ms after target onset",
                                              fontsize=9, y=1.01)

    set_all_axis_same([plot_handles[x] for x in ["scatter_raw", "scatter_learned"]])

    if plot_raw:
        plot_trace_name = "sel_learn_fr_raw"
    else:
        plot_trace_name = "sel_learn_fr"
    # plot_handles['learn_80_fr'].plot(np.arange(trace_win[0], trace_win[1]), np.nanmean(np.vstack(sel_learn_fr), axis=0), color=obs_trace_col, label="observed")
    mean_trace = np.nanmean(np.vstack(sel_traces[plot_trace_name]), axis=0)
    sem_trace = nansem(sel_traces[plot_trace_name], axis=0)
    plot_95_CI(plot_handles['learn_80_fr'], np.arange(trace_win[0], trace_win[1]), mean_trace, sem_trace, mean_color=obs_trace_col, 
               ci_color=obs_trace_col_ci, mean_label="Observed FR", ci_label="95% CI", ci_linewidth=1)
    if not plot_raw:
        plot_handles['learn_80_fr'].plot(np.arange(trace_win[0], trace_win[1]), np.nanmean(np.vstack(sel_traces['sel_learn_hat']), axis=0), color=pred_trace_col, label="linear model")

    plot_handles['learn_80_fr'].set_xlabel("Time from target motion onset (ms)", fontsize=8)
    plot_handles['learn_80_fr'].set_ylabel("FR rate change from fixation (spk/s)", fontsize=8)
    if not save_fyp:
        plot_handles['learn_80_fr'].axvline(0, color=dark_gray, zorder=0)
        plot_handles['learn_80_fr'].axvline(250, color=dark_gray, zorder=0)
    plot_handles['learn_80_fr'].set_title(f"Learning block firing rate on instruction trials {trial_win}", fontsize=9)
    plot_handles['learn_80_fr'].set_xticks(np.arange(0, 900, 200))
    # plot_handles['learn_80_fr'].set_yticks(np.arange(-2, 21, 2))
    plot_handles['learn_80_fr'].set_xlim(trace_win)
    # ylim = [-2, 20]
    ylim = plot_handles['learn_80_fr'].get_ylim()
    ylim = list(ylim)
    ylim[0] = round_to_nearest_five_greatest(ylim[0], round_n=5)
    ylim[1] = round_to_nearest_five_greatest(ylim[1], round_n=5)
    plot_handles['learn_80_fr'].set_ylim(ylim)
    plot_handles['learn_80_fr'].tick_params(axis='both', which='major', labelsize=9)
    # plot_handles['learn_80_fr'].fill_betweenx(ylim, tuning_win[0], tuning_win[1], color=light_gray, alpha=1., zorder=-10)

    if plot_raw:
        plot_trace_name = "sel_learn_fr_raw_late"
    else:
        plot_trace_name = "sel_learn_fr_late"
    # plot_handles['learn_480_fr'].plot(np.arange(trace_win[0], trace_win[1]), np.nanmean(np.vstack(sel_learn_fr), axis=0), color=obs_trace_col, label="observed")
    mean_trace = np.nanmean(np.vstack(sel_traces[plot_trace_name]), axis=0)
    sem_trace = nansem(sel_traces[plot_trace_name], axis=0)
    plot_95_CI(plot_handles['learn_480_fr'], np.arange(trace_win[0], trace_win[1]), mean_trace, sem_trace, mean_color=obs_trace_col, 
               ci_color=obs_trace_col_ci, mean_label="Observed FR", ci_label="95% CI", ci_linewidth=1)

    if not plot_raw:
        plot_handles['learn_480_fr'].plot(np.arange(trace_win[0], trace_win[1]), np.nanmean(np.vstack(sel_traces['sel_learn_hat_late']), axis=0), color=pred_trace_col, label="linear model")

    plot_handles['learn_480_fr'].set_xlabel("Time from target motion onset (ms)", fontsize=8)
    plot_handles['learn_480_fr'].set_ylabel("FR rate change from fixation (spk/s)", fontsize=8)
    # plot_handles['learn_480_fr'].axvline(0, color=dark_gray, zorder=0)
    # plot_handles['learn_480_fr'].axvline(250, color=dark_gray, zorder=0)
    plot_handles['learn_480_fr'].set_title(f"Learning block firing rate on instruction trials {trial_win_late}", fontsize=9)
    plot_handles['learn_480_fr'].set_xticks(np.arange(0, 900, 200))
    # plot_handles['learn_480_fr'].set_yticks(np.arange(-2, 21, 2))
    plot_handles['learn_480_fr'].set_xlim(trace_win)
    # ylim = [-2, 20]
    ylim = plot_handles['learn_480_fr'].get_ylim()
    ylim = list(ylim)
    ylim[0] = round_to_nearest_five_greatest(ylim[0], round_n=5)
    ylim[1] = round_to_nearest_five_greatest(ylim[1], round_n=5)
    plot_handles['learn_480_fr'].set_ylim(ylim)
    plot_handles['learn_480_fr'].tick_params(axis='both', which='major', labelsize=9)
    # plot_handles['learn_480_fr'].fill_betweenx(ylim, tuning_win[0], tuning_win[1], color=light_gray, alpha=1., zorder=-10)

    if plot_raw:
        plot_trace_name = "sel_base_learn_fr_raw"
    else:
        plot_trace_name = "sel_base_learn_fr"
    # plot_handles['learn_base_fr'].plot(np.arange(trace_win[0], trace_win[1]), np.nanmean(np.vstack(sel_base_learn_fr), axis=0), color=obs_trace_col, label="observed")
    mean_trace = np.nanmean(np.vstack(sel_traces[plot_trace_name]), axis=0)
    sem_trace = nansem(sel_traces[plot_trace_name], axis=0)
    plot_95_CI(plot_handles['learn_base_fr'], np.arange(trace_win[0], trace_win[1]), mean_trace, sem_trace, mean_color=obs_trace_col, 
               ci_color=obs_trace_col_ci, mean_label="Observed FR", ci_label="95% CI", ci_linewidth=1)

    if not plot_raw:
        plot_handles['learn_base_fr'].plot(np.arange(trace_win[0], trace_win[1]), np.nanmean(np.vstack(sel_traces['sel_base_learn_hat']), axis=0), color=pred_trace_col, label="linear model")
    plot_handles['learn_base_fr'].set_xlabel("Time from target motion onset (ms)", fontsize=8)
    plot_handles['learn_base_fr'].set_ylabel("FR rate change from fixation (spk/s)", fontsize=8)
    # plot_handles['learn_base_fr'].axvline(0)
    plot_handles['learn_base_fr'].axvline(100)
    plot_handles['learn_base_fr'].axvline(125)
    plot_handles['learn_base_fr'].axvline(150)
    plot_handles['learn_base_fr'].set_title(f"Baseline tuning learning axis firing rate in learning direction", fontsize=9)
    plot_handles['learn_base_fr'].tick_params(axis='both', which='major', labelsize=9)

    if plot_raw:
        plot_trace_name = "sel_base_pursuit_fr_raw"
    else:
        plot_trace_name = "sel_base_pursuit_fr"
    mean_trace = np.nanmean(np.vstack(sel_traces[plot_trace_name]), axis=0)
    sem_trace = nansem(sel_traces[plot_trace_name], axis=0)
    plot_95_CI(plot_handles['pursuit_base_fr'], np.arange(trace_win[0], trace_win[1]), mean_trace, sem_trace, mean_color=obs_trace_col, 
               ci_color=obs_trace_col_ci, mean_label="Observed FR", ci_label="95% CI", ci_linewidth=1)

    if not plot_raw:
        plot_handles['pursuit_base_fr'].plot(np.arange(trace_win[0], trace_win[1]), np.nanmean(np.vstack(sel_traces['sel_base_pursuit_hat']), axis=0), color=pred_trace_col, label="linear model")
    plot_handles['pursuit_base_fr'].legend()
    plot_handles['pursuit_base_fr'].set_xlabel("Time from target motion onset (ms)", fontsize=8)
    plot_handles['pursuit_base_fr'].set_ylabel("FR rate change from fixation (spk/s)", fontsize=8)
    # plot_handles['pursuit_base_fr'].axvline(0)
    plot_handles['pursuit_base_fr'].axvline(100)
    plot_handles['pursuit_base_fr'].axvline(125)
    plot_handles['pursuit_base_fr'].axvline(150)
    plot_handles['pursuit_base_fr'].set_title(f"Baseline tuning pursuit axis firing rate in pursuit direction", fontsize=9)
    plot_handles['pursuit_base_fr'].tick_params(axis='both', which='major', labelsize=9)

    if save_fyp:
        return fig_handles
    else:
        plt.tight_layout()
        plot_handles['fig'].savefig(savename)
        plt.show()
        return plot_handles


def get_neuron_trace_data(neuron, trace_win, sigma=12.5, cutoff_sigma=4):
    """
    """
    # Some currently hard coded variables
    fix_win = [-100, 50]
    fix_adj_params = {'fix_win': fix_win,
                            'sigma': 12.5,
                            'cutoff_sigma': 4.0,
                            'zscore_sigma': 3.0,
                            'rate_offset': 0.0,
                            }
    use_smooth_fix = True
    use_baseline_block = neuron.session.base_and_tune_blocks['tuning_block']
    model = "pcwise_lin_eye_kinematics_acc_x_vel"
    filter_win = 101

    if neuron.session.blocks['Learning'] is None:
        # No learning data so nothing to do here
        return {}

    # Get linear model fit
    fit_eye_model = FitNeuronToEye(neuron, trace_win, use_baseline_block, trial_sets=None,
                                    lag_range_eye=[-75, 150])
    if model.lower() == "pcwise_lin_eye_kinematics":
        fit_eye_model.fit_pcwise_lin_eye_kinematics(bin_width=10, bin_threshold=5,
                                                fit_constant=False, fit_avg_data=False,
                                                quick_lag_step=10, fit_fix_adj_fr=True,
                                                filter_win=filter_win, 
                                                fix_adj_params=fix_adj_params)
    elif model.lower() == "pcwise_lin_eye_kinematics_acc_x_vel":
        fit_eye_model.fit_pcwise_lin_eye_kinematics_acc_x_vel(bin_width=1, bin_threshold=1,
                                                    fit_constant=False, fit_avg_data=True,
                                                    quick_lag_step=10, fit_fix_adj_fr=True,
                                                    filter_win=filter_win, 
                                                    fix_adj_params=fix_adj_params)
    else:
        raise ValueError(f"Unrecognized model type {model}")
    # Save model info so we can predict on averages later
    out_traces = {}
    out_traces['lin_model'] = {'coeffs': fit_eye_model.fit_results[model]['coeffs'],
                               'eye_lag': fit_eye_model.fit_results[model]['eye_lag'],
                               }
    out_traces['trace_win'] = trace_win
    
    # Get 4 direction tuning block traces
    all_tuning_blocks = ["StandTunePre", "StabTunePre", "StabTunePost", 
                         "StandTunePost", "StabTuneWash", "StandTuneWash",
                         "BaselinePre", "BaselinePost", "BaselineWash"]
    for tune_block in all_tuning_blocks:
        out_traces[tune_block] = {}
        for trial_type in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
            out_traces[tune_block][trial_type] = {}
            fr, t_inds = neuron.get_firing_traces_fix_adj(trace_win, tune_block, trial_type, 
                                                        fix_time_window=fix_win, sigma=sigma, 
                                                        cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                        rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                                        return_inds=True)
            out_traces[tune_block][trial_type]['fr'] = fr
            out_traces[tune_block][trial_type]['t_inds'] = t_inds
            if fr.shape[0] == 0:
                out_traces[tune_block][trial_type]['fr_raw'] = fr
                out_traces[tune_block][trial_type]['y_hat'] = fr
                out_traces[tune_block][trial_type]['eyev_p'] = fr
                out_traces[tune_block][trial_type]['eyev_l'] = fr
                out_traces[tune_block][trial_type]['eyep_p'] = fr
                out_traces[tune_block][trial_type]['eyep_l'] = fr
                continue
            fr_raw = neuron.get_firing_traces(trace_win, tune_block, trial_sets=t_inds)
            out_traces[tune_block][trial_type]['fr_raw'] = fr_raw
            if neuron.name[0:2] == "PC":
                # This is a PC with CS so get them
                out_traces[tune_block][trial_type]['cs'] = neuron.get_CS_dataseries_by_trial(trace_win, tune_block, None)
            X_eye, x_shape = fit_eye_model.get_data_by_trial(model, tune_block, t_inds, 
                                                                                        return_shape=True, return_inds=False)
            out_traces[tune_block][trial_type]['y_hat'] = fit_eye_model.predict_by_trial(model, X_eye, x_shape)
            eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", trace_win, blocks=tune_block,
                                                            trial_sets=t_inds, return_inds=False)
            out_traces[tune_block][trial_type]['eyev_p'] = eyev_p
            out_traces[tune_block][trial_type]['eyev_l'] = eyev_l
            eyep_p, eyep_l = neuron.session.get_xy_traces("eye position", trace_win, blocks=tune_block,
                                                            trial_sets=t_inds, return_inds=False)
            out_traces[tune_block][trial_type]['eyep_p'] = eyep_p
            out_traces[tune_block][trial_type]['eyep_l'] = eyep_l
    
    # Now get the Learning block traces
    out_traces['Learning'] = {}
    for trial_type in ["instruction", "learning", "anti_pursuit", "pursuit", "anti_learning"]:
        out_traces['Learning'][trial_type] = {}
        fr, t_inds = neuron.get_firing_traces_fix_adj(trace_win, "Learning", trial_type, 
                                                    fix_time_window=fix_win, sigma=sigma, 
                                                    cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                    rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                                    return_inds=True)
        out_traces['Learning'][trial_type]['fr'] = fr
        out_traces['Learning'][trial_type]['t_inds'] = t_inds
        if len(t_inds) == 0:
            out_traces['Learning'][trial_type]['fr_raw'] = fr
            out_traces['Learning'][trial_type]['y_hat'] = fr
            out_traces['Learning'][trial_type]['n_inst'] = t_inds
            out_traces['Learning'][trial_type]['eyev_p'] = fr
            out_traces['Learning'][trial_type]['eyev_l'] = fr
            out_traces['Learning'][trial_type]['eyep_p'] = fr
            out_traces['Learning'][trial_type]['eyep_l'] = fr
            continue
        fr_raw = neuron.get_firing_traces(trace_win, "Learning", trial_sets=t_inds)
        out_traces['Learning'][trial_type]['fr_raw'] = fr_raw
        if neuron.name[0:2] == "PC":
            # This is a PC with CS so get them
            out_traces['Learning'][trial_type]['cs'] = neuron.get_CS_dataseries_by_trial(trace_win, "Learning", t_inds)
        X_eye, x_shape = fit_eye_model.get_data_by_trial(model, "Learning", t_inds, 
                                                                                        return_shape=True, return_inds=False)
        out_traces['Learning'][trial_type]['y_hat'] = fit_eye_model.predict_by_trial(model, X_eye, x_shape)
        n_inst = np.array([neuron.session.n_instructed[t_ind] for t_ind in t_inds], dtype=np.int64)
        out_traces['Learning'][trial_type]['n_inst'] = n_inst
        eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", trace_win, blocks="Learning",
                                                            trial_sets=t_inds, return_inds=False)
        out_traces['Learning'][trial_type]['eyev_p'] = eyev_p
        out_traces['Learning'][trial_type]['eyev_l'] = eyev_l
        eyep_p, eyep_l = neuron.session.get_xy_traces("eye position", trace_win, blocks="Learning",
                                                            trial_sets=t_inds, return_inds=False)
        out_traces['Learning'][trial_type]['eyep_p'] = eyep_p
        out_traces['Learning'][trial_type]['eyep_l'] = eyep_l

    # And the Washout block
    out_traces['Washout'] = {}
    out_traces['Washout']['instruction'] = {}
    fr, t_inds = neuron.get_firing_traces_fix_adj(trace_win, "Washout", "instruction", 
                                                fix_time_window=fix_win, sigma=sigma, 
                                                cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                                return_inds=True)
    out_traces['Washout']["instruction"]['fr'] = fr
    out_traces['Washout']["instruction"]['t_inds'] = t_inds
    if len(t_inds) == 0:
        out_traces['Washout']["instruction"]['fr_raw'] = fr
        out_traces['Washout']["instruction"]['y_hat'] = fr
        out_traces['Washout']["instruction"]['n_inst'] = t_inds
        out_traces['Washout']["instruction"]['eyev_p'] = fr
        out_traces['Washout']["instruction"]['eyev_l'] = fr
        out_traces['Washout']['instruction']['eyep_p'] = fr
        out_traces['Washout']['instruction']['eyep_l'] = fr
    else:
        fr_raw = neuron.get_firing_traces(trace_win, "Washout", trial_sets=t_inds)
        out_traces['Washout']['instruction']['fr_raw'] = fr_raw
        if neuron.name[0:2] == "PC":
            # This is a PC with CS so get them
            out_traces['Washout']["instruction"]['cs'] = neuron.get_CS_dataseries_by_trial(trace_win, "Washout", t_inds)
        X_eye, x_shape = fit_eye_model.get_data_by_trial(model, "Washout", t_inds, 
                                                                                        return_shape=True, return_inds=False)
        out_traces['Washout']["instruction"]['y_hat'] = fit_eye_model.predict_by_trial(model, X_eye, x_shape)
        n_inst = np.array([neuron.session.n_instructed[t_ind] for t_ind in t_inds], dtype=np.int64)
        out_traces['Washout']["instruction"]['n_inst'] = n_inst
        eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", trace_win, blocks="Washout",
                                                            trial_sets=t_inds, return_inds=False)
        out_traces['Washout']["instruction"]['eyev_p'] = eyev_p
        out_traces['Washout']["instruction"]['eyev_l'] = eyev_l
        eyep_p, eyep_l = neuron.session.get_xy_traces("eye position", trace_win, blocks="Washout",
                                                            trial_sets=t_inds, return_inds=False)
        out_traces['Washout']["instruction"]['eyep_p'] = eyep_p
        out_traces['Washout']["instruction"]['eyep_l'] = eyep_l

    return out_traces