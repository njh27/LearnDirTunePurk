import numpy as np
import scipy.stats as stats
import pickle
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from NeuronAnalysis.general import gauss_convolve, bin_x_func_y
import LearnDirTunePurk.learning_eye_PC_traces as lept



# Set some globally available lookup strings by block and trial names
block_strings = {'FixTunePre': "Fixation",
                    'RandVPTunePre': "Random position/velocity tuning",
                    'StandTunePre': "Standard tuning",
                    'StabTunePre': "Stabilized tuning",
                    'Learning': "Learning",
                    'FixTunePost': "Fixation",
                    'StabTunePost': "Stabilized tuning",
                    'Washout': "Opposite learning",
                    'FixTuneWash': "Fixation",
                    'StabTuneWash': "Stabilized tuning",
                    'StandTuneWash': "Standard tuning",
                    }
trial_strings = {'learning': "Learning",
                 'anti_learning': "Anti-learning",
                 'pursuit': "Pursuit",
                 'anti_pursuit': "Anti-pursuit"}
t_set_color_codes = {"learning": "green",
                    "anti_learning": "orange",
                    "pursuit": "gray",
                    "anti_pursuit": "blue"}
tune_adjust_blocks = ["Learning", "StabTunePost",
                      "Washout", "StabTuneWash",
                      "StandTuneWash"]

def set_equal_axlims(ax_handle):
    xlims = ax_handle.get_xlim()
    ylims = ax_handle.get_ylim()
    max_lim = max(abs(num) for num in [*xlims, *ylims])
    ax_handle.set_xlim([-max_lim, max_lim])
    ax_handle.set_ylim([-max_lim, max_lim])


def set_symmetric_axlims(ax_handle):
    xlims = ax_handle.get_xlim()
    max_lim = max(abs(num) for num in xlims)
    ax_handle.set_xlim([-max_lim, max_lim])
    ylims = ax_handle.get_ylim()
    max_lim = max(abs(num) for num in ylims)
    ax_handle.set_ylim([-max_lim, max_lim])


def round_to_nearest_five_greatest(n, round_n=5):
    if n > 0:
        return n + (round_n - n % round_n) if n % round_n != 0 else n
    else:
        return n - (n % round_n) if n % round_n != 0 else n


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

ax_inds_to_names = {0: "scatter_raw",
                    1: "scatter_learned",
                    2: "inst_trial_course",
                    3: "probe_trial_course_lax",
                    4: "probe_trial_course_pax",
                    5: "before_after_learning",
                    }
# Hard coded used globally rightnow
t_title_pad = 0
tuning_block = "StabTunePre"
fixation_win = [-300, 50]

def setup_axes():
    plot_handles = {}
    plot_handles['fig'], ax_handles = plt.subplots(3, 2, figsize=(8, 11))
    ax_handles = ax_handles.ravel()
    for ax_ind in range(0, ax_handles.size):
        plot_handles[ax_inds_to_names[ax_ind]] = ax_handles[ax_ind]
        # plot_handles[ax_inds_to_names[ax_ind]].set_aspect('equal')

    return plot_handles


def get_scatter_points_win(f_data, block, trial_type, data_type, time_win, nansac=True):
    """ Gets x-y scatterplot data for the inputs where x data is the trial number or the 
    n-instructed number if "Learning" block is input.
    """
    if block == "Learning":
        x_data = f_data[block][trial_type]['n_inst']
    else:
        x_data = f_data[block][trial_type]['t_inds']
    y_data = lept.get_traces_win(f_data, block, trial_type, data_type, time_win, nansac=nansac)
    y_data = np.nanmean(y_data, axis=1)
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError(f"x and y scatter data do not match for block {block} trials {trial_type} with shapes {x_data.shape} and {y_data.shape}")

    return x_data, y_data

def get_instruction_trial_course(all_traces, fnames, data_type, time_win, bin_width=10, nansac=True):
    """ Gets the scatterplot data for instruction trials in learning block and bins by bin_width.
    """
    bin_edges = np.arange(0, 680+bin_width, bin_width)
    # Bin edges specify the output t_vals
    t_vals = np.zeros((len(bin_edges) - 1, ))
    for edge in range(0, len(bin_edges) - 1):
        t_vals[edge] = bin_edges[edge] + (bin_edges[edge+1] - bin_edges[edge]) / 2
    # In order to combine and average we need to fill all possible data with nans first
    frs = np.full((len(fnames), len(bin_edges)-1), np.nan)
    for f_ind, fname in enumerate(fnames):
        f_data = all_traces[fname][0]
        f_t_vals = []
        f_frs = []
        _, y_data = get_scatter_points_win(f_data, tuning_block, "pursuit", data_type, time_win, nansac=nansac)
        baseline_fr = np.nanmean(y_data)
        x_data, y_data = get_scatter_points_win(f_data, "Learning", "instruction", data_type, time_win, nansac=nansac)
        f_t_vals.append(x_data)
        f_frs.append(y_data)
        x_data, y_data = get_scatter_points_win(f_data, "Learning", "pursuit", data_type, time_win, nansac=nansac)
        f_t_vals.append(x_data)
        f_frs.append(y_data)
        f_t_vals = np.hstack(f_t_vals)
        f_frs = np.hstack(f_frs)
        f_frs -= baseline_fr
        _, frs_binned = bin_x_func_y(f_t_vals, f_frs, bin_edges, y_func=np.nanmean)
        frs[f_ind, 0:frs_binned.size] = frs_binned

    return t_vals, frs

def get_probe_trial_course(all_traces, fnames, trial_type, data_type, time_win, bin_width=10, nansac=True):
    """ Gets the scatterplot data for instruction trials in learning block and bins by bin_width.
    """
    bin_edges = np.arange(0, 680+bin_width, bin_width)
    # Bin edges specify the output t_vals
    t_vals = np.zeros((len(bin_edges) - 1, ))
    for edge in range(0, len(bin_edges) - 1):
        t_vals[edge] = bin_edges[edge] + (bin_edges[edge+1] - bin_edges[edge]) / 2
    # In order to combine and average we need to fill all possible data with nans first
    frs = np.full((len(fnames), len(bin_edges)-1), np.nan)
    for f_ind, fname in enumerate(fnames):
        f_data = all_traces[fname][0]
        _, y_data = get_scatter_points_win(f_data, tuning_block, trial_type, data_type, time_win, nansac=nansac)
        baseline_fr = np.nanmean(y_data)
        f_t_vals, f_frs = get_scatter_points_win(f_data, "Learning", trial_type, data_type, time_win, nansac=nansac)
        f_frs -= baseline_fr
        _, frs_binned = bin_x_func_y(f_t_vals, f_frs, bin_edges, y_func=np.nanmean)
        frs[f_ind, 0:frs_binned.size] = frs_binned

    return t_vals, frs

def get_pre_post_scatterpoints(all_traces, fnames, preblock, postblock, trial_type, data_type, win, 
                               pre_trial_win=None, post_trial_win=None, min_trials=5, nansac=True):
    """ Gets the pre baseline and post basline average response for all the files according to the input
    filters.
    """
    all_pre = []
    all_post = []
    for f_ind, fname in enumerate(fnames):
        f_data = all_traces[fname][0]
        f_mean_pre = lept.get_mean_win(f_data, preblock, trial_type, data_type, win, trial_win=pre_trial_win, 
                                       min_trials=min_trials, nansac=nansac)
        f_mean_post = lept.get_mean_win(f_data, postblock, trial_type, data_type, win, trial_win=pre_trial_win, 
                                        min_trials=min_trials, nansac=nansac)
        all_pre.append(f_mean_pre)
        all_post.append(f_mean_post)
    all_pre = np.hstack(all_pre)
    all_post = np.hstack(all_post)
    
    return all_pre, all_post

def make_learning_trial_course_figs(traces_fname, savename, modulation_threshold, way="right", bin_width=10, 
                                    nansac=True, tuning_win=[200, 300], fr_dtype="fr"):
    """ Loads the all traces data file "fname" and makes plots for all the different neuron conditions
    and saves as a PDF.
    """
    light_gray = [.75, .75, .75]
    dark_gray = [.25, .25, .25]
    plotted_col = "green"
    obs_trace_col = "black"
    obs_trace_col_ci = light_gray
    pred_trace_col = "red"
    pred_trace_col_ci = [0.8, 0.2, 0.2]
    base_dot_size = 15
    fr_string = "Raw rates" if fr_dtype == "fr_raw" else "Fixation adjusted rates"
    pre_trial_win = [10, np.inf]
    post_trial_win = [0, 10]

    way = way.lower()
    learn_dir = "off" if modulation_threshold <= 0. else "on"
    plot_handles = setup_axes()
    with open(traces_fname, 'rb') as fp:
        all_traces = pickle.load(fp)
    scatter_data, scatter_xy = lept.get_scatter_data(all_traces, trial_win=[80, 100], nansac=nansac, tuning_win=[200, 300])
    sel_traces = lept.select_neuron_traces(all_traces, scatter_data, modulation_threshold, way, trial_win=[80, 100], nansac=nansac)
    plotted_fnames = sel_traces['plotted_fnames']    

    # Make plots
    plot_handles['fig'].suptitle(f"{way.capitalize()} way SS-{learn_dir.upper()} >= {modulation_threshold} spk/s baseline modulation {fr_string} win {tuning_win} ({len(sel_traces['plotted_fnames'])} PCs)", 
                                 fontsize=11, y=.99)
    # Get scatterplot indices for the files we kept
    plotted_inds = np.zeros(scatter_xy.shape[0], dtype='bool')
    for fname in plotted_fnames:
        plotted_inds[scatter_data[fname][1]] = True
        trace_win = all_traces[fname][0]['trace_win']
    plot_handles['scatter_raw'].scatter(scatter_xy[~plotted_inds, 0], scatter_xy[~plotted_inds, 1],
                                           color=light_gray, s=base_dot_size, zorder=1)
    plot_handles['scatter_raw'].scatter(scatter_xy[plotted_inds, 0], scatter_xy[plotted_inds, 1],
                                           edgecolors=plotted_col, facecolors='none', s=base_dot_size, zorder=1)
    plot_handles['scatter_raw'].axvline(0, color=dark_gray, zorder=0)
    plot_handles['scatter_raw'].axhline(0, color=dark_gray, zorder=0)
    plot_handles['scatter_raw'].set_xticks(np.arange(-75, 76, 25))
    plot_handles['scatter_raw'].set_yticks(np.arange(-30, 31, 10))
    plot_handles['scatter_raw'].set_xlim([-80, 80])
    plot_handles['scatter_raw'].set_ylim([-30, 30])
    plot_handles['scatter_raw'].set_xlabel("Baseline learning axis respone (spk/s)", fontsize=8)
    plot_handles['scatter_raw'].set_ylabel("Learning response (spk/s) \n [instruction trial - baseline pursuit]", fontsize=8)
    plot_handles['scatter_raw'].tick_params(axis='both', which='major', labelsize=9)
    plot_handles['scatter_raw'].set_title(f"Learning vs. tuning pre-learning PC firing \n rates from 200-300 ms after target onset", 
                                          fontsize=9, y=1.01)
    
    plot_handles['scatter_learned'].scatter(scatter_xy[~plotted_inds, 0], scatter_xy[~plotted_inds, 2],
                                           color=light_gray, s=base_dot_size, zorder=1)
    plot_handles['scatter_learned'].scatter(scatter_xy[plotted_inds, 0], scatter_xy[plotted_inds, 2],
                                           edgecolors=plotted_col, facecolors='none', s=base_dot_size, zorder=1)
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
    plot_handles['scatter_learned'].set_title(f"80-100 instruction trial observed - predicted learning vs. tuning \n PC firing rates from 200-300 ms after target onset",
                                              fontsize=9, y=1.01)

    set_all_axis_same([plot_handles[x] for x in ["scatter_raw", "scatter_learned"]])

    t_inds, frs = get_instruction_trial_course(all_traces, plotted_fnames, fr_dtype, tuning_win, bin_width=bin_width)
    if frs.shape[0] > 1:
        frs = np.nanmean(frs, axis=0)
    y_max = np.amax(frs)
    y_min = np.amin(frs)
    plot_handles['inst_trial_course'].scatter(t_inds, frs, color=light_gray, s=base_dot_size, zorder=1, 
                                              label="Observed FR")
    t_inds, frs = get_instruction_trial_course(all_traces, plotted_fnames, "y_hat", tuning_win, bin_width=bin_width)
    if frs.shape[0] > 1:
        frs = np.nanmean(frs, axis=0)
    y_max = max(y_max, np.amax(frs))
    y_min = min(y_min, np.amin(frs))
    plot_handles['inst_trial_course'].scatter(t_inds, frs, color=pred_trace_col, s=base_dot_size, zorder=1,
                                              label="Linear model")
    plot_handles['inst_trial_course'].legend()
    plot_handles['inst_trial_course'].axhline(0, color=dark_gray, zorder=0)
    plot_handles['inst_trial_course'].set_xticks(np.arange(0, 680+bin_width, 100))
    plot_handles['inst_trial_course'].set_yticks(np.arange(y_min, y_max+5/2, 5))
    plot_handles['inst_trial_course'].set_xlim([0, 680])
    plot_handles['inst_trial_course'].set_ylim([y_min, y_max])
    plot_handles['inst_trial_course'].set_xlabel("N completed instruction trials", fontsize=8)
    plot_handles['inst_trial_course'].set_ylabel("Firing rate change from baseline (spk/s)", fontsize=8)
    plot_handles['inst_trial_course'].tick_params(axis='both', which='major', labelsize=9)
    # plot_handles['inst_trial_course'].set_yticklabels([])
    plot_handles['inst_trial_course'].set_title(f"Firing rate trial course for instruction trials \n from {tuning_win} ms after target onset",
                                              fontsize=9, y=1.01)
    
    bin_width = 20
    for t_set in ["learning", "anti_learning"]:
        t_inds, frs = get_probe_trial_course(all_traces, plotted_fnames, t_set, fr_dtype, tuning_win, bin_width=bin_width)
        if frs.shape[0] > 1:
            frs = np.nanmean(frs, axis=0)
        y_max = max(y_max, np.amax(frs))
        y_min = min(y_min, np.amin(frs))
        plot_handles['probe_trial_course_lax'].scatter(t_inds, frs, color=t_set_color_codes[t_set], s=base_dot_size, zorder=1,
                                                   label=trial_strings[t_set])
    plot_handles['probe_trial_course_lax'].legend()
    plot_handles['probe_trial_course_lax'].axhline(0, color=dark_gray, zorder=0)
    plot_handles['probe_trial_course_lax'].set_xticks(np.arange(0, 680+bin_width, 100))
    plot_handles['probe_trial_course_lax'].set_yticks(np.arange(y_min, y_max+5/2, 5))
    plot_handles['probe_trial_course_lax'].set_xlim([0, 680])
    plot_handles['probe_trial_course_lax'].set_ylim([y_min, y_max])
    plot_handles['probe_trial_course_lax'].set_xlabel("N completed instruction trials", fontsize=8)
    plot_handles['probe_trial_course_lax'].set_ylabel("Firing rate change from baseline (spk/s)", fontsize=8)
    plot_handles['probe_trial_course_lax'].tick_params(axis='both', which='major', labelsize=9)
    plot_handles['probe_trial_course_lax'].set_yticklabels([])
    plot_handles['probe_trial_course_lax'].set_title(f"Firing rate trial course for probe trials \n from {tuning_win} ms after target onset",
                                              fontsize=9, y=1.01)
    
    for t_set in ["anti_pursuit"]:
        t_inds, frs = get_probe_trial_course(all_traces, plotted_fnames, t_set, fr_dtype, tuning_win, bin_width=bin_width)
        if frs.shape[0] > 1:
            frs = np.nanmean(frs, axis=0)
        y_max = max(y_max, np.amax(frs))
        y_min = min(y_min, np.amin(frs))
        plot_handles['probe_trial_course_pax'].scatter(t_inds, frs, color=t_set_color_codes[t_set], s=base_dot_size, zorder=1,
                                                   label=trial_strings[t_set])
    plot_handles['probe_trial_course_pax'].legend()
    plot_handles['probe_trial_course_pax'].axhline(0, color=dark_gray, zorder=0)
    plot_handles['probe_trial_course_pax'].set_xticks(np.arange(0, 680+bin_width, 100))
    plot_handles['probe_trial_course_pax'].set_yticks(np.arange(y_min, y_max+5/2, 5))
    plot_handles['probe_trial_course_pax'].set_xlim([0, 680])
    plot_handles['probe_trial_course_pax'].set_ylim([y_min, y_max])
    plot_handles['probe_trial_course_pax'].set_xlabel("N completed instruction trials", fontsize=8)
    plot_handles['probe_trial_course_pax'].set_ylabel("Firing rate change from baseline (spk/s)", fontsize=8)
    plot_handles['probe_trial_course_pax'].tick_params(axis='both', which='major', labelsize=9)
    # plot_handles['probe_trial_course_pax'].set_yticklabels([])
    plot_handles['probe_trial_course_pax'].set_title(f"Firing rate trial course for probe trials \n from {tuning_win} ms after target onset",
                                              fontsize=9, y=1.01)
    y_max = round_to_nearest_five_greatest(y_max, round_n=5)
    y_min = round_to_nearest_five_greatest(y_min, round_n=5)
    y_max = min(y_max, 25)
    y_min = max(y_min, -25)
    plot_handles['inst_trial_course'].set_yticks(np.arange(y_min, y_max+5/2, 5))
    plot_handles['probe_trial_course_lax'].set_yticks(np.arange(y_min, y_max+5/2, 5))
    plot_handles['probe_trial_course_pax'].set_yticks(np.arange(y_min, y_max+5/2, 5))
    set_all_axis_same([plot_handles[x] for x in ["inst_trial_course", "probe_trial_course_pax", "probe_trial_course_lax"]])
    
    # fr_dtype = "fr"
    ylim = -np.inf
    for t_set in ["learning", "anti_learning", "pursuit", "anti_pursuit"]:
        fr_pre, fr_post = get_pre_post_scatterpoints(all_traces, plotted_fnames, "StabTunePre", "StabTunePost", t_set, fr_dtype, 
                                                     tuning_win, pre_trial_win=pre_trial_win, post_trial_win=post_trial_win, 
                                                     min_trials=5)
        
        # plot_handles['before_after_learning'].scatter(fr_pre, fr_post, color=t_set_color_codes[t_set], s=base_dot_size, zorder=1,
        #                                            label=trial_strings[t_set])
        delta_fr = fr_post - fr_pre
        y_max = round_to_nearest_five_greatest(np.nanmax(delta_fr))
        y_min = round_to_nearest_five_greatest(np.nanmin(delta_fr))
        ylim = max(ylim, np.abs(y_max), np.abs(y_min))
        plot_handles['before_after_learning'].scatter(scatter_xy[plotted_inds, 0], delta_fr, color=t_set_color_codes[t_set], s=base_dot_size, zorder=1,
                                                   label=trial_strings[t_set])
        # y_max_pre = round_to_nearest_five_greatest(np.nanmax(fr_pre))
        # y_min_pre = round_to_nearest_five_greatest(np.nanmin(fr_pre))
        # y_max_post = round_to_nearest_five_greatest(np.nanmax(fr_post))
        # y_min_post = round_to_nearest_five_greatest(np.nanmin(fr_post))
        # ylim = max(ylim, np.abs(y_max_pre), np.abs(y_min_pre), np.abs(y_max_post), np.abs(y_min_post))
    y_max, y_min = ylim, -1*ylim
    # if fr_dtype == "fr_raw":
    #     y_min = 0
    # unity_line = np.arange(y_min, y_max+1)

    plot_handles['before_after_learning'].set_xticks(np.arange(-75, 76, 25))
    plot_handles['before_after_learning'].set_xlim([-80, 80])
    plot_handles['before_after_learning'].set_xlabel("Baseline learning axis respone (spk/s)", fontsize=8)

    # plot_handles['before_after_learning'].plot(unity_line, unity_line, color='black', zorder=0)
    plot_handles['before_after_learning'].legend()
    plot_handles['before_after_learning'].axvline(0, color=dark_gray, zorder=0)
    plot_handles['before_after_learning'].axhline(0, color=dark_gray, zorder=0)
    # plot_handles['before_after_learning'].set_xticks(np.arange(y_min, y_max+25, 25))
    plot_handles['before_after_learning'].set_yticks(np.arange(y_min, y_max+25, 25))
    # plot_handles['before_after_learning'].set_xlim([y_min, y_max])
    plot_handles['before_after_learning'].set_ylim([y_min, y_max])
    # plot_handles['before_after_learning'].set_xlabel("Baseline learning axis respone (spk/s)", fontsize=8)
    plot_handles['before_after_learning'].set_ylabel("Tuning block trial response change post - pre learning", fontsize=8)
    plot_handles['before_after_learning'].tick_params(axis='both', which='major', labelsize=9)
    plot_handles['before_after_learning'].set_title(f"Change in tuning block responses from pre-learning \n to post-learning window {tuning_win} ms after target onset", 
                                          fontsize=9, y=1.01)


    # fr_dtype = "fr"
    # ylim = -np.inf
    # bar_labels = []
    # bar_vals = []
    # bar_cis = []
    # for t_set in ["learning", "anti_learning", "pursuit", "anti_pursuit"]:
    #     fr_pre, fr_post = get_pre_post_scatterpoints(all_traces, plotted_fnames, "StabTunePre", "StabTunePost", t_set, fr_dtype, 
    #                                                  tuning_win, trial_win=pre_post_trial_win, min_trials=5)
    #     delta = fr_post - fr_pre
    #     bar_labels.append(t_set)
    #     bar_vals.append(np.nanmean(delta))
    #     sem_delta = lept.nansem(delta)
    #     n_obs = np.count_nonzero(~np.isnan(delta))
    #     ci_delta = sem_delta * stats.t.ppf((1 + 0.95) / 2., n_obs - 1)
    #     bar_cis.append(ci_delta)

    # plot_handles['before_after_learning'].bar(bar_labels, bar_vals, yerr=bar_cis, capsize=5, alpha=0.75)

    # y_max, y_min = ylim, -1*ylim
    # if fr_dtype == "fr_raw":
    #     y_min = 0
    # unity_line = np.arange(y_min, y_max+1)
    # plot_handles['before_after_learning'].plot(unity_line, unity_line, color='black', zorder=0)
    # plot_handles['before_after_learning'].legend()
    # plot_handles['before_after_learning'].axvline(0, color=dark_gray, zorder=0)
    # plot_handles['before_after_learning'].axhline(0, color=dark_gray, zorder=0)
    # plot_handles['before_after_learning'].set_xticks(np.arange(y_min, y_max+1, 25))
    # plot_handles['before_after_learning'].set_yticks(np.arange(y_min, y_max+1, 25))
    # plot_handles['before_after_learning'].set_xlim([y_min, y_max])
    # plot_handles['before_after_learning'].set_ylim([y_min, y_max])
    # plot_handles['before_after_learning'].set_xlabel("Baseline learning axis respone (spk/s)", fontsize=8)
    # plot_handles['before_after_learning'].set_ylabel("Learning response (spk/s) \n [instruction trial - baseline pursuit]", fontsize=8)
    # plot_handles['before_after_learning'].tick_params(axis='both', which='major', labelsize=9)
    # plot_handles['before_after_learning'].set_title(f"Learning vs. tuning pre-learning PC firing rates from \n 200-300 ms after target onset", 
    #                                       fontsize=9, y=1.01)


    plt.tight_layout()
    plot_handles['fig'].savefig(savename)
    plt.show()
    return plot_handles
