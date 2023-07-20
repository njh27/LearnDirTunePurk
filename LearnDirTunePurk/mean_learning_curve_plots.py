import numpy as np
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
                    "anti_learning": "red",
                    "pursuit": "orange",
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
                    3: "fix_learn",
                    4: "learn_500-100",
                    5: "fix_learn_500",
                    }
# Hard coded used globally rightnow
t_title_pad = 0
tuning_block = "StabTunePre"
tuning_win = [200, 300]
fixation_win = [-300, 50]

def setup_axes():
    plot_handles = {}
    plot_handles['fig'], ax_handles = plt.subplots(3, 2, figsize=(8, 11))
    ax_handles = ax_handles.ravel()
    for ax_ind in range(0, ax_handles.size):
        plot_handles[ax_inds_to_names[ax_ind]] = ax_handles[ax_ind]
        # plot_handles[ax_inds_to_names[ax_ind]].set_aspect('equal')

    return plot_handles


def get_scatter_points_win(f_data, block, trial_type, data_type, time_win):
    """ Gets x-y scatterplot data for the inputs where x data is the trial number or the 
    n-instructed number if "Learning" block is input.
    """
    if block == "Learning":
        x_data = f_data[block][trial_type]['n_inst']
    else:
        x_data = f_data[block][trial_type]['t_inds']
    y_data = lept.get_traces_win(f_data, block, trial_type, data_type, time_win)
    y_data = np.nanmean(y_data, axis=1)
    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError(f"x and y scatter data do not match for block {block} trials {trial_type} with shapes {x_data.shape} and {y_data.shape}")

    return x_data, y_data

def get_instruction_trial_course(all_traces, fnames, data_type, time_win, bin_width=10):
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
        _, y_data = get_scatter_points_win(f_data, tuning_block, "pursuit", data_type, time_win)
        baseline_fr = np.nanmean(y_data)
        x_data, y_data = get_scatter_points_win(f_data, "Learning", "instruction", data_type, time_win)
        f_t_vals.append(x_data)
        f_frs.append(y_data)
        x_data, y_data = get_scatter_points_win(f_data, "Learning", "pursuit", data_type, time_win)
        f_t_vals.append(x_data)
        f_frs.append(y_data)
        f_t_vals = np.hstack(f_t_vals)
        f_frs = np.hstack(f_frs)
        f_frs -= baseline_fr
        _, frs_binned = bin_x_func_y(f_t_vals, f_frs, bin_edges, y_func=np.nanmean)
        frs[f_ind, 0:frs_binned.size] = frs_binned

    return t_vals, frs

def make_learning_trial_course_figs(traces_fname, savename, modulation_threshold, way="right", bin_width=10):
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

    way = way.lower()
    learn_dir = "off" if modulation_threshold <= 0. else "on"
    plot_handles = setup_axes()
    with open(traces_fname, 'rb') as fp:
        all_traces = pickle.load(fp)
    scatter_data, scatter_xy = lept.get_scatter_data(all_traces, trial_win=[80, 100])
    sel_traces = lept.select_neuron_traces(all_traces, scatter_data, modulation_threshold, way, trial_win=[80, 100])
    plotted_fnames = sel_traces['plotted_fnames']    

    # Make plots
    plot_handles['fig'].suptitle(f"{way.capitalize()} way learning SS-{learn_dir.upper()} >= {modulation_threshold} spk/s baseline pursuit learning axis modulation({len(sel_traces['plotted_fnames'])} PCs)", 
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
    plot_handles['scatter_raw'].set_title(f"Learning vs. tuning pre-learning PC firing rates from \n 200-300 ms after target onset", 
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
    plot_handles['scatter_learned'].set_title(f"80-100 instruction trial observed - predicted learning vs. tuning PC firing rates \n from 200-300 ms after target onset",
                                              fontsize=9, y=1.01)

    set_all_axis_same([plot_handles[x] for x in ["scatter_raw", "scatter_learned"]])

    t_inds, frs = get_instruction_trial_course(all_traces, plotted_fnames, "fr_raw", tuning_win, bin_width=bin_width)
    if frs.shape[0] > 1:
        frs = np.nanmean(frs, axis=0)
    plot_handles['inst_trial_course'].scatter(t_inds, frs, color=light_gray, s=base_dot_size, zorder=1)
    t_inds, frs = get_instruction_trial_course(all_traces, plotted_fnames, "y_hat", tuning_win, bin_width=bin_width)
    if frs.shape[0] > 1:
        frs = np.nanmean(frs, axis=0)
    plot_handles['inst_trial_course'].scatter(t_inds, frs, color=pred_trace_col, s=base_dot_size, zorder=1)
    plot_handles['inst_trial_course'].axhline(0, color=dark_gray, zorder=0)
    plot_handles['inst_trial_course'].set_xticks(np.arange(0, 680+bin_width, 50))
    plot_handles['inst_trial_course'].set_yticks(np.arange(-30, 31, 5))
    plot_handles['inst_trial_course'].set_xlim([0, 680])
    plot_handles['inst_trial_course'].set_ylim([-30, 30])
    plot_handles['inst_trial_course'].set_xlabel("Baseline learning axis respone (spk/s)", fontsize=8)
    plot_handles['inst_trial_course'].set_ylabel("Observed minus expected \n learning response (spk/s)", fontsize=8)
    plot_handles['inst_trial_course'].tick_params(axis='both', which='major', labelsize=9)
    plot_handles['inst_trial_course'].set_yticklabels([])
    plot_handles['inst_trial_course'].set_title(f"80-100 instruction trial observed - predicted learning vs. tuning PC firing rates \n from 200-300 ms after target onset",
                                              fontsize=9, y=1.01)


    plt.tight_layout()
    plot_handles['fig'].savefig(savename)
    plt.show()
    return plot_handles
