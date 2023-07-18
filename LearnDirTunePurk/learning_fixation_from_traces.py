import numpy as np
import matplotlib.pyplot as plt
import pickle
import LearnDirTunePurk.learning_eye_PC_traces as lept



ax_inds_to_names = {0: "scatter_raw",
                    1: "scatter_learned",
                    2: "fix_baseline",
                    3: "fix_learn",
                    4: "learn_500-100",
                    5: "learn_500-100_hat",
                    }
# Hard coded used globally rightnow
t_title_pad = 0
tuning_block = "StandTunePre"
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

        
def get_all_neuron_traces(all_traces, trial_win=None):
    """ This is like select_neuron_traces except it does not choose according to modulation, right, wrong,
    it just gets all the data for all units.
    """
    mean_traces = lept.get_mean_traces(all_traces, nan_fr_sac=True, y_hat_from_avg=True, trial_win=trial_win)
    
    sel_learn_eye = []
    sel_base_learn_eye = []
    sel_learn_fr = []
    sel_learn_hat = []
    sel_base_learn_fr = []
    sel_base_learn_hat = []
    sel_base_pursuit_fr = []
    sel_base_pursuit_hat = []
    plotted_fnames = []
    for fname in mean_traces.keys():
        # Get response change from baseline
        try:
            outs = lept.gather_traces(mean_traces, fname)
        except:
            print(fname)
            raise
        sel_learn_eye.append(outs[0])
        sel_base_learn_eye.append(outs[1])
        sel_learn_fr.append(outs[2])
        sel_learn_hat.append(outs[3])
        sel_base_learn_fr.append(outs[4])
        sel_base_learn_hat.append(outs[5])
        sel_base_pursuit_fr.append(outs[6])
        sel_base_pursuit_hat.append(outs[7])
        plotted_fnames.append(fname)
    return sel_learn_eye, sel_base_learn_eye, sel_learn_fr, sel_learn_hat, sel_base_learn_fr, sel_base_learn_hat, sel_base_pursuit_fr, sel_base_pursuit_hat, plotted_fnames

def make_fixation_scatter_figs(traces_fname, savename, modulation_threshold, way="right", trial_win=[80, 100]):
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
    scatter_data, scatter_xy = lept.get_scatter_data(all_traces, trial_win=trial_win)
    sel_traces = lept.select_neuron_traces(all_traces, scatter_data, modulation_threshold, way, trial_win=trial_win)
    _, _, _, _, _, _, _, _, plotted_fnames = sel_traces
    sel_traces = get_all_neuron_traces(all_traces, trial_win=trial_win)
    sel_learn_eye, sel_base_learn_eye, sel_learn_fr, sel_learn_hat, sel_base_learn_fr, sel_base_learn_hat, sel_base_pursuit_fr, sel_base_pursuit_hat, _ = sel_traces
    

    # Make plots
    plot_handles['fig'].suptitle(f"{way.capitalize()} way learning SS-{learn_dir.upper()} >= {modulation_threshold} spk/s baseline pursuit learning axis modulation({len(sel_learn_fr)} PCs)", 
                                 fontsize=11, y=.99)
    # Get scatterplot indices for the files we kept
    plotted_inds = np.zeros(scatter_xy.shape[0], dtype='bool')
    for fname in plotted_fnames:
        plotted_inds[scatter_data[fname][1]] = True
        trace_win = all_traces[fname][0]['trace_win']
    t_inds = np.arange(trace_win[0], trace_win[1])
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
    plot_handles['scatter_raw'].set_title(f"Learning vs. tuning PC firing rates from \n 200-300 ms after target onset", 
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
    plot_handles['scatter_learned'].set_title(f"Observed - predicted learning vs. tuning PC firing rates \n from 200-300 ms after target onset",
                                              fontsize=9, y=1.01)

    set_all_axis_same([plot_handles[x] for x in ["scatter_raw", "scatter_learned"]])

    time_inds = np.logical_and(t_inds >= fixation_win[0], t_inds < fixation_win[1])
    sel_base_learn_fr_scatter = np.nanmean(np.vstack(sel_base_learn_fr)[:, time_inds], axis=1)
    plot_handles['fix_baseline'].scatter(scatter_xy[~plotted_inds, 0], sel_base_learn_fr_scatter[~plotted_inds],
                                           color=light_gray, s=base_dot_size, zorder=1)
    plot_handles['fix_baseline'].scatter(scatter_xy[plotted_inds, 0], sel_base_learn_fr_scatter[plotted_inds],
                                           edgecolors=plotted_col, facecolors='none', s=base_dot_size, zorder=1)
    plot_handles['fix_baseline'].axvline(0, color=dark_gray, zorder=0)
    plot_handles['fix_baseline'].axhline(0, color=dark_gray, zorder=0)
    plot_handles['fix_baseline'].set_xticks(np.arange(-75, 76, 25))
    plot_handles['fix_baseline'].set_yticks(np.arange(-30, 31, 10))
    plot_handles['fix_baseline'].set_xlim([-80, 80])
    plot_handles['fix_baseline'].set_ylim([-30, 30])
    plot_handles['fix_baseline'].set_xlabel("Baseline learning axis respone (spk/s)", fontsize=8)
    plot_handles['fix_baseline'].set_ylabel("Observed minus expected \n learning response (spk/s)", fontsize=8)
    plot_handles['fix_baseline'].tick_params(axis='both', which='major', labelsize=9)
    plot_handles['fix_baseline'].set_yticklabels([])
    plot_handles['fix_baseline'].set_title(f"Observed - predicted learning vs. tuning PC firing rates \n from 200-300 ms after target onset",
                                              fontsize=9, y=1.01)
    
    # time_inds = np.logical_and(t_inds >= win[0], t_inds < win[1])
    # sel_learn_fr_scatter = np.nanmean(sel_learn_fr[:, time_inds], axis=1)
    # plot_handles['fix_learn'].scatter(scatter_xy[~plotted_inds, 0], scatter_xy[~plotted_inds, 2],
    #                                        color=light_gray, s=base_dot_size, zorder=1)
    # plot_handles['fix_learn'].scatter(scatter_xy[plotted_inds, 0], scatter_xy[plotted_inds, 2],
    #                                        edgecolors=plotted_col, facecolors='none', s=base_dot_size, zorder=1)

    plt.tight_layout()
    plot_handles['fig'].savefig(savename)
    plt.show()
    return plot_handles
