import numpy as np
import pickle
import matplotlib.pyplot as plt
from NeuronAnalysis.fit_neuron_to_eye import FitNeuronToEye


ax_inds_to_names = {0: "learn_100",
                    1: "learn_100_hat",
                    2: "learn_500",
                    3: "learn_500_hat",
                    4: "learn_500-100",
                    5: "learn_500-100_hat",
                    }
t_title_pad = 0

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


def ax_yxline(ax, color='k', zorder=-1):
    """ Plot diagon line y=x on the axes spanning current x and y limits. """
    # Get the current axes limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # Calculate the diagonal line coordinates
    line_range = [max(xlim[0], ylim[0]), min(xlim[1], ylim[1])]
    # Draw the diagonal line
    ax.plot(line_range, line_range, color=color, zorder=zorder) 


def gather_data(data_dict, data_key):
    """ Reads data from dictionary into numpy arrays for easy plotting. """
    data = np.zeros(len(data_dict))
    for d_ind, d_key in enumerate(data_dict.keys()):
        try:
            data[d_ind] = data_dict[d_key][data_key]
        except KeyError:
            # No data here
            data[d_ind] = np.nan
    return data


def make_scatterplot_forSteve(fname, savename=None):
    """
    """
    if savename is None:
        savename = fname.split(".")[0] + "_steve.pdf"
    # Load data for plotting
    with open(fname, 'rb') as fp:
        neuron_fr_win_means = pickle.load(fp)
    for key in neuron_fr_win_means.keys():
        # Strip the numerical order part of tuple since not using it
        neuron_fr_win_means[key] = neuron_fr_win_means[key][0]
     # Setup figure layout
    plot_handles = {}
    plot_handles['fig'], plot_handles['learn_100'] = plt.subplots()
    # plot_handles['fig'].suptitle(f"Fixation rate ADJUSTED firing rate after 100 instruction trials in \n learning window [200-300] ms from target onset as a function of tuning", fontsize=12, y=.95)

    # THIS IS THE X AXIS FOR EVERY PLOT!
    baseline_tune_l = gather_data(neuron_fr_win_means, "learning_tune")
    base_dot_color = [.2 for _ in range(0, 3)]
    alt_diff_color = [.85, .1, .3]
    """ START LEARNING VS TUNING PLOT"""
    ax_name = "learn_100"
    plot_handles[ax_name].set_title(f"Fixation rate ADJUSTED firing rate after 100 instruction trials in \n learning window [200-300] ms from target onset as a function of tuning", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("100 trial adjusted firing rate (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    baseline_tune_p = gather_data(neuron_fr_win_means, "pursuit_tune")
    learning_l = gather_data(neuron_fr_win_means, "Learning_learn")
    curr_plot = plot_handles[ax_name].scatter(baseline_tune_l, learning_l - baseline_tune_p, 
                                              color=base_dot_color, s=25, zorder=3)
    curr_plot.set_label("Observed rate")
    set_equal_axlims(plot_handles[ax_name])
    plot_handles[ax_name].set_aspect('equal')
    # ax_yxline(plot_handles[ax_name], color='k', zorder=-1)
    """ ********************************************************************* """


    plot_handles['fig'].savefig(savename)
    plt.show()
    return plot_handles


def make_scatterplots(fname, savename=None):
    """
    """
    if savename is None:
        savename = fname.split(".")[0] + ".pdf"
    # Load data for plotting
    with open(fname, 'rb') as fp:
        neuron_fr_win_means = pickle.load(fp)
    for key in neuron_fr_win_means.keys():
        # Strip the numerical order part of tuple since not using it
        neuron_fr_win_means[key] = neuron_fr_win_means[key][0]
     # Setup figure layout
    plot_handles = setup_axes()
    plot_handles['fig'].suptitle(f"Fixation rate ADJUSTED firing rate across learning trials as a function of \n tuning in learning window [200-300] ms from target onset", fontsize=12, y=.95)

    # THIS IS THE X AXIS FOR EVERY PLOT!
    baseline_tune_l = gather_data(neuron_fr_win_means, "learning_tune")
    base_dot_color = [.2 for _ in range(0, 3)]
    alt_diff_color = [.85, .1, .3]
    base_dot_size = 15
    """ START LEARNING VS TUNING PLOT"""
    ax_name = "learn_100"
    plot_handles[ax_name].set_title(f"100 Trial FIX ADJUSTED firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("100 trial adjusted rate (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    baseline_tune_p = gather_data(neuron_fr_win_means, "pursuit_tune")
    baseline_tune_p = 0.
    learning_l = gather_data(neuron_fr_win_means, "Learning_learn")
    curr_plot = plot_handles[ax_name].scatter(baseline_tune_l, learning_l - baseline_tune_p, 
                                              color=base_dot_color, s=base_dot_size, zorder=3)
    curr_plot.set_label("Observed rate")
    learning_l_hat = gather_data(neuron_fr_win_means, "Learning_learn_hat")
    curr_plot = plot_handles[ax_name].scatter(baseline_tune_l, learning_l_hat, 
                                              color=alt_diff_color, s=base_dot_size, zorder=2)
    curr_plot.set_label("Predicted rate")
    delta_legend = plot_handles[ax_name].legend(fontsize='x-small', borderpad=0.2, labelspacing=0.2, 
                                     bbox_to_anchor=(0., 1.), loc='upper left', 
                                     facecolor='white', framealpha=1.)
    delta_legend.set_zorder(20)
    # set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START LEARNING MODEL VS TUNING PLOT"""
    ax_name = "learn_100_hat"
    plot_handles[ax_name].set_title(f"100 Trial observed - predicted firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Diff 100 trial observed - predicted (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction measured response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)

    # learning_l_hat = gather_data(neuron_fr_win_means, "Learning_learn_hat")
    # curr_plot = plot_handles[ax_name].scatter(baseline_tune_l, learning_l_hat, color=base_dot_color, s=base_dot_size)
    # curr_plot.set_label("Predicted rate")
    plot_handles[ax_name].scatter(baseline_tune_l, learning_l - learning_l_hat, 
                                  color=base_dot_color, s=base_dot_size, zorder=5)
    set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START 500 TRIAL LEARNING VS TUNING PLOT"""
    ax_name = "learn_500"
    plot_handles[ax_name].set_title(f"500 trial FIX ADJUSTED firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("500 trial adjusted rate (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    baseline_tune_p = gather_data(neuron_fr_win_means, "pursuit_tune")
    baseline_tune_p = 0.
    learning_l_l = gather_data(neuron_fr_win_means, "Learning_learn_late")
    plot_handles[ax_name].scatter(baseline_tune_l, learning_l_l - baseline_tune_p, 
                                  color=base_dot_color, s=base_dot_size, zorder=3)

    learning_l_l_hat = gather_data(neuron_fr_win_means, "Learning_learn_late_hat")
    plot_handles[ax_name].scatter(baseline_tune_l, learning_l_l_hat, color=alt_diff_color, s=base_dot_size, zorder=2)
    # set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START 500 TRIAL MODEL VS TUNING PLOT"""
    ax_name = "learn_500_hat"
    plot_handles[ax_name].set_title(f"500 Trial observed - predicted firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Diff 500 trial observed - predicted (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].scatter(baseline_tune_l, learning_l_l - learning_l_l_hat, 
                                  color=base_dot_color, s=base_dot_size, zorder=5)
    set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START POSTTUNING VS TUNING PLOT"""
    ax_name = "learn_500-100"
    plot_handles[ax_name].set_title(f"100-500 trial change in FIX ADJUSTED firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Average rate difference 500-100 (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    baseline_tune_p = gather_data(neuron_fr_win_means, "pursuit_tune")
    baseline_tune_p = 0.
    fr_diff = (learning_l_l - baseline_tune_p) - (learning_l - baseline_tune_p)
    plot_handles[ax_name].scatter(baseline_tune_l, fr_diff, color=base_dot_color, s=base_dot_size)
    pred_diff = (learning_l_l_hat - learning_l_hat)
    plot_handles[ax_name].scatter(baseline_tune_l, pred_diff, color=alt_diff_color, s=base_dot_size)
    set_symmetric_axlims(plot_handles[ax_name])
    # for ind in range(0, baseline_tune_l.shape[0]):
    #     xvals = [baseline_tune_l[ind], baseline_tune_l[ind]]
    #     yvals = [post_tune_p[ind] - baseline_tune_p[ind], learning_l[ind] - baseline_tune_p[ind]]
    #     plot_handles[ax_name].plot(xvals, yvals, color='k', linestyle="-", linewidth=0.5, zorder=-1)

    #     curr_plot = plot_handles[ax_name].scatter(xvals[0], yvals[0], color='g', s=base_dot_size, zorder=-1)
    #     if ind == 0:
    #         curr_plot.set_label("100 trial change")
    #     curr_plot = plot_handles[ax_name].scatter(xvals[1], yvals[1], color='r', s=base_dot_size, zorder=-1)
    #     if ind == 0:
    #         curr_plot.set_label("680 trial change")

    # delta_legend = plot_handles[ax_name].legend(fontsize='x-small', borderpad=0.2, labelspacing=0.2, 
    #                                  bbox_to_anchor=(0., 1.), loc='upper left', 
    #                                  facecolor='white', framealpha=1.)
    # delta_legend.set_zorder(20)
    """ ********************************************************************* """

    """ START POSTTUNING VS TUNING PLOT"""
    ax_name = "learn_500-100_hat"
    plot_handles[ax_name].set_title(f"100-500 Trial observed - predicted firing rate change", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Diff 100-500 trial observed - predicted (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].scatter(baseline_tune_l, fr_diff - pred_diff, 
                                  color=base_dot_color, s=base_dot_size, zorder=5)
    set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    set_all_axis_same([plot_handles[x] for x in ["learn_100", "learn_500"]])
    set_all_axis_same([plot_handles[x] for x in ["learn_100_hat", "learn_500_hat", "learn_500-100", "learn_500-100_hat"]])
    plot_handles['fig'].savefig(savename)
    plt.show()
    return plot_handles


def make_scatterplots_fix(fname, savename=None):
    """
    """
    if savename is None:
        savename = fname.split(".")[0] + "_fix.pdf"
    # Load data for plotting
    with open(fname, 'rb') as fp:
        neuron_fr_win_means = pickle.load(fp)
    for key in neuron_fr_win_means.keys():
        # Strip the numerical order part of tuple since not using it
        neuron_fr_win_means[key] = neuron_fr_win_means[key][0]
     # Setup figure layout
    plot_handles = setup_axes()
    plot_handles['fig'].suptitle(f"FIXATION rate across learning trials as a function of \n ftuning in fixation window [-300-0] ms from target oneset",
                                  fontsize=12, y=.95)

    # THIS IS THE X AXIS FOR EVERY PLOT!
    baseline_tune_l = gather_data(neuron_fr_win_means, "learning_tune")
    base_dot_color = [.3, .3, .3]
    alt_diff_color = [.85, .1, .3]
    base_dot_size = 15
    """ START LEARNING VS TUNING PLOT"""
    ax_name = "learn_100"
    plot_handles[ax_name].set_title(f"100 Trial FIXATION firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("100 trial fixation rate (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    learning_fix = gather_data(neuron_fr_win_means, "Learning_fix")
    curr_plot = plot_handles[ax_name].scatter(baseline_tune_l, learning_fix, 
                                              color=base_dot_color, s=base_dot_size, zorder=3)
    curr_plot.set_label("Observed rate")
    baseline_fix = gather_data(neuron_fr_win_means, "base_fix")
    curr_plot = plot_handles[ax_name].scatter(baseline_tune_l, baseline_fix, 
                                              color=alt_diff_color, s=base_dot_size, zorder=2)
    curr_plot.set_label("Predicted rate")
    delta_legend = plot_handles[ax_name].legend(fontsize='x-small', borderpad=0.2, labelspacing=0.2, 
                                     bbox_to_anchor=(0., 1.), loc='upper left', 
                                     facecolor='white', framealpha=1.)
    delta_legend.set_zorder(20)
    # set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START LEARNING MODEL VS TUNING PLOT"""
    ax_name = "learn_100_hat"
    plot_handles[ax_name].set_title(f"100 Trial observed - predicted fixation rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Diff 100 trial observed - predicted (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction measured response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].scatter(baseline_tune_l, learning_fix - baseline_fix, 
                                  color=base_dot_color, s=base_dot_size, zorder=5)
    set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START 500 TRIAL LEARNING VS TUNING PLOT"""
    ax_name = "learn_500"
    plot_handles[ax_name].set_title(f"500 trial FIXATION firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("500 trial fixation rate (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    learning_fix_l = gather_data(neuron_fr_win_means, "Learning_fix_late")
    plot_handles[ax_name].scatter(baseline_tune_l, learning_fix_l, 
                                  color=base_dot_color, s=base_dot_size, zorder=2)
    baseline_fix = gather_data(neuron_fr_win_means, "base_fix")
    plot_handles[ax_name].scatter(baseline_tune_l, baseline_fix, 
                                  color=alt_diff_color, s=base_dot_size, zorder=2)
    # set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START 500 TRIAL MODEL VS TUNING PLOT"""
    ax_name = "learn_500_hat"
    plot_handles[ax_name].set_title(f"500 Trial observed - predicted FIXATION rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Diff 500 trial observed - predicted (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].scatter(baseline_tune_l, learning_fix_l - baseline_fix, 
                                  color=base_dot_color, s=base_dot_size, zorder=5)
    set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START POSTTUNING VS TUNING PLOT"""
    ax_name = "learn_500-100"
    plot_handles[ax_name].set_title(f"100-500 trial change in FIXATION firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Average fixation difference 500-100 (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)

    fr_diff = (learning_fix_l - learning_fix)
    plot_handles[ax_name].scatter(baseline_tune_l, fr_diff, color=base_dot_color, s=base_dot_size)
    pred_diff = (baseline_fix - baseline_fix)
    plot_handles[ax_name].scatter(baseline_tune_l, pred_diff, color=alt_diff_color, s=base_dot_size)
    set_symmetric_axlims(plot_handles[ax_name])
    # for ind in range(0, baseline_tune_l.shape[0]):
    #     xvals = [baseline_tune_l[ind], baseline_tune_l[ind]]
    #     yvals = [post_tune_p[ind] - baseline_tune_p[ind], learning_l[ind] - baseline_tune_p[ind]]
    #     plot_handles[ax_name].plot(xvals, yvals, color='k', linestyle="-", linewidth=0.5, zorder=-1)

    #     curr_plot = plot_handles[ax_name].scatter(xvals[0], yvals[0], color='g', s=base_dot_size, zorder=-1)
    #     if ind == 0:
    #         curr_plot.set_label("100 trial change")
    #     curr_plot = plot_handles[ax_name].scatter(xvals[1], yvals[1], color='r', s=base_dot_size, zorder=-1)
    #     if ind == 0:
    #         curr_plot.set_label("680 trial change")

    # delta_legend = plot_handles[ax_name].legend(fontsize='x-small', borderpad=0.2, labelspacing=0.2, 
    #                                  bbox_to_anchor=(0., 1.), loc='upper left', 
    #                                  facecolor='white', framealpha=1.)
    # delta_legend.set_zorder(20)
    """ ********************************************************************* """

    """ START POSTTUNING VS TUNING PLOT"""
    ax_name = "learn_500-100_hat"
    plot_handles[ax_name].set_title(f"100-500 Trial observed - predicted firing rate change", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Diff 100-500 trial observed - predicted (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].scatter(baseline_tune_l, fr_diff - pred_diff, 
                                  color=base_dot_color, s=base_dot_size, zorder=5)
    set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    set_all_axis_same([plot_handles[x] for x in ["learn_100", "learn_500"]])
    set_all_axis_same([plot_handles[x] for x in ["learn_100_hat", "learn_500_hat", "learn_500-100", "learn_500-100_hat"]])
    plot_handles['fig'].savefig(savename)
    plt.show()
    return plot_handles


def make_scatterplots_raw(fname, savename=None):
    """
    """
    if savename is None:
        savename = fname.split(".")[0] + "_raw.pdf"
    # Load data for plotting
    with open(fname, 'rb') as fp:
        neuron_fr_win_means = pickle.load(fp)
    for key in neuron_fr_win_means.keys():
        # Strip the numerical order part of tuple since not using it
        neuron_fr_win_means[key] = neuron_fr_win_means[key][0]
     # Setup figure layout
    plot_handles = setup_axes()
    plot_handles['fig'].suptitle(f"RAW firing rate across learning trials as a function of \n tuning in learning window [200-300] ms from target onset", fontsize=12, y=.95)

    # THIS IS THE X AXIS FOR EVERY PLOT!
    baseline_tune_l = gather_data(neuron_fr_win_means, "learning_tune")
    base_dot_color = [.3, .3, .3]
    alt_diff_color = [.85, .1, .3]
    base_dot_size = 15
    """ START LEARNING VS TUNING PLOT"""
    ax_name = "learn_100"
    plot_handles[ax_name].set_title(f"100 Trial RAW firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("100 trial raw rate (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    learning_l_raw = gather_data(neuron_fr_win_means, "Learning_learn_raw")
    curr_plot = plot_handles[ax_name].scatter(baseline_tune_l, learning_l_raw, 
                                              color=base_dot_color, s=base_dot_size, zorder=3)
    curr_plot.set_label("Observed rate")
    base_fix_offset = gather_data(neuron_fr_win_means, "base_fix")
    learning_l_hat_raw = gather_data(neuron_fr_win_means, "Learning_learn_hat")
    learning_l_hat_raw += base_fix_offset
    curr_plot = plot_handles[ax_name].scatter(baseline_tune_l, learning_l_hat_raw, 
                                              color=alt_diff_color, s=base_dot_size, zorder=2)
    curr_plot.set_label("Predicted rate")
    delta_legend = plot_handles[ax_name].legend(fontsize='x-small', borderpad=0.2, labelspacing=0.2, 
                                     bbox_to_anchor=(0., 1.), loc='upper left', 
                                     facecolor='white', framealpha=1.)
    delta_legend.set_zorder(20)
    # set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START LEARNING MODEL VS TUNING PLOT"""
    ax_name = "learn_100_hat"
    plot_handles[ax_name].set_title(f"100 Trial observed - predicted firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Diff 100 trial observed - predicted (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction measured response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    curr_plot = plot_handles[ax_name].scatter(baseline_tune_l, learning_l_raw - learning_l_hat_raw, 
                                  color=base_dot_color, s=base_dot_size, zorder=5)
    set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START 500 TRIAL LEARNING VS TUNING PLOT"""
    ax_name = "learn_500"
    plot_handles[ax_name].set_title(f"500 trial RAW firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("500 trial raw rate (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    learning_l_l_raw = gather_data(neuron_fr_win_means, "Learning_learn_late_raw")
    plot_handles[ax_name].scatter(baseline_tune_l, learning_l_l_raw, 
                                  color=base_dot_color, s=base_dot_size, zorder=3)

    base_fix_offset = gather_data(neuron_fr_win_means, "base_fix")
    learning_l_l_hat_raw = gather_data(neuron_fr_win_means, "Learning_learn_late_hat")
    learning_l_l_hat_raw += base_fix_offset
    plot_handles[ax_name].scatter(baseline_tune_l, learning_l_l_hat_raw, 
                                  color=alt_diff_color, s=base_dot_size, zorder=2)
    # set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START 500 TRIAL MODEL VS TUNING PLOT"""
    ax_name = "learn_500_hat"
    plot_handles[ax_name].set_title(f"500 Trial observed - predicted RAW firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Diff 500 trial observed - predicted (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].scatter(baseline_tune_l, learning_l_l_raw - learning_l_l_hat_raw, 
                                  color=base_dot_color, s=base_dot_size, zorder=5)
    set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    """ START POSTTUNING VS TUNING PLOT"""
    ax_name = "learn_500-100"
    plot_handles[ax_name].set_title(f"100-500 trial change in RAW firing rate", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Average rate difference 500-100 (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)

    fr_diff = (learning_l_l_raw - learning_l_raw)
    plot_handles[ax_name].scatter(baseline_tune_l, fr_diff, color=base_dot_color, s=base_dot_size)
    pred_diff = (learning_l_l_hat_raw - learning_l_hat_raw)
    plot_handles[ax_name].scatter(baseline_tune_l, pred_diff, color=alt_diff_color, s=base_dot_size)
    set_symmetric_axlims(plot_handles[ax_name])
    # for ind in range(0, baseline_tune_l.shape[0]):
    #     xvals = [baseline_tune_l[ind], baseline_tune_l[ind]]
    #     yvals = [post_tune_p[ind] - baseline_tune_p[ind], learning_l[ind] - baseline_tune_p[ind]]
    #     plot_handles[ax_name].plot(xvals, yvals, color='k', linestyle="-", linewidth=0.5, zorder=-1)

    #     curr_plot = plot_handles[ax_name].scatter(xvals[0], yvals[0], color='g', s=base_dot_size, zorder=-1)
    #     if ind == 0:
    #         curr_plot.set_label("100 trial change")
    #     curr_plot = plot_handles[ax_name].scatter(xvals[1], yvals[1], color='r', s=base_dot_size, zorder=-1)
    #     if ind == 0:
    #         curr_plot.set_label("680 trial change")

    # delta_legend = plot_handles[ax_name].legend(fontsize='x-small', borderpad=0.2, labelspacing=0.2, 
    #                                  bbox_to_anchor=(0., 1.), loc='upper left', 
    #                                  facecolor='white', framealpha=1.)
    # delta_legend.set_zorder(20)
    """ ********************************************************************* """

    """ START POSTTUNING VS TUNING PLOT"""
    ax_name = "learn_500-100_hat"
    plot_handles[ax_name].set_title(f"100-500 Trial observed - predicted firing rate change", 
                                            fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles[ax_name].set_ylabel("Diff 100-500 trial observed - predicted (Hz)", fontsize=8)
    plot_handles[ax_name].set_xlabel("Tuning block learning direction response (Hz)", fontsize=8)
    plot_handles[ax_name].axhline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].axvline(0., color='k', linestyle="-", linewidth=1., zorder=-1)
    plot_handles[ax_name].scatter(baseline_tune_l, fr_diff - pred_diff, 
                                  color=base_dot_color, s=base_dot_size, zorder=5)
    set_symmetric_axlims(plot_handles[ax_name])
    """ ********************************************************************* """

    set_all_axis_same([plot_handles[x] for x in ["learn_100", "learn_500"]])
    set_all_axis_same([plot_handles[x] for x in ["learn_100_hat", "learn_500_hat", "learn_500-100", "learn_500-100_hat"]])
    plot_handles['fig'].savefig(savename)
    plt.show()
    return plot_handles


def get_neuron_scatter_data(neuron, fix_win, fr_learn_win, sigma=12.5, cutoff_sigma=4):
    """
    """
    # Some currently hard coded variables
    tune_trace_win = [-200, 400] #[-300, 1000]
    use_smooth_fix = True
    use_baseline_block = "StabTunePre"
    use_post_learn_block = "StabTunePost"
    n_t_instructed_win = [80, 100] # Need to hit at least this many learning trials to get counted
    n_t_instructed_win_late = [500, 520] # Need to hit at least this many learning trials to get counted
    n_t_baseline_min = 15 # Need at least 15 baseline trials to get counted
    early_probe_win = [50, 150]
    late_probe_win = [500, np.inf]

    # adjusted the n_instructed win to absolute trial number
    raw_instructed_inds = []
    raw_instructed_inds_late = []
    early_probe_inds = []
    late_probe_inds = []
    if neuron.session.blocks['Learning'] is None:
        # No learning data so nothing to do here
        return {}
    for t_ind in range(neuron.session.blocks['Learning'][0], neuron.session.blocks['Learning'][1]):
        if n_t_instructed_win[0] <= neuron.session.n_instructed[t_ind] < n_t_instructed_win[1]:
            raw_instructed_inds.append(t_ind)
        if n_t_instructed_win_late[0] <= neuron.session.n_instructed[t_ind] < n_t_instructed_win_late[1]:
            raw_instructed_inds_late.append(t_ind)
        if early_probe_win[0] <= neuron.session.n_instructed[t_ind] < early_probe_win[1]:
            early_probe_inds.append(t_ind)
        if late_probe_win[0] <= neuron.session.n_instructed[t_ind] < late_probe_win[1]:
            late_probe_inds.append(t_ind)
    if neuron.session.blocks['Washout'] is not None:
        for rel_ind, t_ind in enumerate(range(neuron.session.blocks['Washout'][0], neuron.session.blocks['Washout'][1])):
            if n_t_instructed_win[0] <= rel_ind < n_t_instructed_win[1]:
                raw_instructed_inds.append(t_ind)
    raw_instructed_inds = np.array(raw_instructed_inds)
    raw_instructed_inds_late = np.array(raw_instructed_inds_late)
    early_probe_inds = np.array(early_probe_inds)
    late_probe_inds = np.array(late_probe_inds)

    # Get linear model fit
    fit_eye_model = FitNeuronToEye(neuron, tune_trace_win, use_baseline_block, trial_sets=None,
                                    lag_range_eye=[-75, 150])
    fit_eye_model.fit_pcwise_lin_eye_kinematics(bin_width=10, bin_threshold=5,
                                                fit_constant=False, fit_avg_data=False,
                                                quick_lag_step=10, fit_fix_adj_fr=True)
    
    # Get learning window tuning block rates
    fr_win_means = {}
    for tune_trial in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
        fr = neuron.get_firing_traces_fix_adj(fr_learn_win, use_baseline_block, tune_trial, 
                                              fix_time_window=fix_win, sigma=sigma, 
                                              cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                              rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                              return_inds=False)
        fr_raw = neuron.get_firing_traces(fr_learn_win, use_baseline_block, tune_trial)
        if len(fr) == 0:
            print(f"No tuning trials found for {tune_trial} in blocks {use_baseline_block}", flush=True)
            return fr_win_means
        elif fr.shape[0] < n_t_baseline_min:
            print(f"Not enough tuning trials found for {tune_trial} in blocks {use_baseline_block}", flush=True)
            return fr_win_means
        else:
            # Store fixation adjusted and raw separately
            fr_win_means[tune_trial + "_tune"] = np.nanmean(np.nanmean(fr, axis=1), axis=0)
            fr_win_means[tune_trial + "_tune_raw"] = np.nanmean(np.nanmean(fr_raw, axis=1), axis=0)
            # Get linear model value
            X_eye = fit_eye_model.get_pcwise_lin_eye_kin_predict_data(use_baseline_block, 
                                                                      tune_trial, time_window=fr_learn_win)
            fr_win_means[tune_trial + "_tune_hat"] = np.nanmean(fit_eye_model.predict_pcwise_lin_eye_kinematics(X_eye))
    fr_fix = neuron.get_firing_traces(fix_win, use_baseline_block, None)
    fr_win_means["base_fix"] = np.nanmean(np.nanmean(fr_fix, axis=1), axis=0)
    
    # Now get the "learning" responses
    for block in ["Learning", "Washout"]:
        for trial_types in [["instruction", "pursuit"]]:
            for t_range in ["early", "late"]:
                if ( (t_range == "late") and (block == "Washout") ):
                    # No data exists for this
                    continue
                use_inst_inds = raw_instructed_inds if t_range == "early" else raw_instructed_inds_late
                # keep backward compatible with nothing for early trials
                tr_string = "" if t_range == "early" else "_late"
                t_set_inds = neuron.session.union_trial_sets_to_indices(trial_types)
                select_inds = neuron.session._parse_blocks_trial_sets([block], [t_set_inds, use_inst_inds])
                fr = neuron.get_firing_traces_fix_adj(fr_learn_win, block, select_inds, 
                                                    fix_time_window=fix_win, sigma=sigma, 
                                                    cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                    rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                                    return_inds=False)
                fr_fix = neuron.get_firing_traces(fix_win, block, select_inds)
                fr_raw = neuron.get_firing_traces(fr_learn_win, block, select_inds)
                if len(fr) == 0:
                    print(f"No trials found for {trial_types} in blocks {block} for trial range {t_range}", flush=True)
                else:
                    fr_win_means[block + "_learn" + tr_string] = np.nanmean(np.nanmean(fr, axis=1), axis=0)
                    fr_win_means[block + "_learn" + tr_string + "_raw"] = np.nanmean(np.nanmean(fr_raw, axis=1), axis=0)
                    fr_win_means[block + "_fix" + tr_string] = np.nanmean(np.nanmean(fr_fix, axis=1), axis=0)
                    # Get linear model value
                    X_eye = fit_eye_model.get_pcwise_lin_eye_kin_predict_data(block, select_inds, fr_learn_win)
                    fr_win_means[block + "_learn" + tr_string + "_hat"] = np.nanmean(fit_eye_model.predict_pcwise_lin_eye_kinematics(X_eye))

    # Get learning block probes
    for probe in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
        # Separate for early and late learning trial ranges
        for t_range in ["early", "late"]:
            use_p_inds = early_probe_inds if t_range == "early" else late_probe_inds
            select_inds = neuron.session._parse_blocks_trial_sets(["Learning"], [probe, use_p_inds])
            fr = neuron.get_firing_traces_fix_adj(fr_learn_win, "Learning", select_inds, 
                                                fix_time_window=fix_win, sigma=sigma, 
                                                cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                                return_inds=False)
            fr_raw = neuron.get_firing_traces(fr_learn_win, "Learning", select_inds)
            if len(fr) == 0:
                print(f"No probe trials found for {probe} in Learning block", flush=True)
            else:
                fr_win_means["probe_" + t_range + "_" + probe] = np.nanmean(np.nanmean(fr, axis=1), axis=0)
                fr_win_means["probe_" + t_range + "_" + probe + "_raw"] = np.nanmean(np.nanmean(fr_raw, axis=1), axis=0)
                # Get linear model value
                X_eye = fit_eye_model.get_pcwise_lin_eye_kin_predict_data("Learning", select_inds, fr_learn_win)
                fr_win_means["probe_" + t_range + "_" + probe + "_hat"] = np.nanmean(fit_eye_model.predict_pcwise_lin_eye_kinematics(X_eye))
        
    for tune_trial in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
        fr = neuron.get_firing_traces_fix_adj(fr_learn_win, use_post_learn_block, tune_trial, 
                                              fix_time_window=fix_win, sigma=sigma, 
                                              cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                              rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                              return_inds=False)
        fr_raw = neuron.get_firing_traces(fr_learn_win, use_post_learn_block, tune_trial)
        if len(fr) == 0:
            print(f"No tuning trials found for {tune_trial} in blocks {use_post_learn_block}", flush=True)
            return fr_win_means
        elif fr.shape[0] < n_t_baseline_min:
            print(f"Not enough tuning trials found for {tune_trial} in blocks {use_post_learn_block}", flush=True)
            return fr_win_means
        else:
            fr_win_means[tune_trial + "_post_tune"] = np.nanmean(np.nanmean(fr, axis=1), axis=0)
            fr_win_means[tune_trial + "_post_tune_raw"] = np.nanmean(np.nanmean(fr_raw, axis=1), axis=0)
            # Get linear model value
            X_eye = fit_eye_model.get_pcwise_lin_eye_kin_predict_data(use_post_learn_block, 
                                                                      tune_trial, time_window=fr_learn_win)
            fr_win_means[tune_trial + "_post_hat"] = np.nanmean(fit_eye_model.predict_pcwise_lin_eye_kinematics(X_eye))
    fr_fix = neuron.get_firing_traces(fix_win, use_post_learn_block, None)
    fr_win_means["base_fix_post_tune"] = np.nanmean(np.nanmean(fr_fix, axis=1), axis=0)

    return fr_win_means

