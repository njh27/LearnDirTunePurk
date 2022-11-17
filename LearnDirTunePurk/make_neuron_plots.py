import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as c_look
import LearnDirTunePurk.analyze_behavior as ab
import LearnDirTunePurk.analyze_neurons as an



def baseline_tuning_2D(ldp_sess, base_block, base_data, colors=None):
    """ Plots the 4 direction 2D tuning for the block and data sepcified from
    the baseline sets stored in ldp_sess. """

    colors = parse_colors(ldp_sess.four_dir_trial_sets, colors)
    fig = plt.figure()
    ax = plt.axes()
    for curr_set in ldp_sess.four_dir_trial_sets:
        ax.scatter(ldp_sess.baseline_tuning[base_block][base_data][curr_set][0, :],
                   ldp_sess.baseline_tuning[base_block][base_data][curr_set][1, :],
                   color=colors[curr_set])
    if "position" in base_data:
        units = " (deg)"
    elif "velocity" in base_data:
        units = " (deg/s)"
    else:
        print("Cannot find postion/velocity units for axis label with data", base_data)
    ax.set_ylabel("Learning axis " + base_data + units)
    ax.set_xlabel("Pursuit axis " + base_data + units)
    ax.set_title("Four direction baseline tuning trials")
    ax.axvline(0, color='k')
    ax.axhline(0, color='k')

    return ax
