import numpy as np
import matplotlib.pyplot as plt
import LearnDirTunePurk.analyze_behavior as ab



def mean_traces(sess, data_name, time_window, blocks,
                plot_ax=None, trial_sets=None, rotate=True):
    """ Makes xy plot of traces for the hardcoded directions for the data
    and info specified. """

    if trial_sets is None:
        trial_sets = sess.four_dir_trial_sets
    if plot_ax is None:
        plt.figure()
        plot_ax = plt.axes()


def scatter_2D(sess, data_name, time_window):
    pass


def binned_scatter_2D():
    pass


def single_traces():
    pass


def binned_traces():
    pass
