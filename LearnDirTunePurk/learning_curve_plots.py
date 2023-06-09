import numpy as np
import matplotlib.pyplot as plt



def setup_axes():
    plot_handles = {}
    plot_handles['fig'] = plt.figure(figsize=(12, 8))
    spec = plot_handles['fig'].add_gridspec(6, 12)
    plot_handles['learning'] = plot_handles['fig'].add_subplot(spec[0, 1])
    plot_handles['anti_pursuit'] = plot_handles['fig'].add_subplot(spec[1, 0])
    plot_handles['pursuit'] = plot_handles['fig'].add_subplot(spec[1, 2])
    plot_handles['anti_learning'] = plot_handles['fig'].add_subplot(spec[2, 1])
    plot_handles['fix_fun'] = plot_handles['fig'].add_subplot(spec[0:3, 3:])
    plot_handles['learn_fun'] = plot_handles['fig'].add_subplot(spec[3:, :])

    return plot_handles

def get_plotting_rates(neuron, blocks, trial_sets, time_window, sigma=12.5, cutoff_sigma=4):

    fr, inds = neuron.get_firing_traces(time_window, blocks, trial_sets, return_inds=True)
    fr = np.nanmean(fr, axis=1)
    smooth_fr, _ = neuron.get_smooth_fr_by_block_gauss(blocks, time_window, sigma=sigma, cutoff_sigma=cutoff_sigma)

    return fr, smooth_fr, inds

def plot_neuron_tuning_learning(neuron, blocks, trial_sets, fix_win, learn_win, sigma=12.5, cutoff_sigma=4):

    plot_handles = setup_axes()
    # Plot each block separte so there is discontinuity between blocks
    for b_name in blocks:
        # Draw lines delineating block starts/stops
        try:
            if b_name == "Washout":
                plot_handles['fix_fun'].axvline(neuron.session.blocks[b_name][0], color='b')
            else:
                plot_handles['fix_fun'].axvline(neuron.session.blocks[b_name][0], color=[.75, .2, .1])
        except TypeError:
            print(f"No blocks available for block {b_name}")
        fr, smooth_fr, inds = get_plotting_rates(neuron, b_name, trial_sets, fix_win, sigma=sigma, cutoff_sigma=cutoff_sigma)
        plot_handles['fix_fun'].scatter(inds, fr, color='k', s=5)
        plot_handles['fix_fun'].plot(inds, smooth_fr, color=[.1, .8, 0], linewidth=5)

        # Draw lines delineating block starts/stops
        try:
            if b_name == "Washout":
                plot_handles['learn_fun'].axvline(neuron.session.blocks[b_name][0], color='b')
            else:
                plot_handles['learn_fun'].axvline(neuron.session.blocks[b_name][0], color=[.75, .2, .1])
        except TypeError:
            print(f"No blocks available for block {b_name}")
        fr, smooth_fr, inds = get_plotting_rates(neuron, b_name, trial_sets, learn_win, sigma=sigma, cutoff_sigma=cutoff_sigma)
        plot_handles['learn_fun'].scatter(inds, fr, color='k', s=5)
        plot_handles['learn_fun'].plot(inds, smooth_fr, color=[.1, .8, 0], linewidth=5)

    plt.show()
    return plot_handles

        

