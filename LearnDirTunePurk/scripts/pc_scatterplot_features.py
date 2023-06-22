import argparse
import pickle
from LearnDirTunePurk.load_directories import fun_all_neurons
from LearnDirTunePurk.learning_scatterplots import get_neuron_scatter_data


# Hard code some windows here
fix_win = [-300, 0]
learn_win = [200, 300]

ax_inds_to_names = {0: "Learn_ax",
                    1: "Linear_model",
                    2: "Fixation",
                    3: "Washout",
                    }

def setup_axes():
    plot_handles = {}
    plot_handles['fig'], ax_handles = plt.subplots(2, 2, figsize=(8, 11))
    ax_handles = ax_handles.ravel()
    for ax_ind in range(0, ax_handles.size):
        plot_handles[ax_inds_to_names[ax_ind]] = ax_handles[ax_ind]

    return plot_handles


def sess_fun(ldp_sess):
    """ Defines a function used to process each ldp_session object within the call
    to "fun_all_neurons". """
    # Add the Gaussian smoothed firing rates to each neuron
    ldp_sess.gauss_convolved_FR(10, cutoff_sigma=4, series_name="_gauss")

    return ldp_sess


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_fname")
    # Setup default directories
    parser.add_argument("--neurons_dir", default="/home/nate/neuron_viz_final/")
    parser.add_argument("--PL2_dir", default="/mnt/isilon/home/nathan/Data/LearnDirTunePurk/PL2FilesRaw/")
    parser.add_argument("--maestro_dir", default="/mnt/isilon/home/nathan/Data/LearnDirTunePurk/MaestroFiles/")
    parser.add_argument("--maestro_save_dir", default="/home/nate/Documents/MaestroPickles/")
    args = parser.parse_args()

    # # Setup figure layout
    # plot_handles = setup_axes()
    # plot_handles['fig'].suptitle(f"Firing rate changes as a function of tuning", fontsize=12, y=.95)

    # Setup intputs for fun_all_neurons and tuning
    cell_types = ["PC", "putPC"]
    n_tune_args = (fix_win, learn_win)
    n_tune_kwargs = {'sigma': 12.5, 
                     'cutoff_sigma': 4, 
                     'show_fig': False}
    neuron_fr_win_means = fun_all_neurons(args.neurons_dir, args.PL2_dir, args.maestro_dir, 
                                            args.maestro_save_dir, cell_types, 
                                            get_neuron_scatter_data, 
                                            sess_fun,
                                            n_tune_args, 
                                            n_tune_kwargs,
                                            n_break=2)

    # Save all the data
    with open(args.save_fname, 'wb') as fp:
        pickle.dump(neuron_fr_win_means, fp)
    print(f"Output data means saved to {args.save_fname}", flush=True)