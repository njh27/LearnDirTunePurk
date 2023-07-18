import argparse
import pickle
from LearnDirTunePurk.load_directories import fun_all_neurons
from LearnDirTunePurk.learning_eye_PC_traces import get_neuron_trace_data



# Hard code some windows here
trace_win = [-100, 800]
sac_ind_cushion = 50


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

    # Setup intputs for fun_all_neurons and tuning
    cell_types = ["PC", "putPC"]
    n_tune_args = (trace_win, )
    n_tune_kwargs = {'sigma': 12.5, 
                     'cutoff_sigma': 4}
    neuron_fr_win_means = fun_all_neurons(args.neurons_dir, args.PL2_dir, args.maestro_dir, 
                                            args.maestro_save_dir, cell_types, 
                                            get_neuron_trace_data, 
                                            sess_fun,
                                            n_tune_args, 
                                            n_tune_kwargs, sac_ind_cushion=sac_ind_cushion)
    # Remove any empty
    for fname in neuron_fr_win_means.keys():
        if len(neuron_fr_win_means[fname]) == 0:
            del neuron_fr_win_means[fname]
    # Save all the data
    with open(args.save_fname, 'wb') as fp:
        pickle.dump(neuron_fr_win_means, fp)
    print(f"Output data means saved to {args.save_fname}", flush=True)