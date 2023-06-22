import argparse
from matplotlib.backends.backend_pdf import PdfPages
from LearnDirTunePurk.load_directories import fun_all_neurons
from LearnDirTunePurk.learning_curve_plots import plot_neuron_tuning_learning



# Some hard coded blocks and windows for analysis
fit_blocks = ["FixTunePre", "RandVPTunePre", "StandTunePre", "StabTunePre", "Learning", 
              "FixTunePost", "StabTunePost", "Washout", "FixTuneWash", "StabTuneWash", 
              "StandTuneWash"]
fit_trial_sets = None
fix_t_win = [-300, 0]
learn_t_win = [200, 300]

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

    # Create a new PDF file
    if len(args.save_fname) > 4:
        if args.save_fname[-4:].lower() == ".pdf":
            sf_list = list(args.save_fname)
            sf_list[-4:] = ".pdf"
            args.save_fname = "".join(sf_list)
        else:
            args.save_fname = args.save_fname + ".pdf"
    else:
        args.save_fname = args.save_fname + ".pdf"
    print(f"Output figures will be saved to file {args.save_fname}", flush=True)
    pdf_pages = PdfPages(args.save_fname)

    # Setup intputs for fun_all_neurons and tuning
    cell_types = ["PC", "putPC"]
    n_tune_args = (fit_blocks, fit_trial_sets, fix_t_win, learn_t_win)
    n_tune_kwargs = {'sigma': 12.5, 
                     'cutoff_sigma': 4, 
                     'show_fig': False}
    neuron_figs = fun_all_neurons(args.neurons_dir, args.PL2_dir, args.maestro_dir, 
                                 args.maestro_save_dir, cell_types, 
                                 plot_neuron_tuning_learning, 
                                 sess_fun,
                                 n_tune_args, 
                                 n_tune_kwargs)

    # Add the figures to the PDF file
    for n_name in neuron_figs.keys():
        # Add a filename number so we can sort these
        if "_PC" in n_name:
            file_num = int(n_name.split("_PC")[0][-2:])
        else:
            file_num = int(n_name.split("_put")[0][-2:])
        neuron_figs[n_name] = (neuron_figs[n_name][0], neuron_figs[n_name][1], file_num)
    # Now sort by file_num so output is sensible
    sorted_plot_handles = [x[0] for x in sorted([neuron_figs[key] for key in neuron_figs.keys()], key=lambda x:x[2])]
    for plot_handles in sorted_plot_handles:
        pdf_pages.savefig(plot_handles['fig'])
    # Close the PDF file
    pdf_pages.close()
    print(f"Output figures saved to {args.save_fname}", flush=True)