import pickle
import ReadMaestro as rm



def load_maestro_directory(fname, check_existing=True, save_data=True,
    combine_targs=True, compress_data=True, save_name=None):

    if check_existing:
        try:
            with open(fname + ".pickle", 'rb') as fp:
                data = pickle.load(fp)
            return data
        except FileNotFoundError:
            pass
        try:
            with open(fname + "_maestro.pickle", 'rb') as fp:
                data = pickle.load(fp)
            return data
        except FileNotFoundError:
            pass
        print("Could not find existing Maestro file. Recomputing from scratch.")

    # Set save always False here. If we are saving, we will save at the end with
    # the compressed data and combined targets
    maestro_data = rm.maestro_read.load_directory(fname, check_existing=False, save_data=False)

    if combine_targs:
        # Combining these two targets is hard coded for LearnDirTunePurk
        rm.format_trials.combine_targets(maestro_data, 'rmfixation1', 'rmpursuit1')

    if compress_data:
        rm.target.compress_target_data(maestro_data)

    if save_name is None:
        # Make default save name
        save_name = fname + "_maestro.pickle"

    if save_data:
        print("Saving Maestro trial data as:", save_name)
    with open(save_name, 'wb') as fp:
        pickle.dump(maestro_data, fp, protocol=-1)

    return maestro_data
