import pickle
import ReadMaestro as rm



def load_maestro_directory(fname, maestro_dir, existing_dir=None, save_dir=None,
    combine_targs=True, compress_data=True, save_name=None):

    fname = fname.split(maestro_dir)[-1]
    if existing_dir is not None:
        existing_dir = existing_dir.split(fname)[0]
        if existing_dir[-1] != "/":
            existing_dir = existing_dir + "/"
        try:
            with open(existing_dir + fname + ".pickle", 'rb') as fp:
                data = pickle.load(fp)
            return data
        except FileNotFoundError:
            pass
        try:
            with open(existing_dir + fname + "_maestro.pickle", 'rb') as fp:
                data = pickle.load(fp)
            return data
        except FileNotFoundError:
            pass
        print("Could not find existing Maestro file. Recomputing from scratch.")

    # Set save always False here. If we are saving, we will save at the end with
    # the compressed data and combined targets
    maestro_dir = maestro_dir.split(fname)[0]
    if maestro_dir[-1] != "/":
        maestro_dir = maestro_dir + "/"
    maestro_data = rm.maestro_read.load_directory(maestro_dir + fname, check_existing=False, save_data=False)

    if combine_targs:
        # Combining these two targets is hard coded for LearnDirTunePurk
        rm.format_trials.combine_targets(maestro_data, 'rmfixation1', 'rmpursuit1')

    if compress_data:
        rm.target.compress_target_data(maestro_data)

    if save_name is None:
        # Make default save name
        save_name = fname + "_maestro.pickle"

    if save_dir is not None:
        save_dir = save_dir.split(fname)[0]
        if save_dir[-1] != "/":
            save_dir = save_dir + "/"
        print("Saving Maestro trial data as:", save_dir + save_name)
        with open(save_dir + save_name, 'wb') as fp:
            pickle.dump(maestro_data, fp, protocol=-1)

    return maestro_data
