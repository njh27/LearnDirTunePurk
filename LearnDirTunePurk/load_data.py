import pickle
import re
import os
from ReadMaestro.maestro_read import load_directory
from ReadMaestro.format_trials import combine_targets
from ReadMaestro.target import compress_target_data
# from LearnDirTunePurk.cluster_old_plx import load_mat_maestro



def load_maestro_directory(fname, maestro_dir, check_existing_maestro=True,
                           save_data=True, save_name=None, combine_targs=True,
                           compress_data=True):
    """ Loads maestro data but can also combine targets and do target data
    compression as is generally required by LearnDirTunePurk files and save
    these conversions in the maestro pickle file. """
    fname = fname.split(maestro_dir)[-1]
    maestro_dir = maestro_dir.split(fname)[0]
    if maestro_dir[-1] != "/":
        maestro_dir = maestro_dir + "/"
    if save_name is not None:
        if (save_name[-7:] != ".pickle") and (save_name[-4:] != ".pkl"):
            save_name = save_name + ".pickle"
    # Set save always False here. If we are saving, we will save at the end with
    # the compressed data and combined targets
    maestro_data, l_exists = load_directory(os.path.join(maestro_dir, fname),
                                        check_existing=check_existing_maestro,
                                        save_data=False,
                                        save_name=save_name,
                                        return_loaded_existing=True)
    if l_exists:
        # assumes that combine targets and compress data are already done!
        print("Loaded maestro data from {0}.".format(save_name))
        return maestro_data
    if combine_targs:
        # Combining these two targets is hard coded for LearnDirTunePurk
        print("Combining targets 'rmfixation1'and 'rmpursuit1'.")
        combine_targets(maestro_data, 'rmfixation1', 'rmpursuit1')
    if compress_data:
        print("Compressing target data for each trial.")
        compress_target_data(maestro_data)

    if save_data:
        if save_name is None:
            save_name = maestro_dir + fname + "_maestro.pickle"
        print("Saving Maestro trial data as:", save_name)
        with open(save_name, 'wb') as fp:
            pickle.dump(maestro_data, fp, protocol=-1)

    return maestro_data


def load_maestro_directory_old_yan(fname, maestro_dir, check_existing_maestro=True,
                           save_data=True, save_name=None, combine_targs=True,
                           compress_data=True):
    """ Loads maestro data but can also combine targets and do target data
    compression as is generally required by LearnDirTunePurk files and save
    these conversions in the maestro pickle file. """
    fname = fname.split(maestro_dir)[-1]
    maestro_dir = maestro_dir.split(fname)[0]
    if maestro_dir[-1] != "/":
        maestro_dir = maestro_dir + "/"
    if save_name is not None:
        if (save_name[-7:] != ".pickle") and (save_name[-4:] != ".pkl"):
            save_name = save_name + ".pickle"
    # Set save always False here. If we are saving, we will save at the end with
    # the compressed data and combined targets
    maestro_data, l_exists = load_mat_maestro(os.path.join(maestro_dir, fname), 
                                              check_existing=check_existing_maestro, 
                                              save_data=False, save_name=save_name, 
                                              return_loaded_existing=True)
    if l_exists:
        # assumes that combine targets and compress data are already done!
        print("Loaded maestro data from {0}.".format(save_name))
        return maestro_data
    if combine_targs:
        # Combining these two targets is hard coded for LearnDirTunePurk
        print("Combining targets 'rmfixation1'and 'rmpursuit1'.")
        combine_targets(maestro_data, 'rmfixation1', 'rmpursuit1')
    if compress_data:
        print("Compressing target data for each trial.")
        compress_target_data(maestro_data)

    if save_data:
        if save_name is None:
            save_name = maestro_dir + fname + "_maestro.pickle"
        print("Saving Maestro trial data as:", save_name)
        with open(save_name, 'wb') as fp:
            pickle.dump(maestro_data, fp, protocol=-1)

    return maestro_data


def maestro_to_pickle_batch(maestro_dir, existing_dir=None, save_dir=None,
    combine_targs=True, compress_data=True, recompute_all=False):

    if not os.path.isdir(maestro_dir):
        raise RuntimeError('Directory name {:s} is not valid'.format(maestro_dir))
    pattern = re.compile('LearnDirTunePurk_[A-Za-z]*_[0-9][0-9]')
    filenames = [f for f in os.listdir(maestro_dir) if os.path.isdir(os.path.join(maestro_dir, f)) and pattern.search(f) is not None]

    for maestro_file in filenames:
        print("Checking file:", maestro_file)
        if existing_dir is not None:
            existing_dir = existing_dir.split(maestro_file)[0]
            if existing_dir[-1] != "/":
                existing_dir = existing_dir + "/"
            if ( (os.path.isfile(existing_dir + maestro_file + "_maestro.pickle")) and
                 (not recompute_all) ):
                # File already exists and not recomputing all, so skip
                print("Maestro pickle file:", existing_dir + filenames[-1] + "_maestro.pickle", "already exists.")
                continue

        # Made it here, so file not found. Make the file.
        if maestro_dir[-1] != "/":
            maestro_dir = maestro_dir + "/"
        print("Loading directory", maestro_dir + maestro_file)
        maestro_data = load_directory(maestro_dir + maestro_file, check_existing=False, save_data=False)

        if combine_targs:
            # Combining these two targets is hard coded for LearnDirTunePurk
            combine_targets(maestro_data, 'rmfixation1', 'rmpursuit1')

        if compress_data:
            compress_target_data(maestro_data)

        # Make default save name
        save_name = maestro_file + "_maestro.pickle"

        if save_dir is not None:
            save_dir = save_dir.split(maestro_file)[0]
            if save_dir[-1] != "/":
                save_dir = save_dir + "/"
            print("Saving Maestro trial data as:", save_dir + save_name)
            with open(save_dir + save_name, 'wb') as fp:
                pickle.dump(maestro_data, fp, protocol=-1)

    return None


def load_mat_maestro(directory_name, check_existing=True, save_data=False,
                        save_name=None, return_loaded_existing=False):
        """Load a directory of maestro files

        Loads a complete directory of maestro files as a list of dictionaries.
        The filenames are assumed to be in the of the form *.[0-9][0-9][0-9][0-9].
        This function attempts to load the files in order of their suffix.
        check_existing true will search for existing pickle files starting with the
        input save_name and then the input directory name by standard naming
        default conventions.
        """
        loaded_existing = False
        if check_existing:
            if save_name is not None:
                try:
                    if (save_name[-7:] != ".pickle") and (save_name[-4:] != ".pkl"):
                        save_name = save_name + ".pickle"
                    with open(save_name, 'rb') as fp:
                        data = pickle.load(fp)
                    loaded_existing = True
                    if return_loaded_existing:
                        return data, loaded_existing
                    else:
                        return data
                except FileNotFoundError:
                    pass
            try:
                with open(directory_name + ".pickle", 'rb') as fp:
                    data = pickle.load(fp)
                loaded_existing = True
                if return_loaded_existing:
                    return data, loaded_existing
                else:
                    return data
            except FileNotFoundError:
                pass
            try:
                with open(directory_name + "_maestro.pickle", 'rb') as fp:
                    data = pickle.load(fp)
                loaded_existing = True
                if return_loaded_existing:
                    return data, loaded_existing
                else:
                    return data
            except FileNotFoundError:
                pass
            print("Could not find existing Maestro file. Recomputing from scratch.")
        if directory_name[-4:] != ".mat":
            directory_name = directory_name + ".mat"
        data = get_maestro_from_mat(directory_name)

        if save_name is not None:
            # save_data = True
            if (save_name[-7:] != ".pickle") and (save_name[-4:] != ".pkl"):
                save_name = save_name + ".pickle"
        if save_data:
            if save_name is None:
                save_name = directory_name.split("/")[-1]
                root_name = directory_name.split("/")[0:-1]
                save_name = save_name + "_maestro.pickle"
                save_name = "".join(x + "/" for x in root_name) + save_name
            print("Saving Maestro trial data as:", save_name)
            with open(save_name, 'wb') as fp:
                pickle.dump(data, fp, protocol=-1)

        if return_loaded_existing:
            return data, loaded_existing
        else:
            return data
