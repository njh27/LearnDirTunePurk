import pickle
import ReadMaestro as rm
import re
import os



def load_maestro_directory(fname, maestro_dir, existing_dir=None, save_dir=None,
    combine_targs=True, compress_data=True, save_name=None):
    """ Loads maestro data but can also combine targets and do target data
    compression as is generally required by LearnDirTunePurk files and save
    these conversions in the maestro pickle file. """
    fname = fname.split(maestro_dir)[-1]
    maestro_dir = maestro_dir.split(fname)[0]
    if maestro_dir[-1] != "/":
        maestro_dir = maestro_dir + "/"
    # Set save always False here. If we are saving, we will save at the end with
    # the compressed data and combined targets
    print("Loading Maestro directory data from {0}.".format(maestro_dir+fname))
    maestro_data, l_exists = rm.maestro_read.load_directory(maestro_dir + fname,
                                        check_existing=False, save_data=False,
                                        return_loaded_existing=True)
    if l_exists:
        # assumes that combine targets and compress data are already done!
        return maestro_data
    if combine_targs:
        # Combining these two targets is hard coded for LearnDirTunePurk
        print("Combining targets 'rmfixation1'and 'rmpursuit1'.")
        rm.format_trials.combine_targets(maestro_data, 'rmfixation1', 'rmpursuit1')
    if compress_data:
        print("Compressing target data for each trial.")
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
        maestro_data = rm.maestro_read.load_directory(maestro_dir + maestro_file, check_existing=False, save_data=False)

        if combine_targs:
            # Combining these two targets is hard coded for LearnDirTunePurk
            rm.format_trials.combine_targets(maestro_data, 'rmfixation1', 'rmpursuit1')

        if compress_data:
            rm.target.compress_target_data(maestro_data)

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
