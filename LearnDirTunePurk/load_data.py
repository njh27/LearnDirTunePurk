import pickle
import ReadMaestro as rm
from ReadMaestro.utils.PL2_maestro import NoEventsError
import re
import os



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
    maestro_data, l_exists = rm.maestro_read.load_directory(maestro_dir + fname,
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
        rm.format_trials.combine_targets(maestro_data, 'rmfixation1', 'rmpursuit1')
    if compress_data:
        print("Compressing target data for each trial.")
        rm.target.compress_target_data(maestro_data)

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


def gather_neurons(neurons_dir, PL2_dir, maestro_dir, maestro_save_dir,
                    cell_type=None, tune_t_win=[50, 300], tune_block='StandTunePre'):
    """ Loads data according to the name of the files input in neurons dir.
    Creates a session from the maestro data and joins the corresponding
    neurons from the neurons file. Neuron objects with the desired matching
    neuron type are output.
    """
    neurons_out = []
    for f in os.listdir(neurons_dir):
        fname = f
        fname = fname.split(".")[0]
        if fname[-4:].lower() == "_viz":
            fname = fname.split("_viz")[0]
        if fname[0:8].lower() == "neurons_":
            fname = fname.split("neurons_")[1]
        save_name = maestro_save_dir + fname + "_maestro"
        fname_PL2 = fname + ".pl2"
        fname_neurons = "neurons_" + fname + "_viz.pkl"
        neurons_file = neurons_dir + fname_neurons

        ldp_sess = create_behavior_session(fname, maestro_dir,
                                            session_name=fname, rotate=True,
                                            check_existing_maestro=True,
                                            save_maestro_data=True,
                                            save_maestro_name=save_name)

        print("Loading neurons from file {0}.".format(fname_neurons))
        try:
            ldp_sess = add_neuron_trials(ldp_sess, maestro_dir, neurons_file,
                                        PL2_dir=PL2_dir, dt_data=1,
                                        save_maestro_name=save_name,
                                        save_maestro_data=True)
        except NoEventsError:
            print("!SKIPPING! file {0} because it has no PL2 events.".format(fname_PL2))
            continue

        # Continue building session and neuron tuning
        ldp_sess = ldp.build_session.format_ldp_trials_blocks(ldp_sess, verbose=False)
        ldp_sess.join_neurons(tune_t_win, tune_block)
        ldp_sess.set_baseline_averages([-100, 800], rotate=True)

        for n_name in ldp_sess.get_neuron_names():
            if cell_type is None:
                neurons_out.append(ldp_sess.neuron_info[n_name])
            elif cell_type.lower() == ldp_sess.neuron_info[n_name].cell_type.lower():
                neurons_out.append(ldp_sess.neuron_info[n_name])
            else:
                # This neuron should not get added to final output
                pass

    return neurons_out
