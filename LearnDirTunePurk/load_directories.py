import os
import numpy as np
from ReadMaestro.utils.PL2_maestro import NoEventsError
from LearnDirTunePurk.build_session import create_behavior_session, add_neuron_trials, format_ldp_trials_blocks
from NeuronAnalysis.general import Indexer



def get_eye_target_pos_and_rate(Neuron, time_window, blocks=None, trial_sets=None,
                                use_series=None):
    """
    """
    if use_series is not None:
        if not isinstance(use_series, str):
            raise ValueError("use_series must be a string that follows the neuron's name.")
        use_series = Neuron.name + "_" + use_series
        # Pull data from this series
        Neuron.set_use_series(use_series)

    epos_p, epos_l, t_inds = Neuron.session.get_xy_traces("eye position", time_window, blocks=blocks,
                             trial_sets=trial_sets, return_inds=True)
    tpos_p, tpos_l = Neuron.session.get_xy_traces("target position", time_window, blocks=blocks,
                             trial_sets=trial_sets, return_inds=False)
    # Unstable neurons can cause firing rate data to be missing so fill nans first
    fr = np.full(tpos_p.shape, np.nan)
    valid_fr, fr_tinds = Neuron.get_firing_traces(time_window, blocks=blocks,
                                    trial_sets=trial_sets, return_inds=True)
    match_t_inds = Indexer(t_inds)
    for t_num, t in enumerate(fr_tinds):
        # Match behavior trials to neurons in case any missing
        match_ind = match_t_inds.move_index_next(t, '=')
        fr[match_ind, :] = valid_fr[t_num, :]
    return np.stack((epos_p, epos_l, tpos_p, tpos_l, fr), axis=2)


def gather_neurons(neurons_dir, PL2_dir, maestro_dir, maestro_save_dir, cell_types, 
                   data_fun, sess_fun=None, data_fun_args=(), data_fun_kwargs={},
                   verbose=True):
    """ Loads data according to the name of the files input in neurons dir.
    Creates a session from the maestro data and joins the corresponding
    neurons from the neurons file. Goes through all neurons and if their name
    is found in the list 'cell_types', then 'data_fun' is called on that neuron
    and its output is appended to a list under the output dict key 'neuron_name'.
    Optionally the input function "sess_fun" can be called with a single input,
    the single ldp_session object, and return a single output, the same single
    ldp_session object, to perform any preprocessing before neurons
    from that session are gathered and data_fun is called.
    """
    rotate = True
    if not rotate:
        print("Getting WITHOUT rotating data!!")
    if not isinstance(cell_types, list):
        cell_types = [cell_types]
    out_data = {}
    n_total_units = 0
    failed_files = []
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

        try:
            ldp_sess = create_behavior_session(fname, maestro_dir,
                                                session_name=fname, rotate=rotate,
                                                check_existing_maestro=True,
                                                save_maestro_data=True,
                                                save_maestro_name=save_name)

            if verbose: print("Loading neurons from file {0}.".format(fname_neurons))
            try:
                ldp_sess = add_neuron_trials(ldp_sess, maestro_dir, neurons_file,
                                            PL2_dir=PL2_dir, dt_data=1,
                                            save_maestro_name=save_name,
                                            save_maestro_data=True)
            except NoEventsError:
                if verbose: print("!SKIPPING! file {0} because it has no PL2 events.".format(fname_PL2))
                failed_files.append((fname, "No PL2 events found.")) # Store error text
                continue

            # Continue building session and neuron tuning
            ldp_sess = format_ldp_trials_blocks(ldp_sess, verbose=False)
            ldp_sess.join_neurons()
            if sess_fun is not None:
                ldp_sess = sess_fun(ldp_sess)

            for n_name in ldp_sess.get_neuron_names():
                n_type = n_name.split("_")[0]
                if n_type in cell_types:
                    # Call data function on this neuron and save to output
                    if n_type in out_data:
                        out_data[n_type].append(data_fun(ldp_sess.neuron_info[n_name], 
                                                         *data_fun_args, 
                                                         **data_fun_kwargs))
                    else:
                        out_data[n_type] = [data_fun(ldp_sess.neuron_info[n_name], 
                                                     *data_fun_args, 
                                                     **data_fun_kwargs)]
                    if verbose: print(f"Adding neuron {n_name}")
                    n_total_units += 1
                    # if n_total_units > 1:
                    #     return out_data
        except Exception as e: # Catch any error
            print("SKIPPING FILE {0} for some error!".format(fname))
            failed_files.append((fname, str(e))) # Store error text
            continue
    if verbose: print(f"Successfully gathered data for {n_total_units} total neurons.")
    if (len(failed_files) > 0) and (verbose):
        print("The following files FAILED to load appropriately: ")
        for ff in failed_files:
            print(f"{ff[0]}...")
            print(ff[1])
    return out_data
