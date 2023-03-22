import os
from ReadMaestro.utils.PL2_maestro import NoEventsError
from LearnDirTunePurk.build_session import create_behavior_session, add_neuron_trials, format_ldp_trials_blocks



def gather_neurons(neurons_dir, PL2_dir, maestro_dir, maestro_save_dir,
                    cell_type=None):
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
        fnum = int(fname[-2:])
        save_name = maestro_save_dir + fname + "_maestro"
        fname_PL2 = fname + ".pl2"
        fname_neurons = "neurons_" + fname + "_viz.pkl"
        neurons_file = neurons_dir + fname_neurons

        try:
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
            ldp_sess = format_ldp_trials_blocks(ldp_sess, verbose=False)
            ldp_sess.join_neurons()
            ldp_sess.set_baseline_averages([-100, 800], rotate=True)

            for n_name in ldp_sess.get_neuron_names():
                if cell_type is None:
                    neurons_out.append(ldp_sess.neuron_info[n_name])
                elif cell_type.lower() == ldp_sess.neuron_info[n_name].cell_type.lower():
                    neurons_out.append(ldp_sess.neuron_info[n_name])
                else:
                    # This neuron should not get added to final output
                    pass
        except:
            print("SKIPPING FILE {0} for some error!".format(fname))
            continue

    return neurons_out
