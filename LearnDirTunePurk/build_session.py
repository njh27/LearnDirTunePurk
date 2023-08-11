import numpy as np
import pickle
from LearnDirTunePurk.load_data import load_maestro_directory
from LearnDirTunePurk import format_trials
from LearnDirTunePurk.ldp_session import LDPSession
from ReadMaestro.format_trials import data_to_target
from ReadMaestro.maestro_read import load_directory as rm_load_directory
from ReadMaestro.utils.PL2_maestro import is_maestro_pl2_synced, add_plexon_events
import SessionAnalysis.utils.format_trial_dicts as sa_format_dicts



def create_neuron_session(fname, neurons_dir, PL2_dir, maestro_dir,
                        save_maestro=True, maestro_save_dir=None,
                        rotate_eye_data=True, sac_ind_cushion=40,
                        nan_saccades=True):
    """ Takes the input Maestro and PL2 files and generates the default LDP
    session object using "builed_session" and joins it with the neurons
    from the PL2/neuro_viz file. Returns the LDP session object with
    neurons joined.
    flag "save_maestro" will save the Maestro file if it was updated to be
    synced with the PL2 file. """
    # Get root fname
    fname = fname.split(".")[0]
    if fname[-4:].lower() == "_viz":
        fname = fname.split("_viz")[0]
    if fname[0:8].lower() == "neurons_":
        fname = fname.split("neurons_")[1]
    if maestro_save_dir is None:
        maestro_save_dir = maestro_dir
    save_name = maestro_save_dir + fname + "_maestro"
    fname_neurons = "neurons_" + fname + "_viz.pkl"
    neurons_file = neurons_dir + fname_neurons

    ldp_sess = create_behavior_session(fname, maestro_dir,
                                        session_name=fname, rotate=rotate_eye_data,
                                        check_existing_maestro=True,
                                        save_maestro_data=save_maestro,
                                        save_maestro_name=save_name,
                                        nan_saccades=nan_saccades)

    ldp_sess = add_neuron_trials(ldp_sess, maestro_dir, neurons_file,
                                PL2_dir=PL2_dir, dt_data=1,
                                save_maestro_name=save_name,
                                save_maestro_data=save_maestro)
    ldp_sess = format_ldp_trials_blocks(ldp_sess, sac_ind_cushion=sac_ind_cushion, verbose=False)
    ldp_sess.join_neurons()

    return ldp_sess


def create_behavior_session(fname, maestro_dir, session_name=None, rotate=True,
            check_existing_maestro=True, save_maestro_data=True,
            save_maestro_name=None, verbose=True, nan_saccades=True):
    """
    """
    if session_name is None:
        session_name = fname
    # Load all Maestro data
    maestro_data = load_maestro_directory(fname, maestro_dir,
                                check_existing_maestro=check_existing_maestro,
                                save_data=save_maestro_data,
                                save_name=save_maestro_name, combine_targs=True,
                                compress_data=True)

    if ( ("Yoda" in session_name) and (int(session_name.split("_")[-1]) < 20) ):
        print("Treating this as a weird Yoda file")
        is_weird_Yoda = True
    else:
        is_weird_Yoda = False
    # Sets trial targets as objects so probably want to do this after loading and not saved
    if verbose: print("Formatting target data and trials.")
    data_to_target(maestro_data)

    # Reformat the multi targets and names of trials into one and name events
    if verbose: print("Renaming trials and events.")

    format_trials.rename_stab_probe_trials(maestro_data)
    format_trials.name_trial_events(maestro_data, is_weird_Yoda)
    is_stab_learning = True
    for trial in maestro_data:
        if ( ("-rt" in trial['header']['name']) or
             ("-up" in trial['header']['name']) or
             ("-lt" in trial['header']['name']) or
             ("-dn" in trial['header']['name']) ):
            if trial['header']['UsedStab']:
                is_stab_learning = True
            else:
                is_stab_learning = False
                print("Learning trials are NOT STABILIZED!!!")
                break

    # Create base list of ApparatusTrial trials from target0
    if verbose: print("Converting data to trial objects.")
    trial_list = sa_format_dicts.maestro_to_apparatus_trial(
                        maestro_data, 0, 1, start_data=0, data_name="target0")

    # Create a second list of BehaviorTrial trials from eye data
    trial_list_bhv = sa_format_dicts.maestro_to_behavior_trial(
                        maestro_data, 1, start_data=0, data_name="eye")

    # Create base session from apparatus trials then add behavior
    if verbose: print("Generating session and adding blocks.")
    ldp_sess = LDPSession(trial_list, session_name=session_name, rotate=rotate, nan_saccades=nan_saccades)
    ldp_sess.add_trial_data(trial_list_bhv, data_type=None)
    # Can add slip data now that we have target and behavior
    ldp_sess.add_retinal_slip_data()
    # Add fields characteristic to this session for future reference
    ldp_sess.is_weird_Yoda = is_weird_Yoda
    ldp_sess.is_stab_learning = is_stab_learning
    ldp_sess.fname = fname

    return ldp_sess


def add_neuron_trials(ldp_sess, maestro_dir, neurons_file, PL2_dir=None,
                      dt_data=1, save_maestro_name=None, save_maestro_data=True):
    """ Adds neuron trials to the LDPSession object from the neurons list
    of dictionaries in neurons. """
    maestro_data = rm_load_directory(maestro_dir+ldp_sess.fname,
                                        check_existing=True,
                                        save_data=save_maestro_data,
                                        save_name=save_maestro_name)
    if len(ldp_sess) != len(maestro_data):
        raise ValueError("The session must have the same number of trials as the input maestro_data to sync! Make sure they are the correct file and none have been removed from the Session.")
    # trial names may have been updated to create sess so check them
    for t_ind in range(0, len(maestro_data)):
        if maestro_data[t_ind]['header']['name'] in ldp_sess[t_ind].name:
            maestro_data[t_ind]['header']['name'] = ldp_sess[t_ind].name
        else:
            raise RuntimeError("Could not match trial names for trial {0} between maestro data {1} and session name {2}.".format(t_ind, maestro_data[t_ind]['header']['name'], ldp_sess[t_ind].name))

    if not is_maestro_pl2_synced(maestro_data, ldp_sess.fname + ".pl2"):
        if save_maestro_name is None:
            print("maestro_data not yet synced with PL2 file {0}. Syncing file but not saving because maestro_save_name not specified.".format(ldp_sess.fname + ".pl2"))
        else:
            print("maestro_data not yet synced with PL2 file {0}. Syncing.".format(ldp_sess.fname + ".pl2"))
        maestro_data = add_plexon_events(maestro_data,
                                                    PL2_dir + ldp_sess.fname + ".pl2",
                                                    maestro_pl2_chan_offset=3,
                                                    remove_bad_inds=False)
        # SAVED BEFORE POTENTIAL DELETING TRIALS!
        if save_maestro_name is not None:
            if isinstance(save_maestro_name, str):
                if (save_maestro_name[-7:] != ".pickle") and (save_maestro_name[-4:] != ".pkl"):
                    save_maestro_name = save_maestro_name + ".pickle"
                print("Saving Maestro trial data as:", save_maestro_name)
                with open(save_maestro_name, 'wb') as fp:
                    pickle.dump(maestro_data, fp, protocol=-1)
            else:
                print("Unrecognized type {0} for save_name {1}, saving skipped!".format(type(save_maestro_name), save_maestro_name))

    with open(neurons_file, 'rb') as fp:
        neurons = pickle.load(fp)

    trial_list_nrn, neuron_meta = sa_format_dicts.maestro_to_neuron_trial(
                                            maestro_data, neurons, dt_data=dt_data,
                                            start_data=0, default_name="n_",
                                            use_class_names=True, data_name='neurons')
    # This is an annoying error better to catch before trying to add
    if len(ldp_sess) != len(trial_list_nrn):
        raise ValueError("List of neuron trials is {0} and not equal to the current session length {1}!".format(len(trial_list_nrn), len(ldp_sess)))
    ldp_sess.add_neuron_trials(trial_list_nrn, neuron_meta, meta_dict_name='meta_data')
    # Delete unsynced trials after matching trials from joining above
    bad_inds = []
    for t_ind, t in enumerate(maestro_data):
        if not t['pl2_synced']:
            print("Adding trial {0} for deletion since it is not PL2 synced.".format(t_ind))
            bad_inds.append(t_ind)
    if len(bad_inds) > 0:
        print("Removing un-synced trials: ", bad_inds)
        ldp_sess.delete_trials(bad_inds)

    return ldp_sess



def format_ldp_trials_blocks(ldp_sess, sac_ind_cushion=40, verbose=True):

    # Align all target related events with monitor refresh rate
    ldp_sess.shift_event_to_refresh('target_onset')
    ldp_sess.shift_event_to_refresh('fixation_onset')
    ldp_sess.shift_event_to_refresh('instruction_onset')
    ldp_sess.shift_event_to_refresh('rand_fix_onset')
    ldp_sess.shift_event_to_refresh('start_stabwin')
    ldp_sess.shift_event_to_refresh('target_offset')

    if verbose: print("Searching for incomplete trials.")
    # Fixation trials will be found and aligned by this many ms after fixation onset
    fixation_trial_t_offset = 1200.
    # Pursuit trials will be found and aligned by this many ms after target onset
    pursuit_trial_min_motion = 500.
    pursuit_trial_t_offset = 0.
    # Remove trials that were not long enough to start
    fix_trial_names = ['d014fix', 'd0-14fix', 'd-1010fix', 'd1010fix', 'd140fix', 'd-10-10fix', 'd10-10fix', 'd-140fix', 'd00fix']
    ldp_sess.add_trial_set("fixation_trials", trials=fix_trial_names, blocks=None)
    ldp_sess.add_trial_set("pursuit_trials", trials=~ldp_sess.trial_sets['fixation_trials'], blocks=None)
    fix_trials_less_than_event = ldp_sess.find_trials_less_than_event("fixation_onset",
                                                          blocks=None,
                                                          trial_sets="fixation_trials",
                                                          event_offset=fixation_trial_t_offset)
    # Find target trials that didn't make it to the end of the trial
    trials_less_than_event = ldp_sess.find_trials_less_than_event("target_onset",
                                blocks=None, trial_sets="pursuit_trials",
                                event_offset=pursuit_trial_min_motion)
    # trials_less_than_event = ldp_sess.find_trials_less_than_event("target_offset",
    #                             blocks=None, trial_sets="pursuit_trials",
    #                             event_offset=pursuit_trial_t_offset)
    # Delete these too short trials
    ldp_sess.delete_trials(np.logical_or(fix_trials_less_than_event, trials_less_than_event))

    weird_yoda_tuning_trials = ['195', '165', '210', '315', '150', '225', '45',
                                '135', '255', '285', '240', '300', '120', '105',
                                '75', '60']

    # Add hard coded blocks by trial names
    block_names = ['FixTune']
    ldp_sess.add_blocks(fix_trial_names, block_names, number_names=True, block_min=9)

    trial_names = ['270RandVP', '90RandVP', '180RandVP', '0RandVP']
    block_names = ['RandVP']
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, block_min=20, n_min_per_trial=5)

    trial_names = ['90', '0', '180','270']
    ignore_trial_names = weird_yoda_tuning_trials if ldp_sess.is_weird_Yoda else ['']
    block_names = ['StandTune']
    ldp_sess.add_blocks(trial_names, block_names, number_names=True,
                        ignore_trial_names=ignore_trial_names, block_min=12,
                        n_min_per_trial=3, max_consec_single=20)

    trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['StabTune']
    ldp_sess.add_blocks(trial_names, block_names, number_names=True,
                        block_min=12, n_min_per_trial=3, max_consec_single=20)

    trial_names = ['0-upStab'] if ldp_sess.is_stab_learning else ["0-up"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if ldp_sess.is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['0Learn90']
    ldp_sess.block_name_to_learn_name['0Learn90'] = '0-upStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['0-dnStab'] if ldp_sess.is_stab_learning else ["0-dn"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if ldp_sess.is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['0Learn270']
    ldp_sess.block_name_to_learn_name['0Learn270'] = '0-dnStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['90-rtStab'] if ldp_sess.is_stab_learning else ["90-rt"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if ldp_sess.is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['90Learn0']
    ldp_sess.block_name_to_learn_name['90Learn0'] = '90-rtStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['90-ltStab'] if ldp_sess.is_stab_learning else ["90-lt"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if ldp_sess.is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['90Learn180']
    ldp_sess.block_name_to_learn_name['90Learn180'] = '90-ltStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['180-upStab'] if ldp_sess.is_stab_learning else ["180-up"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if ldp_sess.is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['180Learn90']
    ldp_sess.block_name_to_learn_name['180Learn90'] = '180-upStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['180-dnStab'] if ldp_sess.is_stab_learning else ["180-dn"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if ldp_sess.is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['180Learn270']
    ldp_sess.block_name_to_learn_name['180Learn270'] = '180-dnStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['270-rtStab'] if ldp_sess.is_stab_learning else ["270-rt"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if ldp_sess.is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['270Learn0']
    ldp_sess.block_name_to_learn_name['270Learn0'] = '270-rtStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['270-ltStab'] if ldp_sess.is_stab_learning else ["270-lt"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if ldp_sess.is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['270Learn180']
    ldp_sess.block_name_to_learn_name['270Learn180'] = '270-ltStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    # Attempt to verify block order and detect trials outside of blocks
    if verbose: print("Checking block assignments.")
    orphan_trials = ldp_sess.verify_blocks()
    # Setup learning direction and trial type metadata
    if verbose: print("Choosing learning/pursuit directions and block names.")
    ldp_sess.assign_block_names()

    if len(orphan_trials) > 0:
        if verbose: print("Attempting to repair {0} orphan trials.".format(len(orphan_trials)))
        n_orphans_assigned = ldp_sess.assign_orphan_trials(orphan_trials)
        if verbose: print("Added {0} orphan trials to blocks.".format(n_orphans_assigned))

    orphan_trials = ldp_sess.verify_blocks()
    if verbose: print("New block assignments leave {0} orphan trials for deletion.".format(len(orphan_trials)))
    ldp_sess.delete_trials(orphan_trials)

    # Align trials on events
    if verbose: print("Aligning trials on events and assigning default trial sets.")
    fixation_blocks = []
    pursuit_blocks = []
    for b_name in ldp_sess.get_block_names():
        if "Fix" in b_name:
            fixation_blocks.append(b_name)
        else:
            pursuit_blocks.append(b_name)
    ldp_sess.align_trial_data('target_onset', alignment_offset=0, blocks=pursuit_blocks)
    ldp_sess.align_trial_data('fixation_onset', alignment_offset=fixation_trial_t_offset, blocks=fixation_blocks)
    ldp_sess.add_default_trial_sets()

    if verbose: print("Adjusting fixation offsets and getting saccades.")
    ldp_sess.add_saccades(time_window=[-200, 0], blocks=None,
                          trial_sets=None, ind_cushion=sac_ind_cushion)

    if verbose: print("Deleting large saccade and position error trials.")
    max_sacc_amp = 6.
    ldp_sess.trial_sets['not_inst'] = ~np.logical_or(
                                        ldp_sess.trial_sets['instruction'],
                                        ldp_sess.trial_sets['fixation_trials'])
    t_inds_to_delete = ldp_sess.set_sacc_and_err_trials([-100, 325], max_sacc_amp=max_sacc_amp,
                            max_pos_err=10., trial_sets="instruction")
    if verbose: print("Set {0} 'instruction' trials with large errors.".format(len(t_inds_to_delete)))
    t_inds_to_delete = ldp_sess.set_sacc_and_err_trials([600, 1200], max_sacc_amp=max_sacc_amp,
                            max_pos_err=5., trial_sets="fixation_trials")
    if verbose: print("Set {0} 'fixation' trials with large errors.".format(len(t_inds_to_delete)))
    t_inds_to_delete = ldp_sess.set_sacc_and_err_trials([-100, 500], max_sacc_amp=max_sacc_amp,
                            max_pos_err=6., trial_sets="not_inst")
    if verbose: print("Set {0} 'non-instruction' trials with large errors.".format(len(t_inds_to_delete)))
    # Set to only pull trial indices from blocks and sets that do not have errors
    ldp_sess.rem_sacc_errs = True
    # Compute the indices for counting by number of preceding learning trials
    ldp_sess.get_n_instructed_trials(100)

    if verbose: print("Found learning block trial as:", ldp_sess.learning_trial_name, "and corresponding directions:", ldp_sess.directions)

    learn_stabilized = ldp_sess.is_stabilized("Learning", trial_set="instruction")
    check_trials = np.ones(len(ldp_sess), dtype='bool')
    for dir in ldp_sess.directions:
        check_trials = np.logical_or(check_trials, ldp_sess.trial_sets[dir])
    probe_stabilized = ldp_sess.is_stabilized("Learning", trial_set=check_trials)

    return ldp_sess
