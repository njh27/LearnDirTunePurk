import numpy as np
import pickle
from LearnDirTunePurk.load_data import load_maestro_directory
from LearnDirTunePurk import format_trials
from LearnDirTunePurk.ldp_session import LDPSession
import ReadMaestro as rm
import SessionAnalysis as sa



def create_behavior_session(fname, maestro_dir=None, session_name=None, existing_dir=None,
        save_dir=None, verbose=True):
    if session_name is None:
        session_name = fname
    # Load all Maestro data
    if verbose: print("Loading Maestro directory data.")
    if maestro_dir is None and existing_dir is not None:
        # Load an existing file directly
        existing_dir = existing_dir + "/" if (existing_dir[-1] != "/") else existing_dir
        with open(existing_dir + fname, 'rb') as fp:
            maestro_data = pickle.load(fp)
    else:
        # Try to find a file from a Maestro directory or existing pickle
        maestro_data = load_maestro_directory(fname, maestro_dir, existing_dir=existing_dir, save_dir=save_dir)
    if ( ("Yoda" in session_name) and (int(session_name.split("_")[-1]) < 20) ):
        print("Treating this as a weird Yoda file")
        is_weird_Yoda = True
    else:
        is_weird_Yoda = False
    # Sets trial targets as objects so probably want to do this after loading and not saved
    if verbose: print("Formatting target data and trials.")
    rm.format_trials.data_to_target(maestro_data)

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
    trial_list = sa.utils.format_trial_dicts.maestro_to_apparatus_trial(
                        maestro_data, 0, 1, start_data=0, data_name="target0")

    # Create a second list of BehaviorTrial trials from eye data
    trial_list_bhv = sa.utils.format_trial_dicts.maestro_to_behavior_trial(
                        maestro_data, 1, start_data=0, data_name="eye")

    # Create base session from apparatus trials then add behavior
    # ldp_sess = sa.session.Session(trial_list, session_name=session_name)
    if verbose: print("Generating session and adding blocks.")
    ldp_sess = LDPSession(trial_list, session_name=session_name)
    ldp_sess.add_trial_data(trial_list_bhv, data_type=None)

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
    ignore_trial_names = weird_yoda_tuning_trials if is_weird_Yoda else ['']
    block_names = ['StandTune']
    ldp_sess.add_blocks(trial_names, block_names, number_names=True,
                        ignore_trial_names=ignore_trial_names, block_min=12,
                        n_min_per_trial=3, max_consec_single=20)

    trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['StabTune']
    ldp_sess.add_blocks(trial_names, block_names, number_names=True,
                        block_min=12, n_min_per_trial=3, max_consec_single=20)

    trial_names = ['0-upStab'] if is_stab_learning else ["0-up"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['0Learn90']
    ldp_sess.block_name_to_learn_name['0Learn90'] = '0-upStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['0-dnStab'] if is_stab_learning else ["0-dn"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['0Learn270']
    ldp_sess.block_name_to_learn_name['0Learn270'] = '0-dnStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['90-rtStab'] if is_stab_learning else ["90-rt"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['90Learn0']
    ldp_sess.block_name_to_learn_name['90Learn0'] = '90-rtStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['90-ltStab'] if is_stab_learning else ["90-lt"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['90Learn180']
    ldp_sess.block_name_to_learn_name['90Learn180'] = '90-ltStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['180-upStab'] if is_stab_learning else ["180-up"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['180Learn90']
    ldp_sess.block_name_to_learn_name['180Learn90'] = '180-upStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['180-dnStab'] if is_stab_learning else ["180-dn"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['180Learn270']
    ldp_sess.block_name_to_learn_name['180Learn270'] = '180-dnStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['270-rtStab'] if is_stab_learning else ["270-rt"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if is_weird_Yoda:
        ignore_trial_names.extend(weird_yoda_tuning_trials)
    block_names = ['270Learn0']
    ldp_sess.block_name_to_learn_name['270Learn0'] = '270-rtStab'
    ldp_sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['270-ltStab'] if is_stab_learning else ["270-lt"]
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    if is_weird_Yoda:
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
    for b_name in ldp_sess.block_names():
        if "Fix" in b_name:
            fixation_blocks.append(b_name)
        else:
            pursuit_blocks.append(b_name)
    ldp_sess.align_trial_data('target_onset', alignment_offset=0, blocks=pursuit_blocks)
    ldp_sess.align_trial_data('fixation_onset', alignment_offset=fixation_trial_t_offset, blocks=fixation_blocks)
    ldp_sess.add_default_trial_sets()

    if verbose: print("Adjusting fixation offsets and getting saccades.")
    ldp_sess.add_saccades(time_window=[-200, 0], blocks=None,
                          trial_sets=None, ind_cushion=20)

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
