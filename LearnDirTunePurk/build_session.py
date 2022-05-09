import numpy as np
from LearnDirTunePurk.load_data import load_maestro_directory
from LearnDirTunePurk import format_trials
from LearnDirTunePurk.ldp_session import LDPSession
import ReadMaestro as rm
import SessionAnalysis as sa



def create_behavior_session(fname, maestro_dir, session_name=None, existing_dir=None,
        save_dir=None, verbose=True):
    if session_name is None:
        session_name = maestro_dir.split("/")[-1]
    # Load all Maestro data
    if verbose: print("Loading Maestro directory data.")
    maestro_data = load_maestro_directory(fname, maestro_dir, existing_dir=existing_dir, save_dir=save_dir)
    # Sets trial targets as objects so probably want to do this after loading and not saved
    if verbose: print("Formatting target data and trials.")
    rm.format_trials.data_to_target(maestro_data)

    # Reformat the multi targets and names of trials into one and name events
    if verbose: print("Renaming trials and events.")
    format_trials.rename_stab_probe_trials(maestro_data)
    format_trials.name_trial_events(maestro_data)

    # Create base list of ApparatusTrial trials from target0
    if verbose: print("Converting data to trial objects.")
    trial_list = sa.utils.format_trial_dicts.maestro_to_apparatus_trial(
                        maestro_data, 0, 1, start_data=0, data_name="target0")
    # Create a second list of BehaviorTrial trials from eye data
    trial_list_bhv = sa.utils.format_trial_dicts.maestro_to_behavior_trial(
                        maestro_data, 1, start_data=0, data_name="eye")

    # Create base session from apparatus trials then add behavior
    # sess = sa.session.Session(trial_list, session_name=session_name)
    if verbose: print("Generating session and adding blocks.")
    sess = LDPSession(trial_list, session_name=session_name)
    sess.add_trial_data(trial_list_bhv, data_type=None)

    # Add hard coded blocks by trial names
    trial_names = ['d014fix', 'd0-14fix', 'd-1010fix', 'd1010fix', 'd140fix', 'd-10-10fix', 'd10-10fix', 'd-140fix', 'd00fix']
    block_names = ['FixTunePre', 'FixTunePost', 'FixTuneWashout']
    sess.add_blocks(trial_names, block_names, block_min=9)

    trial_names = ['270RandVP', '90RandVP', '180RandVP', '0RandVP']
    block_names = ['RandVP']
    sess.add_blocks(trial_names, block_names, block_min=20, n_min_per_trial=5)

    trial_names = ['90', '0', '180','270']
    block_names = ['TunePre', 'TunePost']
    sess.add_blocks(trial_names, block_names, block_min=12, n_min_per_trial=3)

    trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['StabPre', 'StabPost', 'StabWashout']
    sess.add_blocks(trial_names, block_names, block_min=12, n_min_per_trial=3)

    trial_names = ['0-upStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    block_names = ['0Learn90']
    sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['0-dnStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    block_names = ['0Learn270']
    sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['90-rtStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    block_names = ['90Learn0']
    sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['90-ltStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    block_names = ['90Learn180']
    sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['180-upStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    block_names = ['180Learn90']
    sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['180-dnStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    block_names = ['180Learn270']
    sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['270-rtStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    block_names = ['270Learn0']
    sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    trial_names = ['270-ltStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab', '90', '0', '180','270']
    block_names = ['270Learn180']
    sess.add_blocks(trial_names, block_names, number_names=True, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=0, block_min=20, n_min_per_trial=20)

    # Align all target related events with monitor refresh rate
    sess.shift_event_to_refresh('target_onset')
    sess.shift_event_to_refresh('fixation_onset')
    sess.shift_event_to_refresh('instruction_onset')
    sess.shift_event_to_refresh('rand_fix_onset')
    sess.shift_event_to_refresh('start_stabwin')
    sess.shift_event_to_refresh('target_offset')

    # Fixation trials will be found and aligned by this many ms after target onset
    fixation_trial_t_offset = 1200.
    # Remove trials that were not long enough to start
    # Find fixation tuning trials that lasted less than 800 ms
    fixation_blocks = []
    for b_name in sess.block_names():
        if "Fix" in b_name:
            fixation_blocks.append(b_name)
    fix_trials_less_than_event = sess.find_trials_less_than_event("fixation_onset",
                                                          blocks=fixation_blocks,
                                                          trial_sets=None,
                                                          event_offset=fixation_trial_t_offset)
    if np.count_nonzero(fix_trials_less_than_event) <= 5:
        print("Session '{0}' fixation trials are shorter than offset (build_session line ~117).".format(session_name))
    # Find target trials that didn't make it to target motion onset
    trials_less_than_event = sess.find_trials_less_than_event("target_onset", blocks=None, trial_sets=None)
    # Delete these trials
    sess.delete_trials(np.logical_or(fix_trials_less_than_event, trials_less_than_event))

    # Align trials on events
    # First non-fixation only trials
    blocks = []
    for blk in sess.block_names():
        if blk in fixation_blocks:
            continue
        blocks.append(blk)
    sess.align_trial_data('target_onset', alignment_offset=0, blocks=blocks)
    # Then fixation only trials
    sess.align_trial_data('fixation_onset', alignment_offset=fixation_trial_t_offset, blocks=fixation_blocks)

    # Setup learning direction and trial type metadata for easier indexing later
    if verbose: print("Choosing learning/pursuit directions and default trial sets.")
    return sess
    sess.assign_learning_blocks()
    sess.add_default_trial_sets()

    if verbose: print("Adjusting fixation offsets and getting saccades.")
    sess.add_saccades(time_window=[-400, 0], blocks=None, trial_sets=None)

    # Compute the indices for counting by number of preceding learning trials
    sess.get_n_instructed_trials(100)

    return sess
