import numpy as np
from LearnDirTunePurk.load_data import load_maestro_directory
from LearnDirTunePurk import format_trials
import ReadMaestro as rm
import SessionAnalysis as sa



def create_behavior_session(maestro_dir, session_name=None, check_existing=True,
        save_data=True):
    if session_name is None:
        session_name = maestro_dir.split("/")[-1]
    # Load all Maestro data
    maestro_data = load_maestro_directory(maestro_dir, check_existing=True, save_data=True)
    # Sets trial targets as objects so probably want to do this after loading and not saved
    rm.format_trials.data_to_target(maestro_data)

    # Reformat the multi targets and names of trials into one and name events
    format_trials.rename_stab_probe_trials(maestro_data)
    format_trials.name_trial_events(maestro_data)

    # Create base list of ApparatusTrial trials from target0
    trial_list = sa.utils.format_trial_dicts.maestro_to_apparatus_trial(
                        maestro_data, 0, 1, start_data=0, data_name="target0")
    # Create a second list of BehaviorTrial trials from eye data
    trial_list_bhv = sa.utils.format_trial_dicts.maestro_to_behavior_trial(
                        maestro_data, 1, start_data=0, data_name="eye")

    # Create base session from apparatus trials then add behavior
    sess = sa.session.Session(trial_list, session_name=session_name)
    sess.add_trial_data(trial_list_bhv, data_type=None)

    # Add hard coded blocks by trial names
    trial_names = ['d014fix', 'd0-14fix', 'd-1010fix', 'd1010fix', 'd140fix', 'd-10-10fix', 'd10-10fix', 'd-140fix', 'd00fix']
    block_names = ['fix_tune1', 'fix_tune2']
    sess.add_blocks(trial_names, block_names, block_min=9)

    trial_names = ['270RandVP', '90RandVP', '180RandVP', '0RandVP']
    block_names = ['RandVP']
    sess.add_blocks(trial_names, block_names, block_min=20)

    trial_names = ['90', '0', '180','270']
    block_names = ['TunePre', 'TunePost']
    sess.add_blocks(trial_names, block_names, block_min=8)

    trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['StabPre', 'StabPost']
    sess.add_blocks(trial_names, block_names, block_min=8)

    trial_names = ['0-upStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['0Learn90']
    sess.add_blocks(trial_names, block_names, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=10, block_min=20)

    trial_names = ['0-dnStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['0Learn270']
    sess.add_blocks(trial_names, block_names, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=10, block_min=20)

    trial_names = ['90-rtStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['90Learn0']
    sess.add_blocks(trial_names, block_names, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=10, block_min=20)

    trial_names = ['90-ltStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['90Learn180']
    sess.add_blocks(trial_names, block_names, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=10, block_min=20)

    trial_names = ['180-upStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['180Learn90']
    sess.add_blocks(trial_names, block_names, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=10, block_min=20)

    trial_names = ['180-dnStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['180Learn270']
    sess.add_blocks(trial_names, block_names, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=10, block_min=20)

    trial_names = ['270-rtStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['270Learn0']
    sess.add_blocks(trial_names, block_names, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=10, block_min=20)

    trial_names = ['270-ltStab']
    ignore_trial_names = ['90Stab', '0Stab', '180Stab','270Stab']
    block_names = ['270Learn180']
    sess.add_blocks(trial_names, block_names, ignore_trial_names=ignore_trial_names,
                    max_consec_absent=10, block_min=20)


    return sess
