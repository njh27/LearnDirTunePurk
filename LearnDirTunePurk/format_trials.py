from SessionAnalysis.utils.format_trial_dicts import format_maestro_events



def rename_stab_probe_trials(maestro_data):
    """Kept this direct and simple rather than more general and flexible since
    hopefull I won't need to do this a lot.
    """
    tune_names = ["0", "90", "180", "270"]
    learn_names = ['Right-Up', 'Right-Dn',
                   'Up-Rt', 'Up-Lt',
                   'Left-Up', 'Left-Dn',
                   'Down-Rt', 'Down-Lt',
                   'Dn-Rt', 'Dn-Lt']
    learn_names = [x.lower() for x in learn_names]
    found_learn = False
    for t in maestro_data:
        if t['header']['name'] in ["0", "90", "180", "270"]:
            if t['header']['set_name'].lower() in learn_names:
                found_learn = True
            if t['header']['UsedStab']:
                t['header']['name'] = t['header']['name'] + "Stab"

    if not found_learn:
        raise ValueError("Could not find tuning trials within the learning set names provided")

    return None


def name_trial_events(maestro_data):
    """Assigns the event names to even times dictionary for each trial
    IN PLACE.

    NOTE THAT the post learning standard tuning block in Dandy is messed up
    because it has the extra events as used for the Stab tuning!!! So these
    are explictily caught and renamed....
    """

    # Set hard coded variables for expected trial name dictionaries.
    event_names_fixation = {
        "fixation_onset": [2, 0]
        }
    event_names_rand_vp = {
        "fixation_onset": [0, 0],
        "rand_fix_onset": [1, 0],
        "target_onset": [2, 0],
        "start_stabwin": [3, 0],
        "target_offset": [4, 0]
        }
    event_names_stand_tuning = {
        "fixation_onset": [0, 0],
        "rand_fix_onset": [1, 0],
        "target_onset": [2, 0],
        "start_stabwin": [3, 0],
        "target_offset": [4, 0]
        }
    event_names_stab_tuning = {
        "fixation_onset": [0, 0],
        "rand_fix_onset": [1, 0],
        "target_onset": [2, 0],
        "start_stabwin": [3, 0],
        "instruction_onset": [4, 0],
        "target_offset": [5, 0]
        }
    event_names_learning = {
        "fixation_onset": [0, 0],
        "rand_fix_onset": [1, 0],
        "target_onset": [2, 0],
        "start_stabwin": [3, 0],
        "instruction_onset": [4, 0],
        "target_offset": [5, 0]
        }

    # Generate the naming dictionary for each trial name
    maestro_trial_names = set()
    for t in maestro_data:
        if t['header']['name'] not in maestro_trial_names:
            maestro_trial_names.add(t['header']['name'])
    event_names_by_trial = {}
    for t_name in maestro_trial_names:
        if "fix" in t_name:
            event_names_by_trial[t_name] = event_names_fixation
        elif "RandVP" in t_name:
            event_names_by_trial[t_name] = event_names_rand_vp
        elif (t_name in ["0", "90", "180", "270"]):
            event_names_by_trial[t_name] = event_names_stand_tuning
        elif t_name in ["0Stab", "90Stab", "180Stab", "270Stab"]:
            event_names_by_trial[t_name] = event_names_stab_tuning
        elif "-rt" in t_name:
            event_names_by_trial[t_name] = event_names_learning
            after_learing = True
        elif "-up" in t_name:
            event_names_by_trial[t_name] = event_names_learning
            after_learing = True
        elif "-lt" in t_name:
            event_names_by_trial[t_name] = event_names_learning
            after_learing = True
        elif "-dn" in t_name:
            event_names_by_trial[t_name] = event_names_learning
            after_learing = True
        else:
            raise ValueError("T name '{0}' not found!".format(t_name))

    format_maestro_events(maestro_data, event_names_by_trial,
            missing_event=None, convert_to_ms=True)

    return None
