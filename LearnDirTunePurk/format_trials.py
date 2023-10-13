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
    no_set_name = False
    print_stab = True
    print_learn = True
    for t in maestro_data:
        if t['header']['name'] in ['90Stab', '0Stab', '180Stab','270Stab']:
            if not t['header']['UsedStab']:
                raise ValueError("Trial name is {0} but 'UsedStab' is {1}.".format(t['header']['name'], t['header']['UsedStab']))
        if t['header']['name'] in ["0", "90", "180", "270"]:
            # try:
            #     # This is only available in Maestro version >= 4.0
            #     # print(t['header']['name'])
            #     # print(t['header']['set_name'])
            #     if t['header']['set_name'].lower() in learn_names:
            #         found_learn = True
            # except KeyError:
            #     no_set_name = True
            if t['header']['UsedStab']:
                t['header']['name'] = t['header']['name'] + "Stab"
                if print_stab:
                    print("This file has stabilization but trial names do not reflect this! Added 'Stab' to tuning names.")
                    print_stab = False
        if ("-left" in t['header']['name']):
            old_name = t['header']['name']
            new_name = t['header']['name'].replace("-left", "-lt")
            t['header']['name'] = new_name
            if t['header']['UsedStab']:
                if "Stab" not in t['header']['name']:
                    t['header']['name'] = t['header']['name'] + "Stab"
                    new_name = t['header']['name']
            if print_learn:
                print("This file has old learning name {0} which was changed to {1}.".format(old_name, new_name))
                print_learn = False
        if ("-right" in t['header']['name']):
            old_name = t['header']['name']
            new_name = t['header']['name'].replace("-right", "-rt")
            t['header']['name'] = new_name
            if t['header']['UsedStab']:
                if "Stab" not in t['header']['name']:
                    t['header']['name'] = t['header']['name'] + "Stab"
                    new_name = t['header']['name']
            if print_learn:
                print("This file has old learning name {0} which was changed to {1}.".format(old_name, new_name))
                print_learn = False
        if ("-down" in t['header']['name']):
            old_name = t['header']['name']
            new_name = t['header']['name'].replace("-down", "-dn")
            t['header']['name'] = new_name
            if t['header']['UsedStab']:
                if "Stab" not in t['header']['name']:
                    t['header']['name'] = t['header']['name'] + "Stab"
                    new_name = t['header']['name']
            if print_learn:
                print("This file has old learning name {0} which was changed to {1}.".format(old_name, new_name))
                print_learn = False
        if ("-up" in t['header']['name']):
            # Should already be named correctly so just check Stab
            old_name = t['header']['name']
            # new_name = t['header']['name'].replace("-up", "-up")
            # t['header']['name'] = new_name
            if t['header']['UsedStab']:
                if "Stab" not in t['header']['name']:
                    t['header']['name'] = t['header']['name'] + "Stab"
                    new_name = t['header']['name']
                    if print_learn:
                        print("This file has old learning name {0} which was changed to {1}.".format(old_name, new_name))
                        print_learn = False

    # if no_set_name:
    #     print("File does not have set name")
    # elif not found_learn:
    #     raise ValueError("Could not find learning trials within the learning set names provided")

    return None


def name_trial_events(maestro_data, is_weird_Yoda=False, is_weird_Yan=False):
    """Assigns the event names to event times dictionary for each trial
    IN PLACE.

    NOTE THAT the post learning standard tuning block in Dandy is messed up
    because it has the extra events as used for the Stab tuning!!! So these
    are explictily caught and renamed....
    """

    # Set hard coded variables for expected trial name dictionaries.
    if is_weird_Yan:
        event_names_sinusoid = {
            "fixation_onset": [1, 0],
            "rand_fix_onset": [1, 0],
            "target_onset": [1, 2],
            }
        event_names_stand_tuning = {
            "fixation_onset": [1, 0],
            "rand_fix_onset": [1, 0],
            "target_onset": [1, 2],
            "target_offset": [1, 3]
            }
        event_names_learning = {
            "fixation_onset": [1, 0],
            "rand_fix_onset": [1, 0],
            "target_onset": [1, 2],
            "instruction_onset": [1, 3],
            "target_offset": [1, 4]
            }
    elif is_weird_Yoda:
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
            "target_onset": [1, 1],
            "target_offset": [1, 3]
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
            "target_onset": [1, 1],
            "instruction_onset": [1, 2],
            "target_offset": [1, 3]
            }
    else:
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

    weird_yoda_tuning_trials = ['195', '165', '210', '315', '150', '225', '45',
                                '135', '255', '285', '240', '300', '120', '105',
                                '75', '60']
    weird_yan_tuning_trials = ['H', 'V']

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
        elif "-up" in t_name:
            event_names_by_trial[t_name] = event_names_learning
        elif "-lt" in t_name:
            event_names_by_trial[t_name] = event_names_learning
        elif "-dn" in t_name:
            event_names_by_trial[t_name] = event_names_learning
        elif "right" in t_name:
            event_names_by_trial[t_name] = event_names_learning
        elif "up" in t_name:
            event_names_by_trial[t_name] = event_names_learning
        elif "left" in t_name:
            event_names_by_trial[t_name] = event_names_learning
        elif "down" in t_name:
            event_names_by_trial[t_name] = event_names_learning
        elif t_name in weird_yoda_tuning_trials:
            event_names_by_trial[t_name] = event_names_stand_tuning
        elif t_name in weird_yan_tuning_trials:
            event_names_by_trial[t_name] = event_names_sinusoid
        else:
            raise ValueError("T name '{0}' not found! Names present: {1}".format(t_name, maestro_trial_names))

    format_maestro_events(maestro_data, event_names_by_trial,
            missing_event=None, convert_to_ms=True)

    return None
