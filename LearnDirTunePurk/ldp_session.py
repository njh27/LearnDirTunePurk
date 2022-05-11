import numpy as np
import re
from SessionAnalysis.session import Session
from SessionAnalysis.utils import eye_data_series



def make_180(angle):
    """ Helper function that, given an input angle in degrees, returns the same
    angle on the interval [-179, 180]. """
    # reduce the angle
    angle = angle % 360
    # force it to be the positive remainder, so that 0 <= angle < 360
    angle = (angle + 360) % 360
    # force into the minimum absolute value residue class, so that -180 < angle <= 180
    if (angle > 180):
        angle -= 360

    return angle


class LDPSession(Session):

    def __init__(self, trial_data, session_name=None, data_type=None):
        """
        """
        Session.__init__(self, trial_data, session_name, data_type)

    def verify_blocks(self):
        is_blocks_continuous, trials_missing = self._verify_block_continuity()
        if not is_blocks_continuous:
            print("These trial numbers are missing a block: ", trials_missing)
        self._verify_block_overlap()
        return None

    def assign_block_names(self, learn_sn="Learn", fix_sn="FixTune",
            stab_sn="StabTune", stand_sn="StandTune", randvp_sn="RandVP"):
        self.set_learning_block(search_name=learn_sn)
        self.parse_learning_directions(search_name=learn_sn)
        self.set_washout_block()
        self.set_fixation_tuning_blocks(search_name=fix_sn)
        self.set_stab_tuning_blocks(search_name=stab_sn)
        self.set_stand_tuning_blocks(search_name=stand_sn)
        self.set_randvp_tuning_blocks(search_name=randvp_sn)
        return None

    def set_learning_block(self, search_name="Learn"):
        """ Parse through the block names and find the longest learning block
        to designate the "learning" direction and block.
        """
        n_max_learn = 0
        b_max_learn = None
        for block in self.blocks.keys():
            if search_name in block:
                block_len = self.blocks[block][1] - self.blocks[block][0]
                if block_len > n_max_learn:
                    n_max_learn = block_len
                    b_max_learn = block
        if b_max_learn is not None:
            self.blocks['Learning'] = self.blocks[b_max_learn]
            self.learning_trial_name = b_max_learn
        else:
            print("No learning block found!")
            self.blocks['Learning'] = None
        return None

    def parse_learning_directions(self, search_name="Learn"):
        """ Determine the tuning directions with respect to the learning block
        direction. """
        if self.blocks['Learning'] is None:
            raise ValueError("Cannot parse learning directions with no learning block assigned!")
        # We found a learning block so carry on
        pursuit_dir, learn_dir = self.learning_trial_name.split(search_name)
        pursuit_dir = int(pursuit_dir)
        learn_dir = int(learn_dir)
        anti_pursuit_dir = (pursuit_dir + 180) % 360
        anti_learn_dir = (learn_dir + 180) % 360
        self.directions = {}
        self.directions['pursuit'] = pursuit_dir
        self.directions['learning'] = learn_dir
        self.directions['anti_pursuit'] = anti_pursuit_dir
        self.directions['anti_learning'] = anti_learn_dir
        # New learning directions means new rotation matrix
        self._set_rotation_matrix()
        return None

    def set_washout_block(self):
        """ Finds the first block in the anti-learning direction following the
        learning block and sets it as the washout block. """
        if self.blocks['Learning'] is None:
            raise ValueError("Cannot parse washout block with no learning block assigned!")
        washout_trial_name = str(self.directions['pursuit']) + "Learn" + str(self.directions['anti_learning'])
        washout_block_start = np.inf
        washout_block_name = None
        for block in self.blocks.keys():
            # Do split on num for default number counting of multiple blocks
            if ( (block.split("_num")[0] == washout_trial_name) and
                  (self.blocks[block][0] >= self.blocks['Learning'][1]) and
                  (self.blocks[block][0] < washout_block_start) ):
                # This is an anti-learning block that follows learning block
                # the closest, so is washout block
                washout_block_start = self.blocks[block][0]
                washout_block_name = block
        if washout_block_name is None:
            # did not find a washout block
            self.blocks['Washout'] = None
        else:
            self.blocks['Washout'] = self.blocks[washout_block_name]
        return None

    def set_fixation_tuning_blocks(self, search_name="FixTune",
                                    min_trials_by_name=0):
        """ Finds the fixation tuning blocks relative to learning/washout. """
        if self.blocks['Learning'] is None:
            raise ValueError("Cannot parse washout block with no learning block assigned!")
        t_pre_start = -np.inf
        t_post_start = np.inf
        t_wash_start = np.inf
        search_blocks = ["FixTunePre", "FixTunePost", "FixTuneWash"]
        new_block_names = {x: None for x in search_blocks}
        for block in self.blocks.keys():
            # Do split on num for default number counting of multiple blocks
            if search_name in block:
                # Found a fixation trial, now find which one
                if ( (self.blocks[block][1] <= self.blocks['Learning'][0]) and
                     (self.blocks[block][0] >= t_pre_start) ):
                    # Tuning precedes learning and is latest found so far
                    new_block_names['FixTunePre'] = self.blocks[block]
                    t_pre_start = self.blocks[block][0]
                elif ( (self.blocks[block][0] >= self.blocks['Learning'][1]) and
                       (self.blocks[block][0] <= t_post_start) ):
                    # Tuning follows learning and is earliest found so far
                    new_block_names['FixTunePost'] = self.blocks[block]
                    t_post_start = self.blocks[block][0]
                if self.blocks['Washout'] is not None:
                    if ( (self.blocks[block][0] >= self.blocks['Washout'][1]) and
                         (self.blocks[block][0] <= t_wash_start) ):
                        # Tuning follows washout and is earliest found so far
                        new_block_names['FixTuneWash'] = self.blocks[block]
                        t_wash_start = self.blocks[block][0]
        self.blocks.update(new_block_names)
        return None

    def set_stab_tuning_blocks(self, search_name="StabTune",
                                    min_trials_by_name=0):
        """ Finds the fixation tuning blocks relative to learning/washout. """
        if self.blocks['Learning'] is None:
            raise ValueError("Cannot parse washout block with no learning block assigned!")
        t_pre_start = -np.inf
        t_post_start = np.inf
        t_wash_start = np.inf
        search_blocks = ["StabTunePre", "StabTunePost", "StabTuneWash"]
        new_block_names = {x: None for x in search_blocks}
        for block in self.blocks.keys():
            # Do split on num for default number counting of multiple blocks
            if search_name in block:
                # Found a fixation trial, now find which one
                if ( (self.blocks[block][1] <= self.blocks['Learning'][0]) and
                     (self.blocks[block][0] >= t_pre_start) ):
                    # Tuning precedes learning and is latest found so far
                    new_block_names['StabTunePre'] = self.blocks[block]
                    t_pre_start = self.blocks[block][0]
                elif ( (self.blocks[block][0] >= self.blocks['Learning'][1]) and
                       (self.blocks[block][0] <= t_post_start) ):
                    # Tuning follows learning and is earliest found so far
                    new_block_names['StabTunePost'] = self.blocks[block]
                    t_post_start = self.blocks[block][0]
                if self.blocks['Washout'] is not None:
                    if ( (self.blocks[block][0] >= self.blocks['Washout'][1]) and
                         (self.blocks[block][0] <= t_wash_start) ):
                        # Tuning follows washout and is earliest found so far
                        new_block_names['StabTuneWash'] = self.blocks[block]
                        t_wash_start = self.blocks[block][0]
        self.blocks.update(new_block_names)
        return None

    def set_stand_tuning_blocks(self, search_name="StandTune",
                                    min_trials_by_name=0):
        """ Finds the fixation tuning blocks relative to learning/washout. """
        if self.blocks['Learning'] is None:
            raise ValueError("Cannot parse washout block with no learning block assigned!")
        t_pre_start = -np.inf
        t_post_start = np.inf
        t_wash_start = np.inf
        search_blocks = ["StandTunePre", "StandTunePost", "StandTuneWash"]
        new_block_names = {x: None for x in search_blocks}
        for block in self.blocks.keys():
            # Do split on num for default number counting of multiple blocks
            if search_name in block:
                # Found a fixation trial, now find which one
                if ( (self.blocks[block][1] <= self.blocks['Learning'][0]) and
                     (self.blocks[block][0] >= t_pre_start) ):
                    # Tuning precedes learning and is latest found so far
                    new_block_names['StandTunePre'] = self.blocks[block]
                    t_pre_start = self.blocks[block][0]
                elif ( (self.blocks[block][0] >= self.blocks['Learning'][1]) and
                       (self.blocks[block][0] <= t_post_start) ):
                    # Tuning follows learning and is earliest found so far
                    new_block_names['StandTunePost'] = self.blocks[block]
                    t_post_start = self.blocks[block][0]
                if self.blocks['Washout'] is not None:
                    if ( (self.blocks[block][0] >= self.blocks['Washout'][1]) and
                         (self.blocks[block][0] <= t_wash_start) ):
                        # Tuning follows washout and is earliest found so far
                        new_block_names['StandTuneWash'] = self.blocks[block]
                        t_wash_start = self.blocks[block][0]
        self.blocks.update(new_block_names)
        return None

    def set_randvp_tuning_blocks(self, search_name="RandVP",
                                    min_trials_by_name=0):
        """ Finds the fixation tuning blocks relative to learning/washout. """
        if self.blocks['Learning'] is None:
            raise ValueError("Cannot parse washout block with no learning block assigned!")
        t_pre_start = -np.inf
        t_post_start = np.inf
        t_wash_start = np.inf
        search_blocks = ["RandVPTunePre", "RandVPTunePost", "RandVPTuneWash"]
        new_block_names = {x: None for x in search_blocks}
        for block in self.blocks.keys():
            # Do split on num for default number counting of multiple blocks
            if search_name in block:
                # Found a fixation trial, now find which one
                if ( (self.blocks[block][1] <= self.blocks['Learning'][0]) and
                     (self.blocks[block][0] >= t_pre_start) ):
                    # Tuning precedes learning and is latest found so far
                    new_block_names['RandVPTunePre'] = self.blocks[block]
                    t_pre_start = self.blocks[block][0]
                elif ( (self.blocks[block][0] >= self.blocks['Learning'][1]) and
                       (self.blocks[block][0] <= t_post_start) ):
                    # Tuning follows learning and is earliest found so far
                    new_block_names['RandVPTunePost'] = self.blocks[block]
                    t_post_start = self.blocks[block][0]
                if self.blocks['Washout'] is not None:
                    if ( (self.blocks[block][0] >= self.blocks['Washout'][1]) and
                         (self.blocks[block][0] <= t_wash_start) ):
                        # Tuning follows washout and is earliest found so far
                        new_block_names['RandVPTuneWash'] = self.blocks[block]
                        t_wash_start = self.blocks[block][0]
        self.blocks.update(new_block_names)
        return None

    def add_default_trial_sets(self):
        """ Adds boolean masks for all the expected trial types to the
        trial_sets dictionary to the session object which indicates
        whether a trial falls into the classes given in "directions" or whether it
        is a learning trial. """
        instructexp = re.compile(str(self.directions['pursuit'])+"-..")
        self.trial_sets['pursuit'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['learning'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['anti_pursuit'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['anti_learning'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['instruction'] = np.zeros(len(self), dtype='bool')
        for ind, st in enumerate(self._session_trial_data):
            if st['name'] == str(self.directions['pursuit']):
                self.trial_sets['pursuit'][ind] = True
            elif st['name'] == str(self.directions['pursuit']) + "RandVP":
                self.trial_sets['pursuit'][ind] = True
            elif st['name'] == str(self.directions['pursuit']) + "Stab":
                self.trial_sets['pursuit'][ind] = True

            elif st['name'] == str(self.directions['learning']):
                self.trial_sets['learning'][ind] = True
            elif st['name'] == str(self.directions['learning']) + "RandVP":
                self.trial_sets['learning'][ind] = True
            elif st['name'] == str(self.directions['learning']) + "Stab":
                self.trial_sets['learning'][ind] = True

            elif st['name'] == str(self.directions['anti_pursuit']):
                self.trial_sets['anti_pursuit'][ind] = True
            elif st['name'] == str(self.directions['anti_pursuit']) + "RandVP":
                self.trial_sets['anti_pursuit'][ind] = True
            elif st['name'] == str(self.directions['anti_pursuit']) + "Stab":
                self.trial_sets['anti_pursuit'][ind] = True

            elif st['name'] == str(self.directions['anti_learning']):
                self.trial_sets['anti_learning'][ind] = True
            elif st['name'] == str(self.directions['anti_learning']) + "RandVP":
                self.trial_sets['anti_learning'][ind] = True
            elif st['name'] == str(self.directions['anti_learning']) + "Stab":
                self.trial_sets['anti_learning'][ind] = True

            elif bool(re.match(instructexp, st['name'])):
                self.trial_sets['instruction'][ind] = True
        return None

    def _set_rotation_matrix(self):
        """ Sets the rotation matrix for transforming the pursuit_dir to angle
        0 and learn_dir to angle 90 for 2D vectors of eye/target position. Done
        based on the directions assigned to self.directions, e.g. via
        "assign_learning_blocks". """
        if ( (self.directions['pursuit'] >= 360) or (self.directions['pursuit'] < 0) or
             (self.directions['learning'] >= 360) or (self.directions['learning'] < 0) ):
            raise RuntimeError("Pursuit direction and learning direction not valid. Pursuit dir = {0} and Learning dir = {1}.".format(self.directions['pursuit'], self.directions['learning']))

        if ( (np.abs(self.directions['learning'] - self.directions['pursuit']) != 90) and
             ( (360 - np.abs(self.directions['learning'] - self.directions['pursuit'])) != 90) ):
             raise RuntimeError("Pursuit direction and learning direction not valid. Pursuit dir = {0} and Learning dir = {1}.".format(self.directions['pursuit'], self.directions['learning']))

        # Rotate clockwise for pursuit direction to be at angle 0
        rot_angle_rad = np.deg2rad(0 - self.directions['pursuit'])
        # Rotation matrix so pursuit direction is at 0 degrees, rotating clockwise
        rotate_pursuit_mat = np.array([[np.cos(rot_angle_rad), -1*np.sin(rot_angle_rad)],
                                       [np.sin(rot_angle_rad), np.cos(rot_angle_rad)]])
        # Round to nearest 5 decimals so that vectors stay on [0, 1], [1, 0], e.g.
        rotate_pursuit_mat = np.around(rotate_pursuit_mat, 5)
        mirror_xaxis = np.array([[1, 0], [0, -1]])
        if make_180(self.directions['learning'] - self.directions['pursuit']) == -90:
            # Need to mirror about the x-axis
            # Note: MUST be careful in re-using matrix variable names as @ operator
            #       overwrites things in place, so this seems slightly long
            self.rotation_matrix  = mirror_xaxis @ rotate_pursuit_mat
        else:
            self.rotation_matrix  = rotate_pursuit_mat
        return None

    def _verify_block_continuity(self):
        is_blocks_continuous = True
        trials_missing = []
        next_b_start = 0
        last_b_end = 0
        while next_b_start < len(self):
            found_match = False
            for b_name in self.blocks.keys():
                b_win = self.blocks[b_name]
                if b_win[0] == next_b_start:
                    found_match = True
                    last_b_end = b_win[1]
                    next_b_start = b_win[1]
                    break
            if not found_match:
                is_blocks_continuous = False
                # Need to find the next available block
                closest_dist = np.inf
                closest_next_b = None
                for b_name in self.blocks.keys():
                    b_win = self.blocks[b_name]
                    if b_win[0] > next_b_start:
                        if (b_win[0] - next_b_start) < closest_dist:
                            closest_dist = b_win[0] - next_b_start
                            closest_next_b = b_win[0]
                if closest_next_b is None:
                    # Can't find anymore succeeding blocks so break
                    next_b_start = len(self)
                    trials_missing.append(np.arange(last_b_end, len(self)))
                else:
                    next_b_start = closest_next_b
                    trials_missing.append(np.arange(last_b_end, next_b_start))
                # raise RuntimeError("Blocks do not cover all trials present! Last starting block attempted at trial {0}.".format(next_b_start))
        return is_blocks_continuous, np.hstack(trials_missing)

    def _verify_block_overlap(self):
        for b_name in self.block_names():
            curr_start = self.blocks[b_name][0]
            curr_stop = self.blocks[b_name][1]
            for b_name2 in self.block_names():
                if ( (self.blocks[b_name2][0] < curr_stop) and
                     (self.blocks[b_name2][1] > curr_stop) ):
                    raise RuntimeError("The two blocks: '{0}' and '{1}' were found overlapping each other!".format(b_name, b_name2))
                if ( (self.blocks[b_name2][0] < curr_start) and
                     (self.blocks[b_name2][1] > curr_start) ):
                    raise RuntimeError("The two blocks: '{0}' and '{1}' were found overlapping each other!".format(b_name, b_name2))
                if self.blocks[b_name2][0] == curr_start:
                    if self.blocks[b_name2][1] != curr_stop:
                        raise RuntimeError("The two blocks: '{0}' and '{1}' were found overlapping each other!".format(b_name, b_name2))
                if self.blocks[b_name2][1] == curr_stop:
                    if self.blocks[b_name2][0] != curr_start:
                        raise RuntimeError("The two blocks: '{0}' and '{1}' were found overlapping each other!".format(b_name, b_name2))
        return None

    def add_saccades(self, time_window, blocks=None, trial_sets=None):
        """ Adds saccade windows to session and by default nan's these out in
        the current 'eye' data. """
        # Get all eye data during initial fixation
        series_fix_data = {}
        # Hard coded names!
        series_names = ['horizontal_eye_position',
                        'vertical_eye_position',
                        'horizontal_eye_velocity',
                        'vertical_eye_velocity']
        for sn in series_names:
            series_fix_data[sn] = self.get_data_array(sn, time_window, blocks, trial_sets)

        # Find fixation eye offset for each trial, adjust its data, then nan saccades
        for t_ind in range(0, len(self)):
            try:
                # Adjust to target position at -100 ms
                offsets = eye_data_series.find_eye_offsets(
                                series_fix_data['horizontal_eye_position'][t_ind, :],
                                series_fix_data['vertical_eye_position'][t_ind, :],
                                series_fix_data['horizontal_eye_velocity'][t_ind, :],
                                series_fix_data['vertical_eye_velocity'][t_ind, :],
                                x_targ=self[t_ind].get_data('xpos')[-100],
                                y_targ=self[t_ind].get_data('ypos')[-100],
                                epsilon_eye=0.1, max_iter=10, return_saccades=False,
                                ind_cushion=20, acceleration_thresh=1, speed_thresh=30)

                for sn in series_names:
                    if sn == "horizontal_eye_position":
                        self._trial_lists['eye'][t_ind].data[sn] -= offsets[0]
                    elif sn == "vertical_eye_position":
                        self._trial_lists['eye'][t_ind].data[sn] -= offsets[1]
                    elif sn == "horizontal_eye_velocity":
                        self._trial_lists['eye'][t_ind].data[sn] -= offsets[2]
                    elif sn == "vertical_eye_velocity":
                        self._trial_lists['eye'][t_ind].data[sn] -= offsets[3]
                    else:
                        raise RuntimeError("Could not find data series name for offsets.")

                x_vel = self._trial_lists['eye'][t_ind].data['horizontal_eye_velocity']
                y_vel = self._trial_lists['eye'][t_ind].data['vertical_eye_velocity']
                saccade_windows, saccade_index = eye_data_series.find_saccade_windows(
                        x_vel, y_vel, ind_cushion=20, acceleration_thresh=1, speed_thresh=30)
                self._trial_lists['eye'][t_ind].saccade_windows = saccade_windows
                self._trial_lists['eye'][t_ind].saccade_index = saccade_index
                for sn in series_names:
                    # NaN all this eye data
                    self._trial_lists['eye'][t_ind].data[sn][saccade_index] = np.nan
            except:
                print(t_ind)
                raise
        return None

    def get_n_instructed_trials(self, event_offset=0):
        """ Gets the the number of PRECEDING learning trials for each trial
        that successfully reached the instruction event. """
        self.is_instructed = np.zeros(len(self), dtype='bool')
        self.n_instructed = np.zeros(len(self), dtype=np.int32)
        for block in self.blocks.keys():
            if "Learn" not in block:
                # Only care about learning blocks
                continue
            # This is a learning block
            n_learns = 0
            for t_ind in range(self.blocks[block][0], self.blocks[block][1]):
                if self.trial_sets['instruction'][t_ind]:
                    # This trial has a learning instruction trial name
                    if self[t_ind].events['instruction_onset'] is not None:
                        # This trial contains an instruction event
                        if self[t_ind].duration > self[t_ind].events['instruction_onset'] + event_offset:
                            # This trial is longer than the instruction event plus offset so it COUNTS!
                            self.is_instructed[t_ind] = True
                            # We count the number of PRECEDING learning events, so set before incrementing
                            self.n_instructed[t_ind] = n_learns
                            n_learns += 1
                        else:
                            # Numbe of learning trials has not changed
                            self.n_instructed[t_ind] = n_learns
                    else:
                        # Numbe of learning trials has not changed
                        self.n_instructed[t_ind] = n_learns
                else:
                    # Numbe of learning trials has not changed
                    self.n_instructed[t_ind] = n_learns
        return None

    def count_block_trial_names(self, block_name):
        """ Counts the number of each trial type within block block_name.
        """
        for t_ind in range(self.blocks[block_name][0], self.blocks[block_name][1]):
            try:
                count_dict[self._trial_lists['__main'][t_ind].name] += 1
            except KeyError:
                count_dict[self._trial_lists['__main'][t_ind].name] = 1
        return count_dict
