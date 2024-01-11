import numpy as np
import re
import warnings
import h5py
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

    def __init__(self, trial_data, session_name=None, data_type=None,
                 rotate=False, nan_saccades=True):
        """
        """
        Session.__init__(self, trial_data, session_name, data_type)
        self.block_info = {}
        self.block_name_to_learn_name = {}
        self.rotate = rotate
        self.sacc_and_err_trials = np.zeros(len(self), dtype='bool')
        self.rem_sacc_errs = False
        self.nan_saccades = nan_saccades
        self.verbose = True
        # Hard coded default sets and axes
        self.four_dir_trial_sets = ["learning", "anti_learning",
                                    "pursuit", "anti_pursuit"]

    def verify_blocks(self):
        is_blocks_continuous, orphan_trials = self._verify_block_continuity()
        self._verify_block_overlap()
        return orphan_trials

    def assign_block_names(self, learn_sn="Learn", fix_sn="FixTune",
            stab_sn="StabTune", stand_sn="StandTune", randvp_sn="RandVP"):
        self.blocks_found = []
        self.set_learning_block(search_name=learn_sn)
        self.parse_learning_directions(search_name=learn_sn)
        self.set_washout_block()
        self.set_fixation_tuning_blocks(search_name=fix_sn)
        self.set_stab_tuning_blocks(search_name=stab_sn)
        self.set_stand_tuning_blocks(search_name=stand_sn)
        self.set_pursuit_baseline_blocks(search_name=f"{str(self.directions['pursuit'])}Baseline")
        self.set_randvp_tuning_blocks(search_name=randvp_sn)
        for block in self.blocks_found:
            self.count_block_trial_names(block)
        return None

    def set_learning_block(self, search_name="Learn"):
        """ Parse through the block names and find the longest learning block
        to designate the "learning" direction and block.
        """
        n_max_learn = 0
        b_max_learn = None
        for block in self.blocks.keys():
            if self.blocks[block] is None:
                continue
            if search_name in block:
                block_len = self.blocks[block][1] - self.blocks[block][0]
                if block_len > n_max_learn:
                    n_max_learn = block_len
                    b_max_learn = block
        if b_max_learn is not None:
            self.blocks['Learning'] = self.blocks[b_max_learn]
            self.learning_trial_name = b_max_learn
            self.blocks_found.append('Learning')
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
        learn_dir = learn_dir.split("_")[0] # Remove any trailing numerical labels
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
            if self.blocks[block] is None:
                continue
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
            self.blocks_found.append('Washout')
        return None

    def set_fixation_tuning_blocks(self, search_name="FixTune"):
        """ Finds the fixation tuning blocks relative to learning/washout. """
        if self.blocks['Learning'] is None:
            raise ValueError("Cannot parse washout block with no learning block assigned!")
        t_pre_start = -np.inf
        t_post_start = np.inf
        t_wash_start = np.inf
        search_blocks = ["FixTunePre", "FixTunePost", "FixTuneWash"]
        new_block_names = {x: None for x in search_blocks}
        for block in self.blocks.keys():
            if self.blocks[block] is None:
                continue
            # Do split on num for default number counting of multiple blocks
            if search_name in block:
                # Found a fixation trial, now find which one
                if ( (self.blocks[block][1] <= self.blocks['Learning'][0]) and
                     (self.blocks[block][0] >= t_pre_start) ):
                    # Tuning precedes learning and is latest found so far
                    new_block_names['FixTunePre'] = self.blocks[block]
                    t_pre_start = self.blocks[block][0]
                    self.blocks_found.append('FixTunePre')
                elif ( (self.blocks[block][0] >= self.blocks['Learning'][1]) and
                       (self.blocks[block][0] <= t_post_start) ):
                    # Tuning follows learning and is earliest found so far
                    new_block_names['FixTunePost'] = self.blocks[block]
                    t_post_start = self.blocks[block][0]
                    self.blocks_found.append('FixTunePost')
                if self.blocks['Washout'] is not None:
                    if ( (self.blocks[block][0] >= self.blocks['Washout'][1]) and
                         (self.blocks[block][0] <= t_wash_start) ):
                        # Tuning follows washout and is earliest found so far
                        new_block_names['FixTuneWash'] = self.blocks[block]
                        t_wash_start = self.blocks[block][0]
                        self.blocks_found.append('FixTuneWash')
        self.blocks.update(new_block_names)
        return None

    def set_stab_tuning_blocks(self, search_name="StabTune"):
        """ Finds the fixation tuning blocks relative to learning/washout. """
        if self.blocks['Learning'] is None:
            raise ValueError("Cannot parse washout block with no learning block assigned!")
        t_pre_start = -np.inf
        t_post_start = np.inf
        t_wash_start = np.inf
        search_blocks = ["StabTunePre", "StabTunePost", "StabTuneWash"]
        new_block_names = {x: None for x in search_blocks}
        for block in self.blocks.keys():
            if self.blocks[block] is None:
                continue
            # Do split on num for default number counting of multiple blocks
            if search_name in block:
                # Found a fixation trial, now find which one
                if ( (self.blocks[block][1] <= self.blocks['Learning'][0]) and
                     (self.blocks[block][0] >= t_pre_start) ):
                    # Tuning precedes learning and is latest found so far
                    new_block_names['StabTunePre'] = self.blocks[block]
                    t_pre_start = self.blocks[block][0]
                    self.blocks_found.append('StabTunePre')
                elif ( (self.blocks[block][0] >= self.blocks['Learning'][1]) and
                       (self.blocks[block][0] <= t_post_start) ):
                    # Tuning follows learning and is earliest found so far
                    new_block_names['StabTunePost'] = self.blocks[block]
                    t_post_start = self.blocks[block][0]
                    self.blocks_found.append('StabTunePost')
                if self.blocks['Washout'] is not None:
                    if ( (self.blocks[block][0] >= self.blocks['Washout'][1]) and
                         (self.blocks[block][0] <= t_wash_start) ):
                        # Tuning follows washout and is earliest found so far
                        new_block_names['StabTuneWash'] = self.blocks[block]
                        t_wash_start = self.blocks[block][0]
                        self.blocks_found.append('StabTuneWash')
        self.blocks.update(new_block_names)
        return None

    def set_stand_tuning_blocks(self, search_name="StandTune"):
        """ Finds the fixation tuning blocks relative to learning/washout. """
        if self.blocks['Learning'] is None:
            raise ValueError("Cannot parse washout block with no learning block assigned!")
        t_pre_start = -np.inf
        t_post_start = np.inf
        t_wash_start = np.inf
        search_blocks = ["StandTunePre", "StandTunePost", "StandTuneWash"]
        new_block_names = {x: None for x in search_blocks}
        for block in self.blocks.keys():
            if self.blocks[block] is None:
                continue
            # Do split on num for default number counting of multiple blocks
            if search_name in block:
                # Found a fixation trial, now find which one
                if ( (self.blocks[block][1] <= self.blocks['Learning'][0]) and
                     (self.blocks[block][0] >= t_pre_start) ):
                    # Tuning precedes learning and is latest found so far
                    new_block_names['StandTunePre'] = self.blocks[block]
                    t_pre_start = self.blocks[block][0]
                    self.blocks_found.append('StandTunePre')
                elif ( (self.blocks[block][0] >= self.blocks['Learning'][1]) and
                       (self.blocks[block][0] <= t_post_start) ):
                    # Tuning follows learning and is earliest found so far
                    new_block_names['StandTunePost'] = self.blocks[block]
                    t_post_start = self.blocks[block][0]
                    self.blocks_found.append('StandTunePost')
                if self.blocks['Washout'] is not None:
                    if ( (self.blocks[block][0] >= self.blocks['Washout'][1]) and
                         (self.blocks[block][0] <= t_wash_start) ):
                        # Tuning follows washout and is earliest found so far
                        new_block_names['StandTuneWash'] = self.blocks[block]
                        t_wash_start = self.blocks[block][0]
                        self.blocks_found.append('StandTuneWash')
        self.blocks.update(new_block_names)
        return None

    def set_pursuit_baseline_blocks(self, search_name):
        """ Finds the pursuit axis only tuning/baseline blocks relative to learning/washout. """
        if self.blocks['Learning'] is None:
            raise ValueError("Cannot parse washout block with no learning block assigned!")
        t_pre_start = -np.inf
        t_post_start = np.inf
        t_wash_start = np.inf
        search_blocks = ["BaselinePre", "BaselinePost", "BaselineWash"]
        new_block_names = {x: None for x in search_blocks}
        for block in self.blocks.keys():
            if self.blocks[block] is None:
                continue
            # Do split on num for default number counting of multiple blocks
            if search_name in block:
                # Found a Baseline block, now find which one
                if ( (self.blocks[block][1] <= self.blocks['Learning'][0]) and
                     (self.blocks[block][0] >= t_pre_start) ):
                    # 0Baselin precedes learning and is latest found so far
                    new_block_names['BaselinePre'] = self.blocks[block]
                    t_pre_start = self.blocks[block][0]
                    self.blocks_found.append('BaselinePre')
                elif ( (self.blocks[block][0] >= self.blocks['Learning'][1]) and
                       (self.blocks[block][0] <= t_post_start) ):
                    # Tuning follows learning and is earliest found so far
                    new_block_names['BaselinePost'] = self.blocks[block]
                    t_post_start = self.blocks[block][0]
                    self.blocks_found.append('BaselinePost')
                if self.blocks['Washout'] is not None:
                    if ( (self.blocks[block][0] >= self.blocks['Washout'][1]) and
                         (self.blocks[block][0] <= t_wash_start) ):
                        # Tuning follows washout and is earliest found so far
                        new_block_names['BaselineWash'] = self.blocks[block]
                        t_wash_start = self.blocks[block][0]
                        self.blocks_found.append('BaselineWash')
        self.blocks.update(new_block_names)
        return None

    def set_randvp_tuning_blocks(self, search_name="RandVP"):
        """ Finds the fixation tuning blocks relative to learning/washout. """
        if self.blocks['Learning'] is None:
            raise ValueError("Cannot parse washout block with no learning block assigned!")
        t_pre_start = -np.inf
        t_post_start = np.inf
        t_wash_start = np.inf
        search_blocks = ["RandVPTunePre", "RandVPTunePost", "RandVPTuneWash"]
        new_block_names = {x: None for x in search_blocks}
        for block in self.blocks.keys():
            if self.blocks[block] is None:
                continue
            # Do split on num for default number counting of multiple blocks
            if search_name in block:
                # Found a fixation trial, now find which one
                if ( (self.blocks[block][1] <= self.blocks['Learning'][0]) and
                     (self.blocks[block][0] >= t_pre_start) ):
                    # Tuning precedes learning and is latest found so far
                    new_block_names['RandVPTunePre'] = self.blocks[block]
                    t_pre_start = self.blocks[block][0]
                    self.blocks_found.append('RandVPTunePre')
                elif ( (self.blocks[block][0] >= self.blocks['Learning'][1]) and
                       (self.blocks[block][0] <= t_post_start) ):
                    # Tuning follows learning and is earliest found so far
                    new_block_names['RandVPTunePost'] = self.blocks[block]
                    t_post_start = self.blocks[block][0]
                    self.blocks_found.append('RandVPTunePost')
                if self.blocks['Washout'] is not None:
                    if ( (self.blocks[block][0] >= self.blocks['Washout'][1]) and
                         (self.blocks[block][0] <= t_wash_start) ):
                        # Tuning follows washout and is earliest found so far
                        new_block_names['RandVPTuneWash'] = self.blocks[block]
                        t_wash_start = self.blocks[block][0]
                        self.blocks_found.append('RandVPTuneWash')
        self.blocks.update(new_block_names)
        return None

    def _parse_fixation_trials(self, trial_name, trial_index):
        if trial_name == 'd140fix':
            t_dir = [0, None]
        elif trial_name == 'd1010fix':
            t_dir = [0, 90]
        elif trial_name == 'd014fix':
            t_dir = [None, 90]
        elif trial_name == 'd-1010fix':
            t_dir = [180, 90]
        elif trial_name == 'd-140fix':
            t_dir = [180, None]
        elif trial_name == 'd-10-10fix':
            t_dir = [180, 270]
        elif trial_name == 'd0-14fix':
            t_dir = [None, 270]
        elif trial_name == 'd10-10fix':
            t_dir = [0, 270]
        elif trial_name == 'd00fix':
            t_dir = [None, None]
        else:
            raise ValueError("Unrecognized trial name {0}.".format(trial_name))
        fix_set_name = "fix"
        if t_dir[0] is not None:
            for direction in self.directions:
                if self.directions[direction] == t_dir[0]:
                    name_1 = "_" + direction
                    break
        else:
            name_1 = ""
        if t_dir[1] is not None:
            for direction in self.directions:
                if self.directions[direction] == t_dir[1]:
                    name_2 = "_" + direction
                    break
        else:
            name_2 = ""
        # Need to ensure that pursuit direction name comes first
        if "pursuit" in name_2:
            fix_set_name = fix_set_name + name_2 + name_1
        else:
            fix_set_name = fix_set_name + name_1 + name_2
        if fix_set_name == "fix":
            # Was fixation only trial
            fix_set_name = "fix_fix_center"
        self.trial_sets[fix_set_name][trial_index] = True
        return

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
        # Need to initialize all these fixation sets here before call to parse
        # fixation trials which just stops this from getting to messy in loop
        self.trial_sets['fix_pursuit'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['fix_pursuit_learning'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['fix_learning'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['fix_anti_pursuit_learning'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['fix_anti_pursuit'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['fix_anti_pursuit_anti_learning'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['fix_anti_learning'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['fix_pursuit_anti_learning'] = np.zeros(len(self), dtype='bool')
        self.trial_sets['fix_fix_center'] = np.zeros(len(self), dtype='bool')
        for ind, st in enumerate(self._session_trial_data):
            if "fix" in st['name']:
                self._parse_fixation_trials(st['name'], ind)
            elif st['name'] == str(self.directions['pursuit']):
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
        orphan_trials = []
        next_b_start = 0
        last_b_end = 0
        while next_b_start < len(self):
            found_match = False
            for b_name in self.blocks.keys():
                if self.blocks[b_name] is None:
                    continue
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
                    if self.blocks[b_name] is None:
                        continue
                    b_win = self.blocks[b_name]
                    if b_win[0] > next_b_start:
                        if (b_win[0] - next_b_start) < closest_dist:
                            closest_dist = b_win[0] - next_b_start
                            closest_next_b = b_win[0]
                if closest_next_b is None:
                    # Can't find anymore succeeding blocks so break
                    next_b_start = len(self)
                    orphan_trials.append(np.arange(last_b_end, len(self)))
                else:
                    next_b_start = closest_next_b
                    orphan_trials.append(np.arange(last_b_end, next_b_start))
                # raise RuntimeError("Blocks do not cover all trials present! Last starting block attempted at trial {0}.".format(next_b_start))
        if len(orphan_trials) > 0:
            orphan_trials = np.unique(np.hstack(orphan_trials))
        return is_blocks_continuous, orphan_trials

    def _verify_block_overlap(self):
        for b_name in self.get_block_names():
            if self.blocks[b_name] is None:
                continue
            curr_start = self.blocks[b_name][0]
            curr_stop = self.blocks[b_name][1]
            for b_name2 in self.get_block_names():
                if self.blocks[b_name2] is None:
                    continue
                if ( (self.blocks[b_name2][0] < curr_stop) and
                     (self.blocks[b_name2][1] > curr_stop) ):
                     print("Attempting to repair block overlap between {0} and {1}.".format(b_name, b_name2))
                     self._fix_block_overlap(b_name, b_name2)
                     self._verify_block_overlap()
                     print("Repair successful!")
                     return
                    # raise RuntimeError("The two blocks: '{0}' and '{1}' were found overlapping each other!".format(b_name, b_name2))
                if ( (self.blocks[b_name2][0] < curr_start) and
                     (self.blocks[b_name2][1] > curr_start) ):
                     print("Attempting to repair block overlap between {0} and {1}.".format(b_name, b_name2))
                     self._fix_block_overlap(b_name, b_name2)
                     self._verify_block_overlap()
                     print("Repair successful!")
                     return
                    # raise RuntimeError("The two blocks: '{0}' and '{1}' were found overlapping each other!".format(b_name, b_name2))
                if self.blocks[b_name2][0] == curr_start:
                    if self.blocks[b_name2][1] != curr_stop:
                        print("Attempting to repair block overlap between {0} and {1}.".format(b_name, b_name2))
                        self._fix_block_overlap(b_name, b_name2)
                        self._verify_block_overlap()
                        print("Repair successful!")
                        return
                        # raise RuntimeError("The two blocks: '{0}' and '{1}' were found overlapping each other!".format(b_name, b_name2))
                if self.blocks[b_name2][1] == curr_stop:
                    if self.blocks[b_name2][0] != curr_start:
                        print("Attempting to repair block overlap between {0} and {1}.".format(b_name, b_name2))
                        self._fix_block_overlap(b_name, b_name2)
                        self._verify_block_overlap()
                        print("Repair successful!")
                        return
                        # raise RuntimeError("The two blocks: '{0}' and '{1}' were found overlapping each other!".format(b_name, b_name2))
        return None

    def _fix_block_overlap(self, check_block_1, check_block_2):
        if self.blocks[check_block_1][0] < self.blocks[check_block_2][0]:
            # Block 1 starts first
            b_name1 = check_block_1
            b_name2 = check_block_2
        elif self.blocks[check_block_1][0] > self.blocks[check_block_2][0]:
            # Block 2 starts first
            b_name1 = check_block_2
            b_name2 = check_block_1
        else:
            # Blocks start at same trial
            raise ValueError(f"Resolving block overlaps that start on the same trial not implemented! Tried blocks {check_block_1} and {check_block_2}")
        t_names1 = set()
        t_names2 = set()
        for t_ind in range(self.blocks[b_name1][0], self.blocks[b_name1][1]):
            t_names1.add(self[t_ind].name)
        for t_ind in range(self.blocks[b_name2][0], self.blocks[b_name2][1]):
            t_names2.add(self[t_ind].name)
        unique_names1 = t_names1 - t_names2

        for t_ind in reversed(range(self.blocks[b_name1][0], self.blocks[b_name2][0]+1)):
            if self[t_ind].name in unique_names1:
                self.blocks[b_name1][1] = t_ind + 1
                self.blocks[b_name2][0] = t_ind + 1
                break
        return None

    def assign_orphan_trials(self, orphan_trials):
        """ This will not skip trials and just add to nearby blocks so orphan
        so orphan_trials needs to include all orphans and be ordered. """
        if len(orphan_trials) > 1:
            order_t = np.argsort(orphan_trials)
            orphan_trials = orphan_trials[order_t]
        ot_ind = 0
        found_match = False
        n_orphans_assigned = 0
        while ot_ind < len(orphan_trials):
            found_match = False
            for block in self.blocks_found:
                if orphan_trials[ot_ind] < self.blocks['Learning'][0]:
                    # Missing before learning block
                    if orphan_trials[ot_ind] == self.blocks[block][1]:
                        # Prefer to add to end of immediately preceding block
                        if self[orphan_trials[ot_ind]].name in self.block_info[block]:
                            self.blocks[block][1] += 1
                            ot_ind += 1
                            n_orphans_assigned += 1
                            found_match = True
                            while ot_ind < len(orphan_trials):
                                tm_diff = orphan_trials[ot_ind] - orphan_trials[ot_ind - 1]
                                if tm_diff != 1:
                                    break
                                self.blocks[block][1] += 1
                                ot_ind += 1
                                n_orphans_assigned += 1
                            break # for loop over 'block'
                elif orphan_trials[ot_ind] >= self.blocks['Learning'][1]:
                    # Missing after learning block
                    if orphan_trials[ot_ind] == self.blocks[block][1]:
                        # Prefer to add to end of immediately preceding block?
                        if self[orphan_trials[ot_ind]].name in self.block_info[block]:
                            self.blocks[block][1] += 1
                            ot_ind += 1
                            n_orphans_assigned += 1
                            found_match = True
                            while ot_ind < len(orphan_trials):
                                tm_diff = orphan_trials[ot_ind] - orphan_trials[ot_ind - 1]
                                if tm_diff != 1:
                                    break
                                self.blocks[block][1] += 1
                                ot_ind += 1
                                n_orphans_assigned += 1
                            break # for loop over 'block'
                else:
                    # Missing without a learning block present...
                    if orphan_trials[ot_ind] == self.blocks[block][1]:
                        # Prefer to add to end of immediately preceding block?
                        if self[orphan_trials[ot_ind]].name in self.block_info[block]:
                            self.blocks[block][1] += 1
                            ot_ind += 1
                            n_orphans_assigned += 1
                            found_match = True
                            while ot_ind < len(orphan_trials):
                                tm_diff = orphan_trials[ot_ind] - orphan_trials[ot_ind - 1]
                                if tm_diff != 1:
                                    break
                                self.blocks[block][1] += 1
                                ot_ind += 1
                                n_orphans_assigned += 1
                            break # for loop over 'block'
            if not found_match:
                ot_ind += 1
        return n_orphans_assigned

    def add_saccades(self, time_window, blocks=None, trial_sets=None,
                        ind_cushion=30):
        """ Adds saccade windows to session and by default nan's these out in
        the current 'eye' data. """
        # Get all eye data during initial fixation
        series_fix_data = {}
        # Hard coded names!
        series_names = ['horizontal_eye_position',
                        'vertical_eye_position',
                        'horizontal_eye_velocity',
                        'vertical_eye_velocity']
        slip_names = ['horizontal_retinal_slip',
                      'vertical_retinal_slip']
        for sn in series_names:
            # t_inds should be the same for each data series
            series_fix_data[sn], t_inds = self.get_data_array(sn, time_window,
                                            blocks, trial_sets, return_inds=True)

        # Find fixation eye offset for each trial, adjust its data, then nan saccades
        self.saccade_ind_cushion = ind_cushion
        for ind, t_ind in enumerate(t_inds):
            try:
                # Adjust to target position at -100 ms
                offsets = eye_data_series.find_eye_offsets(
                                series_fix_data['horizontal_eye_position'][ind, :],
                                series_fix_data['vertical_eye_position'][ind, :],
                                series_fix_data['horizontal_eye_velocity'][ind, :],
                                series_fix_data['vertical_eye_velocity'][ind, :],
                                x_targ=self[t_ind].get_data('xpos')[-100],
                                y_targ=self[t_ind].get_data('ypos')[-100],
                                epsilon_eye=0.1, max_iter=10, return_saccades=False,
                                ind_cushion=ind_cushion, acceleration_thresh=1, speed_thresh=30)

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
                        x_vel, y_vel, ind_cushion=ind_cushion, acceleration_thresh=1, speed_thresh=30)
                self._trial_lists['eye'][t_ind].saccade_windows = saccade_windows
                self._trial_lists['eye'][t_ind].saccade_index = saccade_index
                if self.nan_saccades:
                    for sn in series_names:
                        # NaN all this eye data
                        self._trial_lists['eye'][t_ind].data[sn][saccade_index] = np.nan
                    # Check if slip is added then nan saccades if so
                    for sn in slip_names:
                        try:
                            self._trial_lists['eye'][t_ind].data[sn][saccade_index] = np.nan
                        except KeyError:
                            # Should mean slip isn't found
                            break
            except:
                print(ind, t_ind)
                raise
        return None

    def get_n_instructed_trials(self, event_offset=0):
        """ Gets the the number of PRECEDING learning trials for each trial
        that successfully reached the instruction event. """
        self.is_instructed = np.zeros(len(self), dtype='bool')
        self.n_instructed = np.zeros(len(self), dtype=np.int32)
        for block in self.blocks_found:
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
                            # Number of learning trials has not changed
                            self.n_instructed[t_ind] = n_learns
                    else:
                        # Number of learning trials has not changed
                        self.n_instructed[t_ind] = n_learns
                else:
                    # Number of learning trials has not changed
                    self.n_instructed[t_ind] = n_learns
        return None

    def count_block_trial_names(self, block_name):
        """ Counts the number of each trial type within block block_name.
        """
        if self.blocks[block_name] is None:
            return None
        else:
            self.block_info[block_name] = {}
        for t_ind in range(self.blocks[block_name][0], self.blocks[block_name][1]):
            try:
                self.block_info[block_name][self._trial_lists['__main'][t_ind].name] += 1
            except KeyError:
                self.block_info[block_name][self._trial_lists['__main'][t_ind].name] = 1
        return None

    def set_baseline_averages(self, time_window, rotate=True):
        """ Adds the position and velocity trials as baseline from the
        info provided over the default 4 axes in trial_sets.
        Will also check for the presence of neurons and set the baseline firing
        rate for them to.

        NOTE: Decided to put neuron tuning in here to make sure they are from
        the identical trials as behavior instead of trying to match them
        after the fact.
        """
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        # Set rotation according to whether baseline is rotated
        if self.rotate != rotate:
            warnings.warn("Input rotation value {0} does not match existing value {1}. Resetting rotation to {2} so previous analyses could be invalid!".format(rotate, self.rotate, rotate))
            self.rotate = rotate
        if self.rotate:
            self.trial_set_base_axis = {'learning': 0,
                                       'anti_learning': 0,
                                       'pursuit': 1,
                                       'anti_pursuit': 1,
                                       'instruction': 1
                                        }
        else:
            self.trial_set_base_axis = {}
            for t_set in self.directions:
                if self.directions[t_set] == 0:
                    self.trial_set_base_axis[t_set] = 1
                elif self.directions[t_set] == 90:
                    self.trial_set_base_axis[t_set] = 0
                elif self.directions[t_set] == 180:
                    self.trial_set_base_axis[t_set] = 1
                elif self.directions[t_set] == 270:
                    self.trial_set_base_axis[t_set] = 0
                else:
                    raise RuntimeError("Could not find base set axes for direction '{0}'".format(self.directions[t_set]))
            self.trial_set_base_axis['instruction'] = self.trial_set_base_axis['pursuit']

        # Setup dictionary for storing tuning data
        tuning_blocks = ["StandTunePre", "StabTunePre", "BaselinePre"]
        data_types = ["eye position", "eye velocity"]
        self.baseline_tuning = {}
        for block in tuning_blocks:
            # Check if block is present
            if self.blocks[block] is None:
                self.baseline_tuning[block] = None
                continue
            self.baseline_tuning[block] = {}
            self.baseline_tuning[block]['instruction'] = {}
            for curr_set in self.four_dir_trial_sets:
                self.baseline_tuning[block][curr_set] = {}
                for data_t in data_types:
                    x, y = self.get_xy_traces(data_t, time_window,
                                    blocks=block, trial_sets=curr_set,
                                    return_inds=False)
                    if x.shape[0] == 0:
                        # Found no matching data
                        self.baseline_tuning[block][curr_set][data_t] = None
                        continue
                    self.baseline_tuning[block][curr_set][data_t] = np.zeros((2, x.shape[1]))
                    with warnings.catch_warnings():
                        self.baseline_tuning[block][curr_set][data_t][0, :] = np.nanmean(x, axis=0)
                        self.baseline_tuning[block][curr_set][data_t][1, :] = np.nanmean(y, axis=0)
                    if curr_set == "pursuit":
                        # Add the instructed set baseline to be same as pursuit!
                        self.baseline_tuning[block]['instruction'][data_t] = self.baseline_tuning[block]['pursuit'][data_t]
                    if hasattr(self, "neuron_info"):
                        for n_name in self.neuron_info['neuron_names']:
                            n_data_t = self.neuron_info[n_name].use_series
                            self.baseline_tuning[block][curr_set][n_data_t] = self.neuron_info[n_name].get_mean_firing_trace(
                                                        time_window, blocks=block,
                                                        trial_sets=curr_set,
                                                        return_inds=False)

        # Save the time window used for future reference
        self.baseline_time_window = time_window
        return None

    def is_stabilized(self, block, trial_set=None):
        not_stab = set()
        if trial_set is None:
            check_trials = np.ones(len(self), dtype='bool')
        elif isinstance(trial_set, str):
            check_trials = self.trial_sets[trial_set]
        else:
            check_trials = trial_set

        for t_ind in range(self.blocks[block][0], self.blocks[block][1]):
            if check_trials[t_ind]:
                if not self[t_ind].used_stab:
                    not_stab.add(self[t_ind].name)
        if len(not_stab) > 0:
            print("Trials: {0} are not stabilized during block {1}.".format(not_stab, block))
            return False
        else:
            return True

    def set_sacc_and_err_trials(self, time_window, max_sacc_amp=np.inf,
                            max_pos_err=np.inf, blocks=None, trial_sets=None):
        """Time window will be deleted based on current alignment for the trials
        input in blocks and trial sets. """
        warnings.filterwarnings(action="ignore", category=RuntimeWarning)
        # Get all eye data during initial fixation
        series_pos_data = {}
        # Hard coded names!
        series_names = ['horizontal_eye_position',
                        'vertical_eye_position',
                        'horizontal_target_position',
                        'vertical_target_position']
        for sn in series_names:
            # t_inds should be the same for each data series
            series_pos_data[sn], t_inds = self.get_data_array(sn, time_window,
                                            blocks, trial_sets, return_inds=True)
            if len(series_pos_data[sn]) == 0:
                # Specified blocks/trials do not return any values
                return []
        t_inds_to_set = []
        for ind, t_ind in enumerate(t_inds):
            eye_x = series_pos_data['horizontal_eye_position'][ind, :]
            eye_y = series_pos_data['vertical_eye_position'][ind, :]
            targ_x = series_pos_data['horizontal_target_position'][ind, :]
            targ_y = series_pos_data['vertical_target_position'][ind, :]
            sacc_amps = eye_data_series.sacc_amp_nan(eye_x, eye_y)
            max_sacc = 0 if len(sacc_amps) == 0 else np.amax(sacc_amps)
            error_x = (targ_x - eye_x)
            error_y = (targ_y - eye_y)
            error_tot = np.sqrt((error_x ** 2) + (error_y ** 2))
            with warnings.catch_warnings():
                max_err = np.nanmax(error_tot)
                max_err = 0 if np.all(np.isnan(max_err)) else max_err
            if ( (max_sacc > max_sacc_amp) or (max_err > max_pos_err) ):
                t_inds_to_set.append(t_ind)

        if len(t_inds_to_set) == 0:
            # No bad trials found to set
            return np.array([], dtype=np.int64)
        t_inds_to_set = np.array(t_inds_to_set)
        self.sacc_and_err_trials[t_inds_to_set] = True
        return t_inds_to_set

    def add_retinal_slip_data(self, target_name="target0"):
        """ Iterates over all trials and computes target velocity - eye velocity
        and adds it to the underlying trial object data
        """
        for t in range(0, len(self)):
            # Get each apparatus trial object
            app_obj = self._trial_lists[target_name][t]
            beh_obj = self._trial_lists['eye'][t]
            x_slip = (app_obj['data']['horizontal_target_velocity'] -
                        beh_obj['data']['horizontal_eye_velocity'])
            y_slip = (app_obj['data']['vertical_target_velocity'] -
                        beh_obj['data']['vertical_eye_velocity'])
            beh_obj.add_data_series('horizontal_retinal_slip', x_slip)
            beh_obj.add_data_series('vertical_retinal_slip', y_slip)
        # Also need to update Session on new series
        self.add_data_series("eye", "horizontal_retinal_slip")
        self.add_data_series("eye", "vertical_retinal_slip")

    def get_xy_traces(self , series_name, time_window, blocks=None,
                     trial_sets=None, return_inds=False):
        """ Simultaneously gets two traces for each dimension of either the eye
        or target data only which sould reduce loops compared to
        Session.get_data_array(). Output data are rotated as per the properties
        of the LDP_Session.rotate.
        """
        series_name = series_name.lower()
        # Parse data name to type and series xy
        if (("eye" in series_name) and not ("slip" in series_name)):
            if "position" in series_name:
                x_name = "horizontal_eye_position"
                y_name = "vertical_eye_position"
            elif "velocity" in series_name:
                x_name = "horizontal_eye_velocity"
                y_name = "vertical_eye_velocity"
            else:
                raise InputError("Data name for 'eye' must also include either 'position' or 'velocity' to specify data type.")
            data_name = "eye"
        elif "target" in series_name:
            if "position" in series_name:
                x_name = "horizontal_target_position"
                y_name = "vertical_target_position"
            elif "velocity" in series_name:
                if "comm" in series_name:
                    x_name = "xvel_comm"
                    y_name = "yvel_comm"
                else:
                    x_name = "horizontal_target_velocity"
                    y_name = "vertical_target_velocity"
            else:
                raise InputError("Data name for 'eye' must also include either 'position' or 'velocity' to specify data type.")
            data_name = "target0"
        elif "slip" in series_name:
            x_name = "horizontal_retinal_slip"
            y_name = "vertical_retinal_slip"
            data_name = "eye"
        else:
            raise InputError("Data name must include either 'eye', 'target', or 'slip' to specify data type.")

        data_out_x = []
        data_out_y = []
        t_inds_out = []
        t_inds = self._parse_blocks_trial_sets(blocks, trial_sets)
        for t in t_inds:
            if not self._session_trial_data[t]['incl_align']:
                # Trial is not aligned with others due to missing event
                continue
            trial_obj = self._trial_lists[data_name][t]
            self._set_t_win(t, time_window)
            valid_tinds = self._session_trial_data[t]['curr_t_win']['valid_tinds']
            out_inds = self._session_trial_data[t]['curr_t_win']['out_inds']

            t_data_x = np.full(out_inds.shape[0], np.nan)
            t_data_y = np.full(out_inds.shape[0], np.nan)
            t_data_x[out_inds] = trial_obj['data'][x_name][valid_tinds]
            t_data_y[out_inds] = trial_obj['data'][y_name][valid_tinds]
            if self.rotate:
                rot_data = self.rotation_matrix @ np.vstack((t_data_x, t_data_y))
                t_data_x = rot_data[0, :]
                t_data_y = rot_data[1, :]
            data_out_x.append(t_data_x)
            data_out_y.append(t_data_y)
            t_inds_out.append(t)

        if return_inds:
            if len(data_out_x) > 0:
                # We found data to concatenate
                return np.vstack(data_out_x), np.vstack(data_out_y), np.hstack(t_inds_out)
            else:
                return np.zeros((0, time_window[1]-time_window[0])), np.zeros((0, time_window[1]-time_window[0])), np.array([], dtype=np.int32)
        else:
            if len(data_out_x) > 0:
                # We found data to concatenate
                return np.vstack(data_out_x), np.vstack(data_out_y)
            else:
                return np.zeros((0, time_window[1]-time_window[0])), np.zeros((0, time_window[1]-time_window[0]))

    def get_mean_xy_traces(self, series_name, time_window, blocks=None,
                            trial_sets=None, rescale=False):
        """ Calls get_xy_traces above and takes the mean over rows of the output. """
        warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
        x, y = self.get_xy_traces(series_name, time_window, blocks=blocks,
                         trial_sets=trial_sets, return_inds=False)
        if x.shape[0] == 0:
            # Found no matching data
            return x, y
        with warnings.catch_warnings():
            x = np.nanmean(x, axis=0)
            y = np.nanmean(y, axis=0)
        return x, y

    def join_neurons(self):
        """
        """
        if ( ("neuron_names" not in self.neuron_info) or
             (len(self.neuron_info['neuron_names']) == 0) ):
             # Neurons not set for this session
             raise ValueError("Session does not have any Neuron Metadata to join! Add neurons to neuron_info attribute.")
        # Add the neuron objects to this ldp session
        for n_name in self.neuron_info['neuron_names']:
            self.neuron_info[n_name].join_session(self)

    """ SOME FUNCTIONS OVERWRITTING THE SESSION OBJECT FUNCTIONS """
    def _parse_blocks_trial_sets(self, blocks=None, trial_sets=None):
        t_inds = super()._parse_blocks_trial_sets(blocks, trial_sets)
        # Filter this output by sacc error trials
        if ( (self.rem_sacc_errs) and (len(t_inds) > 0) ):
            t_sacc_err = self.sacc_and_err_trials[t_inds]
            t_inds = t_inds[~t_sacc_err]
        return t_inds

    def delete_trials(self, indices):
        """ Calls parent delete function, then must update our block info
        according to new blocks. """
        super().delete_trials(indices)
        if len(self.block_info) > 0:
            for block in self.blocks_found:
                self.count_block_trial_names(block)
        # Parse indices
        if type(indices) == np.ndarray:
            if indices.dtype == 'bool':
                indices = np.int64(np.nonzero(indices)[0])
            else:
                # Only unique, sorted, as integers
                indices = np.int64(indices)
        elif type(indices) == list:
            indices = np.array(indices, dtype=np.int64)
        else:
            if type(indices) != int:
                raise ValueError("Indices must be a numpy array, python list, or integer type.")
        indices = np.array(indices)
        indices = np.unique(indices)
        self.sacc_and_err_trials = np.delete(self.sacc_and_err_trials, indices)
        return None
    
    def save_ldp_session(self, filename):
        """ Saves data for this session to H5. This defines a format for saving at least most of the essential data to H5. There is an
        implementation for loading this format back into a python dictionary below but nothing for loading the file back into the
        LDPSession class format.
        """
        # Write some global root info
        with h5py.File(filename, 'w') as file:
            file.attrs['credits'] = ("These experiments were designed, and data collected, processed, and analyzed by Nathan J. Hall while "
                                    "working in the laboratory of Stephen G. Lisberger at Duke University circa 2017-2023")
            # Put the filename in
            file.attrs['filename'] = self.fname
            # Put some single value properties that apply to the entire session
            file.attrs['is_weird_Yan'] = self.is_weird_Yan
            file.attrs['is_weird_Yoda'] = self.is_weird_Yoda
            file.attrs['learning_trial_name'] = self.learning_trial_name
            file.attrs['meta_dict_name'] = self.meta_dict_name
            file.attrs['nan_saccades'] = self.nan_saccades
            file.attrs['rem_sacc_errs'] = self.rem_sacc_errs
            file.attrs['rotate'] = self.rotate
            file.attrs['saccade_ind_cushion'] = self.saccade_ind_cushion
            file.attrs['session_name'] = self.session_name
            file.attrs['verbose'] = self.verbose
            
            # This is a data array so save as array
            file.create_dataset("rotation_matrix", data=self.rotation_matrix)        
            
            # Start by adding a description of this group to the root
            file.attrs['desc_base_and_tune_blocks'] = ("The base_and_tune_blocks give the blocknames that correspond with " 
                                                    "pre/post tuning, washout, etc. This is for convenience.")
            base_and_tune_blocks = file.create_group('base_and_tune_blocks')
            for key, value in self.base_and_tune_blocks.items():
                if value is None:
                    value = "None"
                base_and_tune_blocks.attrs[key] = value
            
            # Start by adding a description of this group to the root
            file.attrs['desc_block_info'] = ("The block_info gives the trial names and number of occurences of each trial " 
                                            "for each block in the file.")
            block_info = file.create_group('block_info')
            for block in self.block_info.keys():
                blockname = block_info.create_group(block)
                for trial, value in self.block_info[block].items():
                    if value is None:
                        value = "None"
                    trialname = blockname.create_group(trial)
                    trialname.attrs[trial] = value
                    
            # Start by adding a description of this group to the root
            file.attrs['desc_trial_sets'] = ("The trial_sets contains several boolean arrays, each of length TRIALS, indicating whether "
                                        "a trial belongs to that trial set. Some examples are 'pursuit' or 'learning' trial sets, "
                                        "which indicate that a trial consists of pursuit eye movements along the pursuit or learning "
                                        "axes, respectively. Also present are trial sets named after neurons, e.g. 'PC_01' trial set "
                                        "indicates the trials for which neural unit PC_01 was present and adequately sorted. "
                                        "These could all be recomputed on the fly but storing these boolean indices makes lookup easy. "
                                        "Some additional description attributes are included. ")
            trial_sets = file.create_group('trial_sets')
            for t_set in self.trial_sets.keys():
                trial_sets.create_dataset(t_set, data=self.trial_sets[t_set])
            # Now add on the miscellaneous attribute trial sets
            trial_sets.attrs['desc_is_stab_learning'] = ("is_stab_learning is a boolean flag indicating that stabilization was used for "
                                                        "the instruction trials (TRUE) or not (FALSE).")
            trial_sets.create_dataset('is_stab_learning', data=self.is_stab_learning)
            trial_sets.attrs['desc_is_instructed'] = ("is_instructed contains a vector of length TRIALS where each element is a boolean "
                                                    "indicating whether that trial contained an instructive direction change (TRUE) or "
                                                    " not (FALSE).")
            trial_sets.create_dataset('is_instructed', data=self.is_instructed)
            trial_sets.attrs['desc_n_instructed'] = ("n_instructed contains a vector of length TRIALS where each element gives the TOTAL "
                                                    "number of learning/instruction trials that have been presented previous to the "
                                                    "trial at index n.")
            trial_sets.create_dataset('n_instructed', data=self.n_instructed)
            trial_sets.attrs['desc_sacc_and_err_trials'] = ("sacc_and_err_trials contains a vector of length TRIALS where each element indicates "
                                                    "whether the trial was flagged for removal due to the presence of excessive saccades or errors.")
            trial_sets.create_dataset('sacc_and_err_trials', data=self.sacc_and_err_trials)
                
                    
            # Start by adding a description of this group to the root
            file.attrs['desc_block_name_to_learn_name'] = ("The block_name_to_learn_name provides a convenience lookup table that returns "
                                                        "the learning trial name for each possible learning block name.")
            block_name_to_learn_name = file.create_group('block_name_to_learn_name')
            for key, value in self.block_name_to_learn_name.items():
                if value is None:
                    value = "None"
                block_name_to_learn_name.attrs[key] = value
                
            # Start by adding a description of this group to the root
            file.attrs['desc_blocks'] = ("The blocks lists all possible block names as keys. Each entry contains an index window "
                                        "of the trial numbers that correspond to that block. Entries are in 'slicing' format and "
                                        "so are not inclusive of the second index. e.g. blocks['blockname'] = [1, 5] means that "
                                        "trials [1, 2, 3, 4] belong to the block 'blockname'.")
            blocks = file.create_group('blocks')
            for key, value in self.blocks.items():
                if value is None:
                    value = "None"
                blocks.attrs[key] = value
                
            # Start by adding a description of this group to the root
            file.attrs['desc_directions'] = ("The directions provides a convenience lookup table that returns "
                                            "the real world pursuit direction corresponding to each pursuit direction "
                                            "relative to the learning trial orientation.")
            directions = file.create_group('directions')
            for key, value in self.directions.items():
                if value is None:
                    value = "None"
                directions.attrs[key] = value
                
            # Start by adding a description of this group to the root
            file.attrs['desc_neuron_info'] = ("The neuron_info gives the neuron names and the names of their associated data series. "
                                            "Also contains the Neuron objects in the original data structure. This is mostly used "
                                            "as a reference in the Session class but it does give all the neuron names. ")
            neuron_info = file.create_group('neuron_info')
            # We are skipping saving anything about the Neuron objects themselves here
            for info_name in ['series_to_name', 'neuron_names', 'dt']:
                if info_name == "series_to_name":
                    # This is a nested dict
                    info_type = neuron_info.create_group(info_name)
                    for s_name, value in self.neuron_info[info_name].items():
                        info_type.attrs[s_name] = value
                elif info_name == "neuron_names":
                    neuron_info.create_dataset("neuron_names", data=np.array(self.neuron_info[info_name], dtype='object'),
                                            dtype=h5py.special_dtype(vlen=str))
                else:
                    neuron_info.attrs[info_name] = self.neuron_info[info_name]
                
                
            # START MAJOR TRIAL DATA GROUP
            file.attrs['desc_trial_data'] = ("The trial_data group contains all of the trial data, such as behavioral measurements, "
                                            "target position, neuron spikes, target motion event times etc. ")
            trial_data = file.create_group('trial_data')
            trial_data.attrs['desc_trial_n'] = ("Trial data contains a subgroup for each trial of data with naming convention "
                                                "trial_n where n is the trial number between 0-N. ")
            # TRIAL INFO data
            trial_data.attrs['desc_name'] = ("The name of the trial ")
            trial_data.attrs['desc_used_stab'] = ("A boolean value indicated whether velocity stabilization was used on this trial (TRUE) or not (FALSE) ")
            trial_data.attrs['desc_duration'] = ("The integer value of trial duration in ms as given by Maestro ")
            trial_data.attrs['desc_events'] = ("A subgroup containing a list of attributes containing the trial time, in ms, of the named event. "
                                            "'fixation_onset': time fixation target turned on; 'rand_fix_onset': time the random duration fixation "
                                            "interval started; 'target_onset': time target motion started; 'start_stabwin': time at which the "
                                            "velocity stabilization orthogonal to target motion began; 'instruction_onset': time of the instruction "
                                            "direction change during the learning trials; 'target_offset': time at which target motion stopped " )
            
            # EYE data
            trial_data.attrs['desc_pos_x'] = ("Eye position on the horizontal, x axis ")
            trial_data.attrs['desc_pos_y'] = ("Eye position on the vertical, y axis ")
            trial_data.attrs['desc_vel_x'] = ("Eye velocity on the horizontal, x axis ")
            trial_data.attrs['desc_vel_y'] = ("Eye velocity on the vertical, y axis ")
            
            # TARGET data
            trial_data.attrs['desc_targ_pos_x'] = ("Target position on the horizontal, x axis ")
            trial_data.attrs['desc_targ_pos_y'] = ("Target position on the vertical, y axis ")
            trial_data.attrs['desc_targ_vel_x'] = ("Target velocity on the horizontal, x axis ")
            trial_data.attrs['desc_targ_vel_y'] = ("Target velocity on the vertical, y axis ")
            
            # NEURON data
            trial_data.attrs['desc_neurons'] = ("Any array of the spike times with respect to trial start, in ms, for the neuron with the name indicated by the data group name. "
                                                "The cell type naming conventions are: PC - Purkinje Cell with confirmed complex spike; putPC - putative "
                                                "Purkinje cell, without a confirmed complex spike; CS - a complex spike without a matching PC simple "
                                                "spike train; MLI - molecular layer interneuron (likely basket cell) with monosynaptic inhibition of "
                                                "a PC; MF - mossy fiber, with triphasic waveform; GC - Golgi cell; UBC - Unipolar brush cell, with "
                                                "prolonged integrative effects related to mossy fiber activity; N - unknown/unclassified unit. "
                                                "Unit names are numerically ordered, arbitrarily, by appending _00, _01, etc. ")
            trial_data.attrs['desc_neurons_CS'] = ("The spike train for the complex spikes for the neuron indicated. Will only be present for "
                                                "confirmed Purkinje Cells, named 'PC_nn'. ")
            
            # Now loop over each trial and add the data
            for t in range(0, len(self)):
                curr_trial = trial_data.create_group(f"trial_{t}")
                # TRIAL INFO data
                curr_trial.attrs['name'] = self._trial_lists['__main'][t]['name']
                curr_trial.attrs['used_stab'] = self._trial_lists['__main'][t]['used_stab']
                curr_trial.attrs['duration'] = self._trial_lists['__main'][t]['dur']
                events = curr_trial.create_group("events")
                for evt_name, evt_val in self._trial_lists['__main'][t]['events'].items():
                    value = evt_val if evt_val is not None else "None"
                    events.attrs[evt_name] = value            
                
                # EYE data
                curr_trial.create_dataset("pos_x", data=self._trial_lists['eye'][t]['eye']['horizontal_eye_position'])
                curr_trial.create_dataset("pos_y", data=self._trial_lists['eye'][t]['eye']['vertical_eye_position'])
                curr_trial.create_dataset("vel_x", data=self._trial_lists['eye'][t]['eye']['horizontal_eye_velocity'])
                curr_trial.create_dataset("vel_y", data=self._trial_lists['eye'][t]['eye']['vertical_eye_velocity'])
                
                # TARGET data
                curr_trial.create_dataset("targ_pos_x", data=self._trial_lists['target0'][t]['data']['horizontal_target_position'])
                curr_trial.create_dataset("targ_pos_y", data=self._trial_lists['target0'][t]['data']['vertical_target_position'])
                curr_trial.create_dataset("targ_vel_x", data=self._trial_lists['target0'][t]['data']['horizontal_target_velocity'])
                curr_trial.create_dataset("targ_vel_y", data=self._trial_lists['target0'][t]['data']['vertical_target_velocity'])
                
                # NEURON data
                # Check neuron names match
                for ni_name, tl_name in zip(sorted(self.neuron_info['neuron_names']), sorted(self._trial_lists['neurons'][t]['meta_data']['neuron_names'])):
                    if ni_name != tl_name:
                        raise ValueError(f"Neuron name from neuron_info {ni_name} does not match the trial list name {tl_name} for trial {t}")
                    try:
                        curr_trial.create_dataset(ni_name, data=self._trial_lists['neurons'][t]['meta_data'][ni_name]['spikes'])
                    except TypeError:
                        # Should be caused by returning None when no spikes are present
                        if self._trial_lists['neurons'][t]['meta_data'][ni_name]['spikes'] is None:
                            # The expected exception
                            curr_trial.create_dataset(ni_name, shape=(0,), dtype=np.float64)
                        else:
                            raise
                    if "complex_spikes" in self._trial_lists['neurons'][t]['meta_data'][ni_name]:
                        try:
                            curr_trial.create_dataset(f"{ni_name}_CS", data=self._trial_lists['neurons'][t]['meta_data'][ni_name]['complex_spikes'])
                        except TypeError:
                            if self._trial_lists['neurons'][t]['meta_data'][ni_name]['complex_spikes'] is None:
                                # The expected exception
                                curr_trial.create_dataset(f"{ni_name}_CS", shape=(0,), dtype=np.float64)
                            else:
                                raise
        return None
    
    @staticmethod
    def print_h5_info(filename):
        """Convenience function that will print the names of attributes and groups and datasets visible at the main outer
        root hierarchy level for an H5 file saved using "save_ldp_session." """
        with h5py.File(filename, 'r') as file:
            print("Root key ATTRIBUTES")
            for root_key in file.attrs:
                print(root_key)
            print("Root key GROUPs and DATASETS")
            for root_key in file.keys():
                print(root_key)

    @staticmethod
    def load_ldp_session(filename):
        """ This prescribes methods to load the data from an H5 file saved by "save_ldp_session". Data are loaded back into a dictionary
        that most closely reflects the underlying H5 hierarchy. There are currently no methods for importing this loaded data
        dictionary back into an LDPSession class object.
        """
        # Define these internal use unpacking functions to un-clutter the main "with" load loop at the bottom
        def unpack_base_and_tune_blocks(file, loaded_dict):
            group = file['base_and_tune_blocks']
            sub_dict = {sub_key: group.attrs[sub_key] for sub_key in group.attrs}
            loaded_dict['base_and_tune_blocks'] = sub_dict
            
        def unpack_block_name_to_learn_name(file, loaded_dict):
            group = file['block_name_to_learn_name']
            sub_dict = {sub_key: group.attrs[sub_key] for sub_key in group.attrs}
            loaded_dict['block_name_to_learn_name'] = sub_dict
            
        def unpack_blocks(file, loaded_dict):
            group = file['blocks']
            sub_dict = {}
            for sub_key in group.attrs:
                if isinstance(group.attrs[sub_key], str):
                    if group.attrs[sub_key] == "None":
                        load_val = None
                else:
                    load_val = list(group.attrs[sub_key])
                sub_dict[sub_key] = load_val
            loaded_dict['blocks'] = sub_dict
            
        def unpack_directions(file, loaded_dict):
            group = file['directions']
            sub_dict = {sub_key: group.attrs[sub_key] for sub_key in group.attrs}
            loaded_dict['directions'] = sub_dict
            
        def unpack_block_info(file, loaded_dict):
            sub_dict = {}
            for bname_group in file['block_info'].keys():
                block_dict = {}
                for tname in file['block_info'][bname_group].keys():
                    block_dict[tname] = file['block_info'][bname_group][tname].attrs[tname]
                sub_dict[bname_group] = block_dict
            loaded_dict['block_info'] = sub_dict
            
        def unpack_neuron_info(file, loaded_dict):
            ninfo_group = file['neuron_info']
            loaded_dict['neuron_info'] = {}
            for info_name in ['series_to_name', 'neuron_names', 'dt']:
                if info_name == "series_to_name":
                    # This is a nested dict
                    sub_dict = {}
                    for nname_group in file['neuron_info'][info_name].attrs:
                        sub_dict[nname_group] = file['neuron_info'][info_name].attrs[nname_group]
                    loaded_dict['neuron_info'][info_name] = sub_dict
                elif info_name == "neuron_names":
                    loaded_dict['neuron_info'][info_name] = list(ninfo_group[info_name][()])
                    # Convert byte string to unicode string
                    loaded_dict['neuron_info'][info_name] = [s.decode('utf-8') for s in loaded_dict['neuron_info'][info_name]]
                else:
                    loaded_dict['neuron_info'][info_name] = ninfo_group.attrs[info_name]
                    
        def unpack_trial_sets(file, loaded_dict):
            trial_sets_group = file['trial_sets']
            loaded_dict['trial_sets'] = {}
            for t_set in trial_sets_group.keys():
                loaded_dict['trial_sets'][t_set] = np.array(trial_sets_group[t_set])
                
        def unpack_trial_data(file, loaded_dict):
            trial_data_group = file['trial_data']
            loaded_dict['trial_data'] = {}
            for t_name in trial_data_group.keys():
                curr_trial_group = trial_data_group[t_name]
                loaded_dict['trial_data'][t_name] = {}
                # Gets the basic trial info and events
                loaded_dict['trial_data'][t_name]['name'] = curr_trial_group.attrs['name']
                loaded_dict['trial_data'][t_name]['used_stab'] = curr_trial_group.attrs['used_stab']
                loaded_dict['trial_data'][t_name]['duration'] = curr_trial_group.attrs['duration']
                loaded_dict['trial_data'][t_name]['events'] = {}
                for att in curr_trial_group['events'].attrs:
                    load_val = curr_trial_group['events'].attrs[att]
                    load_val = load_val if load_val != "None" else None
                    loaded_dict['trial_data'][t_name]['events'][att] = load_val
                # Adds in the behavior and neuron data 
                for dataset in curr_trial_group.keys():
                    if not isinstance(curr_trial_group[dataset], h5py.Dataset):
                        # Some group so skip it
                        continue
                    loaded_dict['trial_data'][t_name][dataset] = np.array(curr_trial_group[dataset])

        with h5py.File(filename, 'r') as file:
            loaded_dict = {}
            # Loop over file attributes
            for root_key in file.attrs:
                if root_key == "credits":
                    loaded_dict['credits'] = file.attrs['credits']
                elif root_key == "filename":
                    loaded_dict['filename'] = file.attrs['filename']
                elif root_key == "is_weird_Yan":
                    loaded_dict['is_weird_Yan'] = file.attrs['is_weird_Yan']
                elif root_key == "is_weird_Yoda":
                    loaded_dict['is_weird_Yoda'] = file.attrs['is_weird_Yoda']
                elif root_key == "learning_trial_name":
                    loaded_dict['learning_trial_name'] = file.attrs['learning_trial_name']
                elif root_key == "meta_dict_name":
                    loaded_dict['meta_dict_name'] = file.attrs['meta_dict_name']
                elif root_key == "nan_saccades":
                    loaded_dict['nan_saccades'] = file.attrs['nan_saccades']
                elif root_key == "rem_sacc_errs":
                    loaded_dict['rem_sacc_errs'] = file.attrs['rem_sacc_errs']
                elif root_key == "rotate":
                    loaded_dict['rotate'] = file.attrs['rotate']
                elif root_key == "saccade_ind_cushion":
                    loaded_dict['saccade_ind_cushion'] = file.attrs['saccade_ind_cushion']
                elif root_key == "session_name":
                    loaded_dict['session_name'] = file.attrs['session_name']
                elif root_key == "verbose":
                    loaded_dict['verbose'] = file.attrs['verbose']
                elif root_key[0:5] == "desc_":
                    # Skip these since they are just descriptions of data groups
                    pass
                else:
                    raise ValueError(f"Unrecognized root directory group name {root_key}")
            
            # Loop over data subgroups
            for root_key in file.keys():
                if root_key == "rotation_matrix":
                    loaded_dict['rotation_matrix'] = np.array(file['rotation_matrix'])
                elif root_key == "base_and_tune_blocks":
                    unpack_base_and_tune_blocks(file, loaded_dict)
                elif root_key == "block_info":
                    unpack_block_info(file, loaded_dict)
                elif root_key == "neuron_info":
                    unpack_neuron_info(file, loaded_dict)
                elif root_key == "block_name_to_learn_name":
                    unpack_block_name_to_learn_name(file, loaded_dict)
                elif root_key == "blocks":
                    unpack_blocks(file, loaded_dict)
                elif root_key == "directions":
                    unpack_directions(file, loaded_dict)
                # elif root_key == "trial_map":
                #     unpack_trial_map(file, loaded_dict)
                elif root_key == "trial_sets":
                    unpack_trial_sets(file, loaded_dict)
                elif root_key == "trial_data":
                    unpack_trial_data(file, loaded_dict)
                else:
                    raise ValueError(f"Unrecognized root directory group name {root_key}")
        return loaded_dict
    
    def set_base_and_tune_blocks(self):
        """ Creates a dictionary that indicates which blocks should be used for calculating the
        pre/post learning baseline pursuit responses and also the blocks that should be used
        for calculating linear model fits of the data.
        """
        self.base_and_tune_blocks = {}
        # Search for tuning block
        if self.blocks['StabTunePre'] is not None:
            # Prefer stab tuning
            if (self.blocks['Learning'][0] - self.blocks['StabTunePre'][1]) > 50:
                print(f"Pre block 'StabTunePre' precedes learning by over 50 trials")
            self.base_and_tune_blocks['tuning_block'] = "StabTunePre"
            self.base_and_tune_blocks['baseline_block'] = "StabTunePre"
        else:
            if self.blocks['StandTunePre'] is not None:
                # Then Standard tuning
                if (self.blocks['Learning'][0] - self.blocks['StandTunePre'][1]) > 50:
                    print(f"Pre block 'StandTunePre' precedes learning by over 50 trials")
                self.base_and_tune_blocks['tuning_block'] = "StandTunePre"
                self.base_and_tune_blocks['baseline_block'] = "StandTunePre"
            elif self.blocks['BaselinePre'] is not None:
                # I guess use baseline trials if we can find them
                if (self.blocks['Learning'][0] - self.blocks['BaselinePre'][1]) > 50:
                    print(f"Pre block 'BaselinePre' precedes learning by over 50 trials")
                self.base_and_tune_blocks['tuning_block'] = "BaselinePre"
            else:
                # We can't find a tuning block
                print("Can not find tuning block")
                self.base_and_tune_blocks['tuning_block'] = None

            if self.is_weird_Yan:
                # Check if the weird Yan tuning trials are way before learning we need to use the baseline block
                if (self.blocks['Learning'][0] - self.blocks['StandTunePre'][1]) > 50:
                    self.base_and_tune_blocks['baseline_block'] = "BaselinePre"

        # Same but search for POST tuning block
        if self.blocks['StabTunePost'] is not None:
            # Prefer stab tuning
            if (self.blocks['StabTunePost'][0] - self.blocks['Learning'][1]) > 50:
                print(f"Post block 'StabTunePost' follows learning by over 50 trials")
            self.base_and_tune_blocks['post_tuning_block'] = "StabTunePost"
        elif self.blocks['StandTunePost'] is not None:
            # Then Standard tuning
            if (self.blocks['StandTunePost'][0] - self.blocks['Learning'][1]) > 50:
                print(f"Post block 'StandTunePost' follows learning by over 50 trials")
            self.base_and_tune_blocks['post_tuning_block'] = "StandTunePost"
        elif self.blocks['BaselinePost'] is not None:
            # Try Baseline tuning
            if (self.blocks['BaselinePost'][0] - self.blocks['Learning'][1]) > 50:
                print(f"Post block 'BaselinePost' follows learning by over 50 trials")
            self.base_and_tune_blocks['post_tuning_block'] = "BaselinePost"
        else:
            # We can't find a post tuning block
            print("Can not find post tuning block")
            self.base_and_tune_blocks['post_tuning_block'] = None

        # Same but search for post WASHOUT block
        if self.blocks['StabTuneWash'] is not None:
            # Prefer stab tuning
            if (self.blocks['StabTuneWash'][0] - self.blocks['Washout'][1]) > 50:
                print(f"Washout block 'StabTuneWash' follows washout by over 50 trials")
            self.base_and_tune_blocks['wash_tuning_block'] = "StabTuneWash"
        elif self.blocks['StandTuneWash'] is not None:
            # Then Standard tuning
            if (self.blocks['StandTuneWash'][0] - self.blocks['Washout'][1]) > 50:
                print(f"Washout block 'StandTuneWash' follows washout by over 50 trials")
            self.base_and_tune_blocks['wash_tuning_block'] = "StandTuneWash"
        elif self.blocks['BaselineWash'] is not None:
            # Try Baseline tuning
            if (self.blocks['BaselineWash'][0] - self.blocks['Washout'][1]) > 50:
                print(f"Washout block 'BaselineWash' follows washout by over 50 trials")
            self.base_and_tune_blocks['wash_tuning_block'] = "BaselineWash"
        else:
            # We can't find a post washout block
            self.base_and_tune_blocks['wash_tuning_block'] = None
            
