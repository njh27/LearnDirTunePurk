import numpy as np
import re
from SessionAnalysis.session import Session



def parse_learning_directions(sess):
    """ Parse through the block/trial names established above to find the main
    learning block and direction. This then allows us to determine the other
    directions and unlearning with respect to this etc. """
    n_max_learn = 0
    b_max_learn = None
    for block in sess.blocks.keys():
        if "Learn" in block:
            block_len = sess.blocks[block][1] - sess.blocks[block][0]
            if block_len > n_max_learn:
                n_max_learn = block_len
                b_max_learn = block
    if b_max_learn is None:
        # We did NOT find a learning block
        return
    # We found a learning block so carry on
    pursuit_dir, learn_dir = b_max_learn.split("Learn")
    pursuit_dir = int(pursuit_dir)
    learn_dir = int(learn_dir)
    anti_pursuit_dir = (pursuit_dir + 180) % 360
    anti_learn_dir = (learn_dir + 180) % 360

    directions = {}
    directions['pursuit'] = pursuit_dir
    directions['learning'] = learn_dir
    directions['anti_pursuit'] = anti_pursuit_dir
    directions['anti_learning'] = anti_learn_dir

    return directions


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

    def assign_learning_blocks(self):
        """ Adds a directions dictionary indicating the learning and pursuit directions
        and a rotation matrix property. Finishes by verifying block order and span."""
        self.directions = parse_learning_directions(self)
        # New learning directions means new rotation matrix
        self._set_rotation_matrix()

        # Set learning and washout block alias names
        learn_trial_name = str(self.directions['pursuit']) + "Learn" + str(self.directions['learning'])
        max_learn_len = 0
        new_blocks = {}
        # Now name the learning block and washout blocks accordingly
        for block in self.blocks.keys():
            # Choose learning block as longest block in learning direction
            block_len = self.blocks[block][1] - self.blocks[block][0]
            # Do split on num for default number counting of multiple blocks
            if ( (block.split("_num")[0] == learn_trial_name) and (block_len > max_learn_len) ):
                new_blocks['Learning'] = self.blocks[block]
                max_learn_len = block_len
        self.blocks.update(new_blocks)
        new_blocks = {}

        washout_trial_name = str(self.directions['pursuit']) + "Learn" + str(self.directions['anti_learning'])
        max_wash_len = 0
        max_washpre_len = 0
        for block in self.blocks.keys():
            block_len = self.blocks[block][1] - self.blocks[block][0]
            # Do split on num for default number counting of multiple blocks
            if ( (block.split("_num")[0] == washout_trial_name) ):
                if block_len > max_learn_len:
                    # Washout block found longer than learning block?
                    raise ValueError("Washout block is longer than learning block! This probably isn't working correctly!")

                if ( (self.blocks[block][0] >= self.blocks['Learning'][1]) and
                     (block_len > max_wash_len) ):
                    # Follows the learning block and longest current, so is washout
                    new_blocks['Washout'] = self.blocks[block]
                    max_wash_len = block_len
                elif ( (self.blocks[block][1] <= self.blocks['Learning'][0]) and
                       (block_len > max_washpre_len) ):
                    # Precedes the learning block so is washout pre
                    new_blocks['WashoutPre'] = self.blocks[block]
                    max_washpre_len = block_len
                else:
                    # Some washout direction not learning?
                    pass
        self.blocks.update(new_blocks)
        # Recheck block order
        self._verify_block_order()
        self._verify_block_continuity()
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
        next_b_start = 0
        while next_b_start < len(self):
            found_match = False
            for b_name in self.blocks.keys():
                if b_name == '270Learn180_num01':
                    continue
                b_win = self.blocks[b_name]
                if b_win[0] == next_b_start:
                    found_match = True
                    next_b_start = b_win[1]
                    break
            if not found_match:
                raise ValueError("Blocks do not cover all trials present! Last starting block attempted at trial {0}.".format(next_b_start))
        return None

    def _verify_block_order(self):
        """ Checks that the pre and post block names in fact come before/after the
        selected learning block. """
        learn_start, learn_stop = self.blocks['Learning']
        for block in self.blocks.keys():
            if block == "FixTunePre":
                if self.blocks[block][1] > learn_start:
                    raise RuntimeError("{0} block ends at trial {1} which is after the learning block begins at trial {2}.".format(block, self.blocks[block][1], learn_start))
            elif block == "FixTunePost":
                if self.blocks[block][0] < learn_stop:
                    raise RuntimeError("{0} block starts at trial {1} which is before the learning block ends at trial {2}.".format(block, self.blocks[block][0], learn_stop))
            elif block == "FixTuneWashout":
                try:
                    wash_start, wash_stop = self.blocks['Washout']
                except KeyError:
                    raise RuntimeError("Fixation tuning washout block was found but no washout blocks are present!")
                if self.blocks[block][0] < wash_stop:
                    raise RuntimeError("{0} block starts at trial {1} which is before the washout block ends at trial {2}.".format(block, self.blocks[block][0], wash_stop))
                if self.blocks[block][0] < learn_stop:
                    raise RuntimeError("{0} block ends at trial {1} which is before the learning block ends at trial {2}.".format(block, self.blocks[block][1], learn_start))
            elif block == "RandVP":
                if self.blocks[block][1] > learn_start:
                    raise RuntimeError("{0} block ends at trial {1} which is after the learning block begins at trial {2}.".format(block, self.blocks[block][1], learn_start))
            elif block == "StabPre":
                if self.blocks[block][1] > learn_start:
                    raise RuntimeError("{0} block ends at trial {1} which is after the learning block begins at trial {2}.".format(block, self.blocks[block][1], learn_start))
            elif block == "StabPost":
                if self.blocks[block][0] < learn_stop:
                    raise RuntimeError("{0} block starts at trial {1} which is before the learning block ends at trial {2}.".format(block, self.blocks[block][0], learn_stop))
            elif block == "StabWashout":
                try:
                    wash_start, wash_stop = self.blocks['Washout']
                except KeyError:
                    raise RuntimeError("Stabilized tuning washout block was found but no washout blocks are present!")
                if self.blocks[block][0] < wash_stop:
                    raise RuntimeError("{0} block starts at trial {1} which is before the washout block ends at trial {2}.".format(block, self.blocks[block][0], wash_stop))
                if self.blocks[block][0] < learn_stop:
                    raise RuntimeError("{0} block ends at trial {1} which is before the learning block ends at trial {2}.".format(block, self.blocks[block][1], learn_start))
            elif block == "TunePre":
                if self.blocks[block][1] > learn_start:
                    raise RuntimeError("{0} block ends at trial {1} which is after the learning block begins at trial {2}.".format(block, self.blocks[block][1], learn_start))
            elif block == "TunePost":
                if self.blocks[block][0] < learn_stop:
                    raise RuntimeError("{0} block starts at trial {1} which is before the learning block ends at trial {2}.".format(block, self.blocks[block][0], learn_stop))
            else:
                pass
                # raise RuntimeError("Could not find verification order condition for block named {0}.".format(block))
        return None
