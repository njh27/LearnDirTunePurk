import numpy as np
import warnings



def subtract_baseline_firing(ldp_sess, base_block, base_set, base_data, x, y,
                             alpha_scale_factors=None):
    """ Subtracts the baseline data on the orthogonal axis relative to the
    4 axes defined by ldp_sess.trial_set_base_axis.
    Relatively simple function but cleans up other code since this is clumsy.
    Data are subtracted from x and y IN PLACE!"""
    if (len(x) == 0) or (len(y) == 0):
        return x, y
    # alpha_scale_factors = None
    # print("NO SCALING IN BASELINE OFF!!!!!!")
    if ldp_sess.trial_set_base_axis[base_set] == 0:
        if alpha_scale_factors is None:
            x = x - ldp_sess.baseline_tuning[base_block][base_data][base_set][0, :]
        else:
            x = x - (alpha_scale_factors * ldp_sess.baseline_tuning[base_block][base_data][base_set][0, :])
    elif ldp_sess.trial_set_base_axis[base_set] == 1:
        if alpha_scale_factors is None:
            y = y - ldp_sess.baseline_tuning[base_block][base_data][base_set][1, :]
        else:
            y = y - (alpha_scale_factors * ldp_sess.baseline_tuning[base_block][base_data][base_set][1, :])
    else:
        raise ValueError("Could not match baseline for subtraction for block '{0}', set '{1}', and data '{2}'.".format(base_block, base_set, base_data))

    return x, y
