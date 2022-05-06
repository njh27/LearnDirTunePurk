import numpy as np


def get_position(ldp_sess):
    pass


def get_data_array(ldp_sess, series_name, time_window, blocks=None,
                   trial_sets=None, return_inds=False):
        """ Returns a n trials by m time points numpy array of the requested
        timeseries data. Missing data points are filled in with np.nan.
        Call "data_names()" to get a list of available data names. """
        data_out = []
        t_inds_out = []
        data_name = self.__series_names[series_name]
        t_inds = self.__parse_blocks_trial_sets(blocks, trial_sets)
        for t in t_inds:
            if not self._session_trial_data[t]['incl_align']:
                # Trial is not aligned with others due to missing event
                continue
            trial_obj = self._trial_lists[data_name][t]
            self._set_t_win(t, time_window)
            valid_tinds = self._session_trial_data[t]['curr_t_win']['valid_tinds']
            out_inds = self._session_trial_data[t]['curr_t_win']['out_inds']
            t_data = np.full(out_inds.shape[0], np.nan)
            t_data[out_inds] = trial_obj['data'][series_name][valid_tinds]
            data_out.append(t_data)
            t_inds_out.append(t)
        if return_inds:
            return np.vstack(data_out), np.hstack(t_inds_out)
        else:
            return np.vstack(data_out)
