

















data_out = []
        data_name = self.__series_names[series_name]
        if data_name == "neurons":
            # If requested data is a neuron we need to add its valid trial
            # index to the input trial_sets
            check_missing = True
            neuron_name = self.neuron_info['series_to_name'][series_name]
            trial_sets = self.neuron_info[neuron_name].append_valid_trial_set(trial_sets)
        else:
            check_missing = False
        t_inds = self._parse_blocks_trial_sets(blocks, trial_sets)
        t_inds_to_delete = []
        for i, t in enumerate(t_inds):
            if not self._session_trial_data[t]['incl_align']:
                # Trial is not aligned with others due to missing event
                t_inds_to_delete.append(i)
                continue
            trial_obj = self._trial_lists[data_name][t]
            if check_missing:
                if trial_obj[self.meta_dict_name][neuron_name]['spikes'] is None:
                    # Data are missing for this neuron trial series
                    t_inds_to_delete.append(i)
                    continue
            self._set_t_win(t, time_window)
            valid_tinds = self._session_trial_data[t]['curr_t_win']['valid_tinds']
            out_inds = self._session_trial_data[t]['curr_t_win']['out_inds']
            t_data = np.full(out_inds.shape[0], np.nan)
            t_data[out_inds] = trial_obj['data'][series_name][valid_tinds]
            data_out.append(t_data)

        data_out = [] if len(data_out) == 0 else np.vstack(data_out)
        if return_inds:
            if len(t_inds_to_delete) > 0:
                del_sel = np.zeros(t_inds.size, dtype='bool')
                del_sel[np.array(t_inds_to_delete, dtype=np.int64)] = True
                t_inds = t_inds[~del_sel]
            return data_out, t_inds
        else:
            return data_out
