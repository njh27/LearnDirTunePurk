import numpy as np
from NeuronAnalysis.fit_neuron_to_eye import FitNeuronToEye



def get_neuron_trace_data(neuron, trace_win, sigma=12.5, cutoff_sigma=4):
    """
    """
    # Some currently hard coded variables
    fix_win = [-300, 0]
    use_smooth_fix = True
    use_baseline_block = "StandTunePre"

    if neuron.session.blocks['Learning'] is None:
        # No learning data so nothing to do here
        return {}

    # Get linear model fit
    fit_eye_model = FitNeuronToEye(neuron, trace_win, use_baseline_block, trial_sets=None,
                                    lag_range_eye=[-75, 150])
    fit_eye_model.fit_pcwise_lin_eye_kinematics(bin_width=10, bin_threshold=5,
                                                fit_constant=False, fit_avg_data=False,
                                                quick_lag_step=10, fit_fix_adj_fr=True)
    
    # Get 4 direction tuning block traces
    out_traces = {}
    for tune_block in ["StandTunePre", "StabTunePre", "StabTunePost", "StandTunePost", "StabTuneWash", "StandTuneWash"]:
        out_traces[tune_block] = {}
        for trial_type in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
            out_traces[tune_block][trial_type] = {}
            fr, t_inds = neuron.get_firing_traces_fix_adj(trace_win, tune_block, trial_type, 
                                                        fix_time_window=fix_win, sigma=sigma, 
                                                        cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                        rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                                        return_inds=True)
            out_traces[tune_block][trial_type]['fr'] = fr
            if fr.shape[0] == 0:
                out_traces[tune_block][trial_type]['y_hat'] = fr
                out_traces[tune_block][trial_type]['eyev_p'] = fr
                out_traces[tune_block][trial_type]['eyev_l'] = fr
                continue
            if neuron.name[0:2] == "PC":
                # This is a PC with CS so get them
                out_traces[tune_block][trial_type]['cs'] = neuron.get_CS_dataseries_by_trial(trace_win, tune_block, None)
            X_eye, x_shape = fit_eye_model.get_pcwise_lin_eye_kin_predict_data_by_trial(tune_block, t_inds, 
                                                                                        return_shape=True, return_inds=False)
            out_traces[tune_block][trial_type]['y_hat'] = fit_eye_model.predict_pcwise_lin_eye_kinematics_by_trial(X_eye, x_shape)
            eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", trace_win, blocks=tune_block,
                                                            trial_sets=t_inds, return_inds=False)
            out_traces[tune_block][trial_type]['eyev_p'] = eyev_p
            out_traces[tune_block][trial_type]['eyev_l'] = eyev_l
    
    # Now get the Learning block traces
    out_traces['Learning'] = {}
    for trial_type in ["instruction", "learning", "anti_pursuit", "pursuit", "anti_learning"]:
        out_traces['Learning'][trial_type] = {}
        fr, t_inds = neuron.get_firing_traces_fix_adj(trace_win, "Learning", trial_type, 
                                                    fix_time_window=fix_win, sigma=sigma, 
                                                    cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                    rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                                    return_inds=True)
        out_traces['Learning'][trial_type]['fr'] = fr
        if len(t_inds) == 0:
            out_traces['Learning'][trial_type]['y_hat'] = fr
            out_traces['Learning'][trial_type]['n_inst'] = t_inds
            out_traces['Learning'][trial_type]['eyev_p'] = fr
            out_traces['Learning'][trial_type]['eyev_l'] = fr
            continue
        if neuron.name[0:2] == "PC":
            # This is a PC with CS so get them
            out_traces['Learning'][trial_type]['cs'] = neuron.get_CS_dataseries_by_trial(trace_win, "Learning", t_inds)
        X_eye, x_shape = fit_eye_model.get_pcwise_lin_eye_kin_predict_data_by_trial("Learning", t_inds, 
                                                                                        return_shape=True, return_inds=False)
        out_traces['Learning'][trial_type]['y_hat'] = fit_eye_model.predict_pcwise_lin_eye_kinematics_by_trial(X_eye, x_shape)
        n_inst = np.array([neuron.session.n_instructed[t_ind] for t_ind in t_inds], dtype=np.int64)
        out_traces['Learning'][trial_type]['n_inst'] = n_inst
        eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", trace_win, blocks="Learning",
                                                            trial_sets=t_inds, return_inds=False)
        out_traces['Learning'][trial_type]['eyev_p'] = eyev_p
        out_traces['Learning'][trial_type]['eyev_l'] = eyev_l

    # And the Washout block
    out_traces['Washout'] = {}
    out_traces['Washout']['instruction'] = {}
    fr, t_inds = neuron.get_firing_traces_fix_adj(trace_win, "Washout", "instruction", 
                                                fix_time_window=fix_win, sigma=sigma, 
                                                cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                                return_inds=True)
    out_traces['Washout']["instruction"]['fr'] = fr
    if len(t_inds) == 0:
        out_traces['Washout']["instruction"]['y_hat'] = fr
        out_traces['Washout']["instruction"]['n_inst'] = t_inds
        out_traces['Washout']["instruction"]['eyev_p'] = fr
        out_traces['Washout']["instruction"]['eyev_l'] = fr
    else:
        if neuron.name[0:2] == "PC":
            # This is a PC with CS so get them
            out_traces['Washout']["instruction"]['cs'] = neuron.get_CS_dataseries_by_trial(trace_win, "Washout", t_inds)
        X_eye, x_shape = fit_eye_model.get_pcwise_lin_eye_kin_predict_data_by_trial("Washout", t_inds, 
                                                                                        return_shape=True, return_inds=False)
        out_traces['Washout']["instruction"]['y_hat'] = fit_eye_model.predict_pcwise_lin_eye_kinematics_by_trial(X_eye, x_shape)
        n_inst = np.array([neuron.session.n_instructed[t_ind] for t_ind in t_inds], dtype=np.int64)
        out_traces['Washout']["instruction"]['n_inst'] = n_inst
        eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", trace_win, blocks="Washout",
                                                            trial_sets=t_inds, return_inds=False)
        out_traces['Washout']["instruction"]['eyev_p'] = eyev_p
        out_traces['Washout']["instruction"]['eyev_l'] = eyev_l

    return out_traces