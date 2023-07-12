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
        for tune_trial in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
            out_traces[tune_block][tune_trial] = {}
            fr, t_inds = neuron.get_firing_traces_fix_adj(trace_win, tune_block, tune_trial, 
                                                        fix_time_window=fix_win, sigma=sigma, 
                                                        cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                        rate_offset=0., use_smooth_fix=use_smooth_fix, 
                                                        return_inds=True)
            out_traces[tune_block][tune_trial]['fr'] = fr
            if fr.shape[0] == 0:
                out_traces[tune_block][tune_trial]['y_hat'] = fr
                out_traces[tune_block][tune_trial]['eyev_p'] = fr
                out_traces[tune_block][tune_trial]['eyev_l'] = fr
                continue
            X_eye = fit_eye_model.get_pcwise_lin_eye_kin_predict_data(tune_block, t_inds, time_window=trace_win)
            out_traces[tune_block][tune_trial]['y_hat'] = fit_eye_model.predict_pcwise_lin_eye_kinematics(X_eye)
            eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", trace_win, blocks=tune_block,
                                                            trial_sets=t_inds, return_inds=False)
            out_traces[tune_block][tune_trial]['eyev_p'] = eyev_p
            out_traces[tune_block][tune_trial]['eyev_l'] = eyev_l
    
    # Now get the Learning block traces
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
        X_eye = fit_eye_model.get_pcwise_lin_eye_kin_predict_data("Learning", t_inds, trace_win)
        out_traces['Learning'][trial_type]['y_hat'] = fit_eye_model.predict_pcwise_lin_eye_kinematics(X_eye)
        n_inst = np.array([neuron.session.n_instructed[t_ind] for t_ind in t_inds], dtype=np.int64)
        out_traces['Learning'][trial_type]['n_inst'] = n_inst
        eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", trace_win, blocks="Learning",
                                                            trial_sets=t_inds, return_inds=False)
        out_traces['Learning'][trial_type]['eyev_p'] = eyev_p
        out_traces['Learning'][trial_type]['eyev_l'] = eyev_l

    # And the Washout block
    out_traces['Washout'][trial_type] = {}
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
        X_eye = fit_eye_model.get_pcwise_lin_eye_kin_predict_data("Washout", t_inds, trace_win)
        out_traces['Washout']["instruction"]['y_hat'] = fit_eye_model.predict_pcwise_lin_eye_kinematics(X_eye)
        n_inst = np.array([neuron.session.n_instructed[t_ind] for t_ind in t_inds], dtype=np.int64)
        out_traces['Washout']["instruction"]['n_inst'] = n_inst
        eyev_p, eyev_l = neuron.session.get_xy_traces("eye velocity", trace_win, blocks="Washout",
                                                            trial_sets=t_inds, return_inds=False)
        out_traces['Washout']["instruction"]['eyev_p'] = eyev_p
        out_traces['Washout']["instruction"]['eyev_l'] = eyev_l

    return out_traces