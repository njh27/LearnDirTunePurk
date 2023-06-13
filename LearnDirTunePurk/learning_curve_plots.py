import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.backends.backend_pdf import PdfPages
from NeuronAnalysis.general import gauss_convolve
from NeuronAnalysis.neurons import PurkinjeCell



# Set some globally available lookup strings by block and trial names
block_strings = {'FixTunePre': "Fixation",
                    'RandVPTunePre': "Random position/velocity tuning",
                    'StandTunePre': "Standard tuning",
                    'StabTunePre': "Stabilized tuning",
                    'Learning': "Learning",
                    'FixTunePost': "Fixation",
                    'StabTunePost': "Stabilized tuning",
                    'Washout': "Opposite learning",
                    'FixTuneWash': "Fixation",
                    'StabTuneWash': "Stabilized tuning",
                    'StandTuneWash': "Standard tuning",
                    }
trial_strings = {'learning': "Learning",
                 'anti_learning': "Anti-learning",
                 'pursuit': "Pursuit",
                 'anti_pursuit': "Anti-pursuit"}
t_set_color_codes = {"learning": "green",
                    "anti_learning": "red",
                    "pursuit": "orange",
                    "anti_pursuit": "blue"}


def setup_axes():
    plot_handles = {}
    plot_handles['fig'] = plt.figure(figsize=(8, 11))

    col_scale = 5

    tuning_row_start = 0
    tuning_col_start = 0
    tuning_height = 10
    tuning_width = 2 * col_scale
    tuning_v_pad = 3
    tuning_h_pad = 3
    t_course_height = 32
    t_course_v_pad = 3
    t_course_gap = 15
    t_course_start = tuning_height * 3 + tuning_v_pad * 3 + t_course_gap

    total_rows = tuning_height * 3 + tuning_v_pad * 3 + t_course_height * 3 + t_course_v_pad * 3 + t_course_gap
    total_cols = 12 * col_scale
    spec = plot_handles['fig'].add_gridspec(total_rows, total_cols)

    row_slice = slice(tuning_row_start, tuning_row_start + tuning_height)
    col_slice = slice(tuning_width + tuning_h_pad, tuning_width + tuning_h_pad + tuning_width)
    plot_handles['learning'] = plot_handles['fig'].add_subplot(spec[row_slice, col_slice])

    tuning_row_start += (tuning_height + tuning_v_pad)
    row_slice = slice(tuning_row_start, tuning_row_start + tuning_height)
    col_slice = slice(0, tuning_width)
    plot_handles['anti_pursuit'] = plot_handles['fig'].add_subplot(spec[row_slice, col_slice])
    col_slice = slice(1*(tuning_width + tuning_h_pad), 1*(tuning_width + tuning_h_pad) + tuning_width)
    plot_handles['ax_schematic'] = plot_handles['fig'].add_subplot(spec[row_slice, col_slice])
    col_slice = slice(2*(tuning_width + tuning_h_pad), 2*(tuning_width + tuning_h_pad) + tuning_width)
    plot_handles['pursuit'] = plot_handles['fig'].add_subplot(spec[row_slice, col_slice])
    tuning_row_start += (tuning_height + tuning_v_pad)
    row_slice = slice(tuning_row_start, tuning_row_start + tuning_height)
    col_slice = slice(tuning_width + tuning_h_pad, tuning_width + tuning_h_pad + tuning_width)
    plot_handles['anti_learning'] = plot_handles['fig'].add_subplot(spec[row_slice, col_slice])

    # We want this to be offset a bit on its own
    row_slice = slice(tuning_row_start + 2*tuning_v_pad, tuning_row_start + tuning_height + 2*tuning_v_pad)
    col_slice = slice(2*(tuning_width + tuning_h_pad) + tuning_h_pad, 2*(tuning_width + tuning_h_pad) + tuning_width + tuning_h_pad)
    plot_handles['ax_inst_schem'] = plot_handles['fig'].add_subplot(spec[row_slice, col_slice])

    row_slice = slice(0, 3*tuning_height)
    col_slice = slice(3*(tuning_width + tuning_h_pad), total_cols)
    plot_handles['polar'] = plot_handles['fig'].add_subplot(spec[row_slice, col_slice], projection='polar')
    plot_handles['polar'].tick_params(axis='both', which='major', labelsize=6)
    # Then learning plots each get their own row
    fix_slice = slice(t_course_start, t_course_start+t_course_height)
    t_course_start += (t_course_height+t_course_v_pad)
    learn_slice = slice(t_course_start, t_course_start+t_course_height)
    t_course_start += (t_course_height+t_course_v_pad)
    behav_slice = slice(t_course_start, t_course_start+t_course_height)
    plot_handles['fix_fun'] = plot_handles['fig'].add_subplot(spec[fix_slice, :])
    plot_handles['learn_fun'] = plot_handles['fig'].add_subplot(spec[learn_slice, :])
    plot_handles['behav_fun'] = plot_handles['fig'].add_subplot(spec[behav_slice, :])

    # Set aspect ratio to be equal, so the arrows will appear as having the same length
    arrow_width = .05
    plot_handles['ax_schematic'].set_aspect('equal')
    plot_handles['ax_schematic'].quiver(0, 0, 1, 0, angles='xy', scale_units='xy', 
                                        scale=1, color=t_set_color_codes['pursuit'],
                                        width=arrow_width)  # x direction
    plot_handles['ax_schematic'].quiver(0, 0, 0, 1, angles='xy', scale_units='xy', 
                                        scale=1, color=t_set_color_codes['learning'],
                                        width=arrow_width)  # y direction
    plot_handles['ax_schematic'].quiver(0, 0, -1, 0, angles='xy', scale_units='xy', 
                                        scale=1, color=t_set_color_codes['anti_pursuit'],
                                        width=arrow_width)  # x direction
    plot_handles['ax_schematic'].quiver(0, 0, 0, -1, angles='xy', scale_units='xy', 
                                        scale=1, color=t_set_color_codes['anti_learning'],
                                        width=arrow_width)  # y direction
    # Set the x and y axis limits
    plot_handles['ax_schematic'].set_xlim(-1, 1)
    plot_handles['ax_schematic'].set_ylim(-1, 1)
    plot_handles['ax_schematic'].set_xticks([])
    plot_handles['ax_schematic'].set_yticks([])
    plot_handles['ax_schematic'].spines['left'].set_visible(False)
    plot_handles['ax_schematic'].spines['right'].set_visible(False)
    plot_handles['ax_schematic'].spines['bottom'].set_visible(False)
    plot_handles['ax_schematic'].spines['top'].set_visible(False)

    # Now setup aspect and lines for the instuction schematic
    plot_handles['ax_inst_schem'].set_aspect('equal')
    plot_handles['ax_inst_schem'].plot([0, 250, 270], [0, 0, 30], color="black", linewidth=2)
    # Draw an arrow from (250, 0) to (500, 375)
    plot_handles['ax_inst_schem'].annotate("", xy=(500, 375), xytext=(250, 0), 
                                            arrowprops=dict(facecolor='black', 
                                                                    edgecolor='black', 
                                                                    arrowstyle="->",
                                                                    linewidth=2))
                            
    plot_handles['ax_inst_schem'].set_xlim(0, 600)
    plot_handles['ax_inst_schem'].set_ylim(0, 475)
    plot_handles['ax_inst_schem'].set_xticks(np.arange(0, 600, 250))
    plot_handles['ax_inst_schem'].tick_params(axis='both', which='major', labelsize=6)
    plot_handles['ax_inst_schem'].set_yticks([])
    # plot_handles['ax_inst_schem'].spines['left'].set_visible(False)
    plot_handles['ax_inst_schem'].spines['right'].set_visible(False)
    # plot_handles['ax_inst_schem'].spines['bottom'].set_visible(False)
    plot_handles['ax_inst_schem'].spines['top'].set_visible(False)
    plot_handles['ax_inst_schem'].text(0.5, 1., 'Instruction trials', ha='center', va='bottom', 
                                       transform=plot_handles['ax_inst_schem'].transAxes, fontsize=8)

    return plot_handles

def get_plotting_rates(neuron, blocks, trial_sets, time_window, sigma=12.5, cutoff_sigma=4):

    fr, inds = neuron.get_firing_traces(time_window, blocks, trial_sets, return_inds=True)
    if len(fr) == 0:
        return np.array([]), np.array([]), np.array([])
    fr = np.nanmean(fr, axis=1)
    smooth_fr, _ = neuron.get_smooth_fr_by_block_gauss(blocks, time_window, sigma=sigma, cutoff_sigma=cutoff_sigma)

    return fr, smooth_fr, inds

def polar_arrow_and_annotate(ax_h, ang, mag, ann_text, color='black', linestyle="-", edgecolor="black", facecolor="black"):
    """ Helper function that draws our desired arrows in polar coordinates and annotes them with "ann_text" adjusted
    according to the quadrant in which the arrow falls. It is assumed that ax_h is a polar axes.
    """
    # Best alignment depends on the pref dir
    # Check in which quadrant the direction is
    if 0 <= ang < np.pi/2:  # First quadrant
        ha, va = 'left', 'bottom'
        text_h_scale = -np.pi / 10
        text_mag_scale = 1.1
    elif np.pi/2 <= ang < np.pi:  # Second quadrant
        ha, va = 'right', 'bottom'
        text_h_scale = -np.pi / 10
        text_mag_scale = 1.1
    elif np.pi <= ang < 3*np.pi/2:  # Third quadrant
        ha, va = 'right', 'top'
        text_h_scale = np.pi / 10
        text_mag_scale = 1.1
    else:  # Fourth quadrant
        ha, va = 'left', 'top'
        text_h_scale = np.pi / 10
        text_mag_scale = 1.1
    ax_h.annotate('', xy=(ang, mag), 
                                    xytext=(0, 0), 
                                    arrowprops=dict(facecolor=facecolor, 
                                                    edgecolor=edgecolor, 
                                                    width=0.5, 
                                                    headwidth=8,
                                                    linestyle=linestyle),
                                    zorder=10)
    ax_h.text(ang, text_mag_scale*mag, ann_text, color=color, fontsize=6,
                    ha=ha, va=va, weight='bold', zorder=10)

def plot_neuron_tuning_learning(neuron, blocks, trial_sets, fix_win, learn_win, sigma=12.5, 
                                cutoff_sigma=4, show_fig=False):
    """
    """
    # Append valid neuron trials to input trial_sets
    trial_sets = neuron.append_valid_trial_set(trial_sets)
    # Number of STDs of plotted values to set ylims by
    y_std = 4.
    t_title_pad = 0
    # Setup figure layout
    plot_handles = setup_axes()
    plot_handles['fig'].suptitle(f"Tuning and learning file: {neuron.session.fname}; unit: {neuron.name}", fontsize=12, y=.95)
    # First plot the basic tuning responses
    tune_trace_win = [-300, 1000]
    t_vals = np.arange(tune_trace_win[0], tune_trace_win[1])
    pol_t_win = [100, 175]
    pol_t_inds = (t_vals >= pol_t_win[0]) & (t_vals < pol_t_win[1])
    tune_fr_max = 0.
    tune_fr_min = np.inf
    polar_map = {"learning": np.pi/2, "anti_learning": 3*np.pi/2, "pursuit": 0, "anti_pursuit": np.pi}
    polar_vals = np.zeros((5, 2))
    n_tune = 0
    mean_fix_fr = 0.
    for tune_trial in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
        plot_handles[tune_trial].axvline(0., color='k', linestyle="--", linewidth=0.5, zorder=-1)
        # plot_handles[tune_trial].axvline(250., color='k', linestyle="--", linewidth=0.5, zorder=-1)
        fr = neuron.get_mean_firing_trace(tune_trace_win, blocks="StandTunePre", trial_sets=tune_trial)
        plot_handles[tune_trial].plot(t_vals, fr, color="k")
        curr_max_fr = np.nanmax(fr)
        curr_min_fr = np.nanmin(fr)
        if curr_max_fr > tune_fr_max:
            tune_fr_max = curr_max_fr
        if curr_min_fr < tune_fr_min:
            tune_fr_min = curr_min_fr
        polar_vals[n_tune, 0] = polar_map[tune_trial]
        polar_vals[n_tune, 1] = np.nanmean(fr[pol_t_inds])
        n_tune += 1
        mean_fix_fr += np.nanmean(fr[(t_vals >= fix_win[0]) & (t_vals < fix_win[1])])
    p_order = np.argsort(polar_vals[0:4, 0])
    polar_vals[0:4, :] = polar_vals[0:4, :][p_order, :]
    polar_vals[4, :] = polar_vals[0, :]
    plot_handles['polar'].plot(polar_vals[:, 0], polar_vals[:, 1], color=[.2, .2, .2], zorder=9)

    # Add the tuning vectors
    # Define the vector's magnitude and angle
    if isinstance(neuron, PurkinjeCell):
        neuron.set_optimal_pursuit_vector(pol_t_win, block='StandTunePre', cs_time_window=[30, 250])
        pref_dir = neuron.optimal_cos_vectors['StandTunePre']
        pref_mag = neuron.optimal_cos_funs['StandTunePre'](pref_dir)
        # Annotate the plot with an arrow at the CS angle and SS magnitude
        polar_arrow_and_annotate(plot_handles['polar'], 
                                 neuron.optimal_cos_vectors_cs['StandTunePre'], pref_mag, 
                                 "CS pref", color='black')
    else:
        neuron.set_optimal_pursuit_vector(pol_t_win, block='StandTunePre')
        pref_dir = neuron.optimal_cos_vectors['StandTunePre']
        pref_mag = neuron.optimal_cos_funs['StandTunePre'](pref_dir)
    # Annotate the plot with an arrow at the preferred SS angle and magnitude
    polar_arrow_and_annotate(plot_handles['polar'], 
                                 pref_dir, pref_mag, 
                                 "SS pref", color='black', facecolor='none', linestyle="--")
    # Attach a header for top tuning section to the polar axes
    plot_handles['polar'].text(np.radians(145), 4.5*pref_mag, "Four-direction Standard Tuning", color="black", fontsize=10,
                            ha="center", va="top", weight='bold', zorder=15)
        
    # Set all axes the same
    tune_fr_max = 5 * int(np.ceil(tune_fr_max / 5))
    tune_fr_min = 5 * int(np.floor(tune_fr_min / 5))
    plot_handles['polar'].set_rlim(0, tune_fr_max)
    tick_round = 10 if tune_fr_max > 50 else 5
    tick_steps = tick_round * int(np.ceil((tune_fr_max / 5) / tick_round))
    plot_handles['polar'].set_yticks(np.arange(0, tune_fr_max, tick_steps))
    for tune_trial in ["learning", "anti_pursuit", "pursuit", "anti_learning"]:
        plot_handles[tune_trial].tick_params(axis='both', which='major', labelsize=6)
        plot_handles[tune_trial].set_xticks(np.arange(-250, 1100, 250))
        plot_handles[tune_trial].axhline(mean_fix_fr/4, color='k', linestyle="--", linewidth=0.5, zorder=-1)
        plot_handles[tune_trial].set_ylim([tune_fr_min, tune_fr_max])
        plot_handles[tune_trial].fill_between(pol_t_win, tune_fr_min, tune_fr_max, color=[.8, .8, .8], alpha=1., zorder=-10)
        plot_handles[tune_trial].fill_between(learn_win, tune_fr_min, tune_fr_max, color=[.6, .6, .6], alpha=1., zorder=-10)
        plot_handles[tune_trial].set_title(f"{trial_strings[tune_trial]}", color=t_set_color_codes[tune_trial], fontsize=8, pad=t_title_pad)
        plot_handles[tune_trial].spines['top'].set_color(t_set_color_codes[tune_trial])
        plot_handles[tune_trial].spines['bottom'].set_color(t_set_color_codes[tune_trial])
        plot_handles[tune_trial].spines['left'].set_color(t_set_color_codes[tune_trial])
        plot_handles[tune_trial].spines['right'].set_color(t_set_color_codes[tune_trial])
        # Add in colored axes for polar plot
        plot_handles['polar'].plot([polar_map[tune_trial], polar_map[tune_trial]], [0, tune_fr_max], color=t_set_color_codes[tune_trial], alpha=1., linewidth=2, zorder=8)
        
    # Add labels
    plot_handles['ax_inst_schem'].fill_between(learn_win, 0, 600, color=[.6, .6, .6], alpha=1., zorder=-10)
    plot_handles['anti_learning'].set_xlabel("Time from target onset (ms)", fontsize=8)
    plot_handles['anti_pursuit'].set_ylabel("Firing rate (Hz)", fontsize=8)
    plot_handles['anti_pursuit'].annotate(f"Trial \n analysis \n window \n {learn_win}", xy=(learn_win[1], .05), xytext=(0.9*learn_win[1], -1.15), 
                                          textcoords=('data', 'axes fraction'),
                                          xycoords=('data', 'axes fraction'),
                                          fontsize=6,
                                          ha='center',
                                          arrowprops=dict(facecolor='black', 
                                                            edgecolor='black', 
                                                            arrowstyle="->",
                                                            linewidth=0.75))
    plot_handles['pursuit'].annotate(f"Polar \n tuning \n window \n {pol_t_win}", xy=(pol_t_win[0] + .25 * (pol_t_win[1] - pol_t_win[0]), .7), xytext=(-350, 1.4), 
                                          textcoords=('data', 'axes fraction'),
                                          xycoords=('data', 'axes fraction'),
                                          ha='center',
                                          fontsize=6,
                                          arrowprops=dict(facecolor='black', 
                                                            edgecolor='black', 
                                                            arrowstyle="->",
                                                            linewidth=0.75))
    
    # Plot each block separte so there is discontinuity between blocks
    fix_mean = 0.
    fix_std = 0.
    fix_n = 0.
    learn_mean = 0.
    learn_std = 0.
    learn_n = 0.
    behav_mean = 0.
    behav_std = 0.
    behav_n = 0.
    labels_found = []
    for b_name in blocks:
        # First plot behavior for this block
        # Get instructiont trials in learning blocks
        if b_name in ["Learning", "Washout"]:
            _, eyev_l, eye_inds = neuron.session.get_xy_traces(
                                        "eye velocity", learn_win, blocks=b_name,
                                        trial_sets="instruction", return_inds=True)
            if len(eyev_l) > 0:
                eyev_l = np.nanmean(eyev_l, axis=1)
                curr_plot = plot_handles['behav_fun'].scatter(eye_inds, eyev_l, color=[.5, .5, .5], s=5)
                if "instruction" not in labels_found:
                    curr_plot.set_label("instruction")
                    labels_found.append("instruction")
                if sigma*cutoff_sigma >= eyev_l.shape[0]:
                    # Just use block mean if it's shorter than trial win
                    smooth_eyev_l = np.full(eyev_l.shape, np.nanmean(eyev_l))
                else:
                    smooth_eyev_l = gauss_convolve(eyev_l, sigma, cutoff_sigma, pad_data=True)
                curr_plot = plot_handles['behav_fun'].plot(eye_inds, smooth_eyev_l, color='k', linewidth=1.5)
                if "smooth" not in labels_found:
                    curr_plot[0].set_label("smooth")
                    labels_found.append("smooth")
                behav_mean += np.nanmean(eyev_l) * eyev_l.shape[0]
                behav_std += np.nanstd(eyev_l) * eyev_l.shape[0]
                behav_n += eyev_l.shape[0]
            else:
                print(f"No instruction trials for block {b_name}")

        if "fix" in b_name.lower():
            # Cant plot eye velocity for fixation trials
            pass
        else:
            # Pursuit axis trials for all other block types
            for trial_type in ["anti_pursuit", "pursuit"]:
                _, eyev_l, eye_inds = neuron.session.get_xy_traces(
                                                "eye velocity", learn_win, blocks=b_name,
                                                trial_sets=trial_type, return_inds=True)
                if len(eyev_l) > 0:
                    eyev_l = np.nanmean(eyev_l, axis=1)
                    curr_plot = plot_handles['behav_fun'].scatter(eye_inds, eyev_l, color=t_set_color_codes[trial_type], s=5, zorder=10)
                    if trial_strings[trial_type] not in labels_found:
                        curr_plot.set_label(trial_strings[trial_type])
                        labels_found.append(trial_strings[trial_type])
                    behav_mean += np.nanmean(eyev_l) * eyev_l.shape[0]
                    behav_std += np.nanstd(eyev_l) * eyev_l.shape[0]
                    behav_n += eyev_l.shape[0]
                else:
                    print(f"No {trial_type} trials for block {b_name}")       


        # Check if this neuron has any data for this block
        valid_b_inds = neuron.session._parse_blocks_trial_sets([b_name], trial_sets)
        if len(valid_b_inds) > 0:
            fr, smooth_fr, inds = get_plotting_rates(neuron, b_name, trial_sets, fix_win, sigma=sigma, cutoff_sigma=cutoff_sigma)
            plot_handles['fix_fun'].scatter(inds, fr, color=[.5, .5, .5], s=5)
            plot_handles['fix_fun'].plot(inds, smooth_fr, color="k", linewidth=1.5)
            fix_mean += np.nanmean(fr) * fr.shape[0]
            fix_std += np.nanstd(fr) * fr.shape[0]
            fix_n += fr.shape[0]
        else:
            print(f"No neuron trials found for block {b_name}")

        if b_name in ["Learning", "Washout"]:
            fr, inds = neuron.get_firing_traces_fix_adj(learn_win, b_name, "instruction", fix_time_window=fix_win, 
                                                        sigma=sigma, cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                        rate_offset=0., return_inds=True)
            if len(fr) > 0:
                fr = np.nanmean(fr, axis=1)
                if sigma*cutoff_sigma >= fr.shape[0]:
                    # Just use block mean if it's shorter than trial win
                    smooth_fr = np.full(fr.shape, np.nanmean(fr))
                else:
                    smooth_fr = gauss_convolve(fr, sigma, cutoff_sigma, pad_data=True)
                plot_handles['learn_fun'].scatter(inds, fr, color=[.5, .5, .5], s=5)
                plot_handles['learn_fun'].plot(inds, smooth_fr, color='k', linewidth=1.5)
                learn_mean += np.nanmean(fr) * fr.shape[0]
                learn_std += np.nanstd(fr) * fr.shape[0]
                learn_n += fr.shape[0]
            else:
                print(f"No instruction trials for block {b_name}")

            # fr, inds = neuron.get_firing_traces(learn_win, b_name, "instruction", return_inds=True)
            # fr_fix, _ = neuron.get_firing_traces(fix_win, b_name, "instruction", return_inds=True)
            # if len(fr) > 0:
            #     fr = np.nanmean(fr, axis=1)
            #     fr_fix = np.nanmean(fr_fix, axis=1)
            #     fr -= fr_fix
            #     if sigma*cutoff_sigma >= fr.shape[0]:
            #         # Just use block mean if it's shorter than trial win
            #         smooth_fr = np.full(fr.shape, np.nanmean(fr))
            #     else:
            #         smooth_fr = gauss_convolve(fr, sigma, cutoff_sigma, pad_data=True)
            #     plot_handles['learn_fun'].scatter(inds, fr, color=[.75, .0, .1], s=5)
            #     plot_handles['learn_fun'].plot(inds, smooth_fr, color='r', linewidth=3)
            #     learn_mean += np.nanmean(fr) * fr.shape[0]
            #     learn_std += np.nanstd(fr) * fr.shape[0]
            #     learn_n += fr.shape[0]
            # else:
            #     print(f"No instruction trials for block {b_name}")

        if "fix" in b_name.lower():
            # So no fixation trials
            pass
        else:
            # Pursuit axis trials for all other block types
            for trial_type in ["anti_pursuit", "pursuit"]:
                # Scatter pursuit axis trials in different colors
                fr, inds = neuron.get_firing_traces_fix_adj(learn_win, b_name, trial_type, fix_time_window=fix_win, 
                                                            sigma=sigma, cutoff_sigma=cutoff_sigma, zscore_sigma=3.0, 
                                                            rate_offset=0., return_inds=True)
                if len(fr) > 0:
                    fr = np.nanmean(fr, axis=1)
                    plot_handles['learn_fun'].scatter(inds, fr, color=t_set_color_codes[trial_type], s=5, zorder=10)
                    learn_mean += np.nanmean(fr) * fr.shape[0]
                    learn_std += np.nanstd(fr) * fr.shape[0]
                    learn_n += fr.shape[0]
                else:
                    print(f"No {trial_type} trials for block {b_name}")

    behav_mean /= behav_n
    behav_std /= behav_n
    plot_handles['behav_fun'].set_title(f"Learning axis eye velocity by trial in time window {learn_win} ms", fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles['behav_fun'].set_xlabel("Trial number in session")
    plot_handles['behav_fun'].set_ylabel("Learning axis eye \n velocity (deg/s)", fontsize=8)
    plot_handles['behav_fun'].set_ylim([-y_std * behav_std, y_std * behav_std])
    plot_handles['behav_fun'].set_xlim([-1, len(neuron.session)+1])
    plot_handles['behav_fun'].axhline(0., color='k', linestyle="--", linewidth=0.5, zorder=-1)
    # # Add the labels that we found and the legend
    plot_handles['behav_fun'].legend(fontsize='x-small', borderpad=0.2, labelspacing=0.2, bbox_to_anchor=(.05, .65))
    fix_mean /= fix_n
    fix_std /= fix_n
    plot_handles['fix_fun'].text(.5, 1.18, "Responses Across Blocks and Trials", color="black", fontsize=10,
                                    ha="center", va="top", weight='bold', zorder=15, transform=plot_handles["fix_fun"].transAxes)
    plot_handles['fix_fun'].set_title(f"Fixation window rate by trial in time window {fix_win} ms", fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles['fix_fun'].set_ylabel("Fixation firing rate (Hz)", fontsize=8)
    plot_handles['fix_fun'].set_xticks([])
    # plot_handles['fix_fun'].yaxis.set_label_position("right")
    # plot_handles['fix_fun'].yaxis.tick_right()
    plot_handles['fix_fun'].set_ylim([fix_mean - y_std*fix_std, fix_mean + y_std*fix_std])
    plot_handles['fix_fun'].set_xlim([-1, len(neuron.session)+1])
    plot_handles['fix_fun'].axhline(fix_mean, color='k', linestyle="--", linewidth=0.5, zorder=-1)
    learn_mean /= fix_n
    learn_std /= fix_n
    plot_handles['learn_fun'].set_title(f"Pursuit response by trial in time window {learn_win} ms", fontsize=8, pad=t_title_pad, weight='bold')
    plot_handles['learn_fun'].set_ylabel("Learning window firing rate \n fixation subtracted (Hz)", fontsize=8)
    plot_handles['learn_fun'].set_xticks([])
    plot_handles['learn_fun'].set_ylim([learn_mean - y_std*learn_std, learn_mean + y_std*learn_std])
    plot_handles['learn_fun'].set_xlim([-1, len(neuron.session)+1])
    plot_handles['learn_fun'].axhline(0., color='k', linestyle="--", linewidth=0.5, zorder=-1)

    # Shade blocks
    n_block = 0
     # Create a blend of two transformations for specifying block text labels
    trans = transforms.blended_transform_factory(plot_handles["fix_fun"].transData, plot_handles["fix_fun"].transAxes)
    for b_name in blocks:
        if neuron.session.blocks[b_name] is None:
            continue
        gray_shade = [.8, .8, .8] if n_block % 2 == 0 else [.9, .9, .9]
        # text_h = .85 if n_block % 2 == 0 else .95
        # if b_name == "Washout":
        #     gray_shade = [.7, .75, 1.]
        for ax_h in ["behav_fun", "fix_fun", "learn_fun"]:
            ymin, ymax = plot_handles[ax_h].get_ylim()
            x_shade_win = [neuron.session.blocks[b_name][0], neuron.session.blocks[b_name][1]]
            plot_handles[ax_h].fill_between(x_shade_win, ymin, ymax, color=gray_shade, alpha=1., zorder=-10)
            if ax_h == "fix_fun":
                # Add in block names
                x_center = x_shade_win[0] + (x_shade_win[1] - x_shade_win[0])/2
                # ylims = plot_handles['fix_fun'].get_ylim()
                if b_name == "FixTunePre":
                    ha = "left"
                    x_center = 0.
                else:
                    ha = "center"
                if n_block % 2 == 0:
                    y_center = .94
                else:
                    y_center = .08
                label_text = block_strings[b_name].split(" ")
                plot_handles[ax_h].text(x_center, y_center, 
                                        "\n".join(label_text), ha=ha, va='center', 
                                        fontsize=5, color="black",
                                        transform=trans)
        n_block += 1

    if show_fig:
        plt.show()

    return plot_handles



        

