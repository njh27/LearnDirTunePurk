import numpy as np
import scipy.io
import pickle
import matplotlib.pyplot as plt
import umap
import os
import csv
from spikesorting_fullpursuit import sort
from spikesorting_fullpursuit.analyze_spike_timing import zero_symmetric_ccg



def matlab_purkinje_maestro_struct_2_python(matlab_file, sampling_rate=40000):
    """ Input the old plexon Matlab file name where the data structures are stored. This then loads them
    into numpy datastructures and converts those to a list of dictionaries as output.
    """
    matlab_data = scipy.io.loadmat(matlab_file)
    data_list = []
    n_spikes = 0
    printed = [False]
    for trial in range(0, matlab_data['PurkinjeMaestroStruct'].shape[1]):
        # Go through each trial
        trial_dict = {'TrialName': matlab_data['PurkinjeMaestroStruct'][0, trial]['TrialName'][0],
                      'DIOtherTimes': np.float64(matlab_data['PurkinjeMaestroStruct'][0, trial]['DIOtherTimes']),
                      'TimeSpikeChanStart': np.float64(matlab_data['PurkinjeMaestroStruct'][0, trial]['TimeSpikeChanStart'][0]),
                      'TimeSpikeChanStop': np.float64(matlab_data['PurkinjeMaestroStruct'][0, trial]['TimeSpikeChanStop'][0]),
                      'MaestroFilename': matlab_data['PurkinjeMaestroStruct'][0, trial][1][0],
                     }
        # Make these updated names
        trial_dict['TrialName'] = trial_dict['TrialName'].replace("left", "lt")
        trial_dict['TrialName'] = trial_dict['TrialName'].replace("down", "dn")
        trial_dict['TrialName'] = trial_dict['TrialName'].replace("right", "rt")
        trial_dict['SpikeWaves'] = []
        trial_dict['SpikeIndices'] = []
        for n_neuron in range(0, matlab_data['PurkinjeMaestroStruct'][0, trial]['SimpleSpikes'][0].size):
            # Go over all possible neurons and just recombine any data since we will sort again
            if ( (matlab_data['PurkinjeMaestroStruct'][0, trial]['SimpleSpikes'][0, n_neuron].size == 0)
                and (matlab_data['PurkinjeMaestroStruct'][0, trial]['ComplexSpikes'][0, n_neuron].size == 0) ):
                continue
            else:
                if not printed[n_neuron]:
                    print(f"Found simple spikes for unit number {n_neuron}")
                    printed[n_neuron] = True
                    printed.append(False)
            trial_dict['SpikeWaves'].append(np.vstack((np.float64(matlab_data['PurkinjeMaestroStruct'][0, trial]['SimpleWaves'][0, n_neuron].T),
                                                       np.float64(matlab_data['PurkinjeMaestroStruct'][0, trial]['ComplexWaves'][0, n_neuron].T)))
                                            )
            ss_axis = None if matlab_data['PurkinjeMaestroStruct'][0, trial]['SimpleSpikes'][0, n_neuron].size == 0 else 1
            cs_axis = None if matlab_data['PurkinjeMaestroStruct'][0, trial]['ComplexSpikes'][0, n_neuron].size == 0 else 1
            unit_spikes = [np.float64(np.squeeze(matlab_data['PurkinjeMaestroStruct'][0, trial]['SimpleSpikes'][0, n_neuron], axis=ss_axis)),
                           np.float64(np.squeeze(matlab_data['PurkinjeMaestroStruct'][0, trial]['ComplexSpikes'][0, n_neuron], axis=cs_axis))]
            unit_spikes = np.hstack([u_spks for u_spks in unit_spikes if u_spks.size > 0])
            if unit_spikes.size > 0:
                trial_dict['SpikeIndices'].append(unit_spikes)
        if len(trial_dict['SpikeIndices']) > 0:
            # Combine all the trial data
            trial_dict['SpikeWaves'] = np.vstack([n_waves for n_waves in trial_dict['SpikeWaves'] if n_waves.size > 0])
            trial_dict['SpikeIndices'] = np.hstack(trial_dict['SpikeIndices'])
            # Adjust the spike times to global index numbers
            trial_dict['SpikeIndices'] += trial_dict['TimeSpikeChanStart']
            trial_dict['SpikeIndices'] *= (sampling_rate / 1000)
            trial_dict['SpikeIndices'] = np.int64(np.around(trial_dict['SpikeIndices']))

            # Assign a unique spike ID for each spike
            trial_dict['SpikeNums'] = np.arange(n_spikes, n_spikes + trial_dict['SpikeIndices'].size)
            n_spikes += trial_dict['SpikeIndices'].size
        else:
            trial_dict['SpikeNums'] = []
            
        data_list.append(trial_dict)

    return data_list



def load_mat_maestro(directory_name, check_existing=True, save_data=False,
                        save_name=None, return_loaded_existing=False):
        """Load a directory of maestro files

        Loads a complete directory of maestro files as a list of dictionaries.
        The filenames are assumed to be in the of the form *.[0-9][0-9][0-9][0-9].
        This function attempts to load the files in order of their suffix.
        check_existing true will search for existing pickle files starting with the
        input save_name and then the input directory name by standard naming
        default conventions.
        """
        loaded_existing = False
        if check_existing:
            if save_name is not None:
                try:
                    if (save_name[-7:] != ".pickle") and (save_name[-4:] != ".pkl"):
                        save_name = save_name + ".pickle"
                    with open(save_name, 'rb') as fp:
                        data = pickle.load(fp)
                    loaded_existing = True
                    if return_loaded_existing:
                        return data, loaded_existing
                    else:
                        return data
                except FileNotFoundError:
                    pass
            try:
                with open(directory_name + ".pickle", 'rb') as fp:
                    data = pickle.load(fp)
                loaded_existing = True
                if return_loaded_existing:
                    return data, loaded_existing
                else:
                    return data
            except FileNotFoundError:
                pass
            try:
                with open(directory_name + "_maestro.pickle", 'rb') as fp:
                    data = pickle.load(fp)
                loaded_existing = True
                if return_loaded_existing:
                    return data, loaded_existing
                else:
                    return data
            except FileNotFoundError:
                pass
            print("Could not find existing Maestro file. Recomputing from scratch.")
        if directory_name[-4:] != ".mat":
            directory_name = directory_name + ".mat"
        data = get_maestro_from_mat(directory_name)

        if save_name is not None:
            # save_data = True
            if (save_name[-7:] != ".pickle") and (save_name[-4:] != ".pkl"):
                save_name = save_name + ".pickle"
        if save_data:
            if save_name is None:
                save_name = directory_name.split("/")[-1]
                root_name = directory_name.split("/")[0:-1]
                save_name = save_name + "_maestro.pickle"
                save_name = "".join(x + "/" for x in root_name) + save_name
            print("Saving Maestro trial data as:", save_name)
            with open(save_name, 'wb') as fp:
                pickle.dump(data, fp, protocol=-1)

        if return_loaded_existing:
            return data, loaded_existing
        else:
            return data
        
def get_maestro_from_mat(filename):
    """ For the old Yan files that the Maestro reader cannot load, create the maestro_data from the matlab file directly.
    """
    matlab_data = scipy.io.loadmat(filename)
    mat_maestro_data = []
    targ_names = ['rmfixation1', 'rmpursuit1']
    for trial in range(0, matlab_data['PurkinjeMaestroStruct'].shape[1]):
        # Go through each trial
        header_dict = {'name': str(matlab_data['PurkinjeMaestroStruct'][0, trial]['TrialName'][0]),
                    '_num_saved_scans': matlab_data['PurkinjeMaestroStruct'][0, trial]['Key']['nScansSaved'][0][0][0][0],
                    'num_scans': matlab_data['PurkinjeMaestroStruct'][0, trial]['Key']['nScansSaved'][0][0][0][0],
                    'vstab_sliding_window': matlab_data['PurkinjeMaestroStruct'][0, trial]['Key']['iVStabWinLen'][0][0][0][0],
                    'version': matlab_data['PurkinjeMaestroStruct'][0, trial]['Key']['version'][0][0][0][0],
                    'display_framerate': matlab_data['PurkinjeMaestroStruct'][0, trial]['Key']['d_framerate'][0][0][0][0],
                    'UsedStab': False,
                    }
        header_dict['name'] = header_dict['name'].replace("down", "dn")
        header_dict['name'] = header_dict['name'].replace("right", "rt")
        header_dict['name'] = header_dict['name'].replace("left", "lt")
        targets = []
        for n_targ in range(0, matlab_data['PurkinjeMaestroStruct'][0, trial]['Targets']['targnums'][0][0][0].size):
            targ_dict = {'target_name': targ_names[n_targ],
                        }
            targets.append(targ_dict)
        events = []
        for DIO_num in range(0, 14):
            chan_events = []
            for n_event in range(0, matlab_data['PurkinjeMaestroStruct'][0, trial]['DIOtherTimes'].shape[0]):
                if matlab_data['PurkinjeMaestroStruct'][0, trial]['DIOtherTimes'][n_event, 0] == DIO_num:
                    chan_events.append(matlab_data['PurkinjeMaestroStruct'][0, trial]['DIOtherTimes'][n_event, 1] / 1000)
            events.append(chan_events)
            
        trial_dict = {'filename': matlab_data['PurkinjeMaestroStruct'][0, trial]['FileName'][0],
                    'header': header_dict,
                    'targets': targets,
                    'events': events,
                    'horizontal_eye_position': matlab_data['PurkinjeMaestroStruct'][0, trial]['EyePos'][:, 0],
                    'vertical_eye_position': matlab_data['PurkinjeMaestroStruct'][0, trial]['EyePos'][:, 1],
                    'horizontal_eye_velocity': matlab_data['PurkinjeMaestroStruct'][0, trial]['EyeVel'][:, 0],
                    'vertical_eye_velocity': matlab_data['PurkinjeMaestroStruct'][0, trial]['EyeVel'][:, 1],
                    'horizontal_target_position': matlab_data['PurkinjeMaestroStruct'][0, trial]['Targets']['hpos'][0][0],
                    'vertical_target_position': matlab_data['PurkinjeMaestroStruct'][0, trial]['Targets']['vpos'][0][0],
                    'horizontal_target_velocity': matlab_data['PurkinjeMaestroStruct'][0, trial]['Targets']['hvel'][0][0],
                    'vertical_target_velocity': matlab_data['PurkinjeMaestroStruct'][0, trial]['Targets']['vvel'][0][0],
                    'compressed_target': False,
                    }
        
        # Need to remove transients and the stupid oscillations in velocity
        trial_dict['horizontal_target_velocity'][np.abs(trial_dict['horizontal_target_velocity']) > 100] = 0.
        move_inds = trial_dict['horizontal_target_velocity'] != 0.
        trial_dict['horizontal_target_velocity'][move_inds] = np.around(np.mean(trial_dict['horizontal_target_velocity'][move_inds]))
        
        trial_dict['vertical_target_velocity'][np.abs(trial_dict['vertical_target_velocity']) > 100] = 0.
        move_inds = trial_dict['vertical_target_velocity'] != 0.
        trial_dict['vertical_target_velocity'][move_inds] = np.around(np.mean(trial_dict['vertical_target_velocity'][move_inds]))
        
        mat_maestro_data.append(trial_dict)

    return mat_maestro_data
    


def umap_and_cluster_by_seg(data_list, t_start=0, t_step=200, t_overlap=50, max_trial=np.inf, p_value_cut_thresh=0.05, branch_umap=True,
                            n_random=1000, show_plots=True):
    """ Goes through the converted MatLab data in "data_list" that is output by "matlab_purkinje_maestro_struct_2_python".
    Data are broken into segments of trial size "t_step" and umap clustered separately. The data are then pieced back 
    together using the spike num IDs and a fulloutput of unique spike labels, indices, and waveforms is output.
    """
    # Start the sorting by segment
    all_waveforms = []
    all_nums = []
    all_labels = []
    all_indices = []
    t_stop = min(t_start + t_step, len(data_list))
    while t_start < len(data_list):
        print(f"Clustering segment from trials {t_start}-{t_stop}")
        seg_waveforms = []
        seg_nums = []
        seg_inds = []
        # Grab all trial data in this segment
        for trial in range(t_start, t_stop):
            if len(data_list[trial]['SpikeIndices']) == 0:
                # No spikes this trial
                continue
            seg_waveforms.append(data_list[trial]['SpikeWaves'])
            seg_nums.append(data_list[trial]['SpikeNums'])
            seg_inds.append(data_list[trial]['SpikeIndices'])
        # Combine trial data
        seg_waveforms = np.vstack(seg_waveforms)
        all_waveforms.append(seg_waveforms)
        seg_nums = np.hstack(seg_nums)
        all_nums.append(seg_nums)            
        seg_inds = np.hstack(seg_inds)
        all_indices.append(seg_inds)
        
        # Do the umap sorting
        neuron_labels, umap_scores = umap_and_cluster(seg_waveforms, p_value_cut_thresh=p_value_cut_thresh, 
                                                      branch_umap=branch_umap, n_random=n_random)
        all_labels.append(neuron_labels)
        
        if show_plots:
            for id in np.unique(neuron_labels):
                plt.scatter(umap_scores[neuron_labels == id, 0], umap_scores[neuron_labels == id, 1])
                print(np.count_nonzero(neuron_labels == id))
            plt.show()
        
        if ( (t_stop >= len(data_list)) or (t_stop > max_trial) ):
            break
        t_start = (t_stop - t_overlap)
        t_stop = t_start + t_step
        if (len(data_list) - t_stop) < t_step:
            t_stop = len(data_list)
        if t_stop > max_trial:
            t_stop = max_trial + 1

    # Start combining the labels across segments
    # Start with the current labeling scheme in first segment,
    # which is assumed to be ordered from 0-N (as output by sorter)
    # Find first segment with labels and start there
    start_seg = 0
    while start_seg < len(all_labels):
        if len(all_labels[start_seg]) == 0:
            start_seg += 1
            continue
        real_labels = np.unique(all_labels[start_seg]).tolist()
        next_real_label = max(real_labels) + 1
        if len(real_labels) > 0:
            break
        start_seg += 1

    start_new_seg = False
    # Go through each segment as the "current segment" and set the labels
    # in the next segment according to the scheme in current
    for curr_seg in range(start_seg, len(all_labels) - 1):
        next_seg = curr_seg + 1
        if len(all_labels[curr_seg]) == 0:
            # curr_seg is now failed next seg of last iteration
            start_new_seg = True
            if self.verbose: print("skipping seg", curr_seg, "with no spikes")
            continue
        if start_new_seg:
            # Update new labels dictionary
            old_n_labels = np.unique(all_labels[curr_seg])
            # Map all units in this segment to new real labels
            all_labels[curr_seg] += next_real_label # Keep out of range
            old_n_labels += next_real_label
            for nl in old_n_labels:
                # Add these new labels to real_labels for tracking
                real_labels.append(nl)
                next_real_label += 1
        if len(all_labels[next_seg]) == 0:
            # No units sorted in NEXT segment so start fresh next segment
            start_new_seg = True
            if self.verbose: print("skipping_seg", curr_seg, "because NEXT seg has no spikes")
            continue
            
        # Make 'fake_labels' for next segment that do not overlap with
        # the current segment and make a work space for the next
        # segment labels so we can compare curr and next without
        # losing track of original labels
        fake_labels = np.copy(np.unique(all_labels[next_seg]))
        fake_labels += next_real_label # Keep these out of range of the real labels
        fake_labels = fake_labels.tolist()
        next_label_workspace = np.copy(all_labels[next_seg])
        next_label_workspace += next_real_label
        
        # Merge test all mutually closest clusters and track any labels
        # in the next segment (fake_labels) that do not find a match.
        # These are assigned a new real label.
        leftover_labels = [x for x in fake_labels]
        main_labels = [x for x in real_labels]
        previously_compared_pairs = []
        while len(main_labels) > 0 and len(leftover_labels) > 0:
            max_intersect = -1
            best_pair = None
            for ml in main_labels:
                m_nums = all_nums[curr_seg][all_labels[curr_seg] == ml]
                for ll in leftover_labels:
                    n_intersect = len(np.intersect1d(m_nums, all_nums[next_seg][all_labels[next_seg] == ll]))
                    if n_intersect > max_intersect:
                        max_intersect = n_intersect
                        best_pair = [ml, ll]
            
            # Choose next seg spikes based on original fake label workspace
            fake_select = next_label_workspace == best_pair[1]
            # Update new labels dictionary
            old_label = np.unique(all_labels[next_seg][fake_select])
            # Update actual next segment label data with same labels
            # used in curr_seg
            all_labels[next_seg][fake_select] = best_pair[0]
            if old_label.size > 1:
                raise RuntimeError("Found too many labels to update label dictionary!")
            else:
                old_label = old_label[0]
            leftover_labels.remove(best_pair[1])
            main_labels.remove(best_pair[0])

        # Assign units in next segment that do not match any in the
        # current segment a new real label
        for ll in leftover_labels:
            ll_select = next_label_workspace == ll
            # Update new labels dictionary
            old_label = np.unique(all_labels[next_seg][ll_select])
            all_labels[next_seg][ll_select] = next_real_label
            if old_label.size > 1:
                raise RuntimeError("Found too many labels to update label dictionary!")
            else:
                old_label = old_label[0]
            real_labels.append(next_real_label)
            next_real_label += 1

        # If we made it here then we are not starting a new seg
        start_new_seg = False

    # It is possible to leave loop without checking last seg in the
    # event it is a new seg
    if start_new_seg and len(all_labels[-1]) > 0:
        # Update new labels dictionary FOR THIS CHANNEL ONLY!
        old_n_labels = np.unique(all_labels[-1])
        # Map all units in this segment to new real labels
        # Seg labels start at zero, so just add next_real_label. This
        # is last segment for this channel so no need to increment
        all_labels[-1] += next_real_label
        real_labels.extend((old_n_labels + next_real_label).tolist())

    # Combine all the data and remove overlap duplicates
    combined_labels = np.hstack(all_labels)
    combined_nums = np.hstack(all_nums)
    combined_waves = np.vstack(all_waveforms)
    combined_spk_inds = np.hstack(all_indices)
    unq_nums, unq_inds = np.unique(combined_nums, return_index=True)
    unq_labels = combined_labels[unq_inds]
    unq_waves = combined_waves[unq_inds, :]
    unq_spk_inds = combined_spk_inds[unq_inds]

    # Make sure everything is still in temporal sorted order
    spike_order = np.argsort(unq_spk_inds)
    unq_labels = unq_labels[spike_order]
    unq_spk_inds = unq_spk_inds[spike_order]
    unq_waves = unq_waves[spike_order, :]

    return unq_labels, unq_spk_inds, unq_waves


def branch_umap_cluster(neuron_labels, clips, p_value_cut_thresh=0.01, n_random_min=100):
    """
    """
    neuron_labels_copy = np.copy(neuron_labels)
    clusters_to_check = [ol for ol in np.unique(neuron_labels_copy)]
    next_label = int(np.amax(clusters_to_check) + 1)
    while len(clusters_to_check) > 0:
        curr_clust = clusters_to_check.pop()
        curr_clust_bool = neuron_labels_copy == curr_clust

        clust_clips = clips[curr_clust_bool, :]
        if clust_clips.ndim == 1:
            clust_clips = np.expand_dims(clust_clips, 0)
        if clust_clips.shape[0] <= 1:
            # Only one spike so don't try to sort
            continue
        median_cluster_size = min(100, int(np.around(clust_clips.shape[0] / 1000)))

        print("Starting branch cluster")
        # Re-cluster and sort using only clips from current cluster
        clust_scores = umap.UMAP().fit_transform(clust_clips)
        clust_scores = np.float64(clust_scores)
        n_random = max(n_random_min, np.around(clust_clips.shape[0] / 100))
        clust_labels = sort.initial_cluster_farthest(clust_scores, median_cluster_size, n_random=n_random)
        clust_labels = sort.merge_clusters(clust_scores, clust_labels,
                                            p_value_cut_thresh=p_value_cut_thresh,
                                            match_cluster_size=True, check_splits=True)
        new_labels = np.unique(clust_labels)
        if new_labels.size > 1:
            print(f"Found {new_labels.size} new clusters!")
            # Found at least one new cluster within original so reassign labels
            for nl in new_labels:
                temp_labels = neuron_labels_copy[curr_clust_bool]
                temp_labels[clust_labels == nl] = next_label
                neuron_labels_copy[curr_clust_bool] = temp_labels
                clusters_to_check.append(next_label)
                next_label += 1
        else:
            print("Found zero new clusters!")

    return neuron_labels_copy


def umap_and_cluster(clips, p_value_cut_thresh=0.01, n_random=1000, branch_umap=True, clip_labels=None, merge_only=False):
    """
    """
    umap_scores = umap.UMAP().fit_transform(clips, y=clip_labels)
    umap_scores = np.float64(umap_scores)
    median_cluster_size = min(100, int(np.around(umap_scores.shape[0] / 1000)))
    neuron_labels = sort.initial_cluster_farthest(umap_scores, median_cluster_size, n_random=n_random)
    neuron_labels = sort.merge_clusters(umap_scores, neuron_labels,
                        split_only = False,
                        merge_only = merge_only,
                        p_value_cut_thresh=p_value_cut_thresh,
                        match_cluster_size=True, check_splits=True)
    
    if branch_umap:
        neuron_labels = branch_umap_cluster(neuron_labels, clips, p_value_cut_thresh=p_value_cut_thresh, n_random_min=n_random)

    sort.reorder_labels(neuron_labels)
    return neuron_labels, umap_scores


def old_data_list_spikes_to_viz(save_labels, save_neuron_label, filename, neuron_labels, spike_indices,
                                save_fname=None, channel_ids=0, sampling_rate=40000):
    if len(save_neuron_label) != len(save_labels):
        raise ValueError("Must input a neuron label name for each neuron saved to output!")
    if not isinstance(channel_ids, int):
        if len(channel_ids) != len(save_labels):
            raise ValueError("Must input a channel number ID integer for each unit or a single scalar applied to all units")
    else:
        channel_ids = [x for x in range(0, len(save_labels))]
    
    if save_fname is None:
        save_fname = "neurons_" + fname + "_viz.pkl"
    save_fname = save_fname.rstrip(".pickle")
    if save_fname[-4:] != ".pkl":
        save_fname = save_fname + ".pkl"
    # Use the default required NeuroViz keys
    nv_keys = ['filename__',
               'channel_id__',
               'spike_indices_channel__',
               'sampling_rate__']

    nv_neurons = []
    for n_ind, label in enumerate(save_labels):
        viz_dict = {}
        if save_neuron_label[n_ind] == "PC":
            if len(label) != 2:
                print(f"Input label {label} at index {n_ind} is a PC but does not have 2 inputs for simple and complex spikes")
            if np.count_nonzero(neuron_labels == label[0]) < np.count_nonzero(neuron_labels == label[1]):
                # Label[0] is the CS
                viz_dict['cs_spike_indices__'] = np.uint32(spike_indices[neuron_labels == label[0]])
                viz_dict['spike_indices__'] = np.uint32(spike_indices[neuron_labels == label[1]])
            else:
                # Label[1] is the CS
                viz_dict['cs_spike_indices__'] = np.uint32(spike_indices[neuron_labels == label[1]])
                viz_dict['spike_indices__'] = np.uint32(spike_indices[neuron_labels == label[0]])
            viz_dict['type__'] = 'NeurophysToolbox.PurkinjeCell'
            viz_dict['label'] = "pc"
        elif save_neuron_label[n_ind] in ["putative_cs", "CS"]:
            viz_dict['spike_indices__'] = np.uint32(spike_indices[neuron_labels == label]) + 1
            viz_dict['type__'] = 'NeurophysToolbox.ComplexSpikes'
            viz_dict['label'] = 'putative_cs'
        elif save_neuron_label[n_ind] in ['putPC', 'putative_pc']:
            viz_dict['spike_indices__'] = np.uint32(spike_indices[neuron_labels == label]) + 1
            viz_dict['type__'] = 'Neuron'
            viz_dict['label'] = 'putative_pc'
        elif save_neuron_label[n_ind] in ["putative_basket", "MLI"]:
            viz_dict['spike_indices__'] = np.uint32(spike_indices[neuron_labels == label]) + 1
            viz_dict['type__'] = 'Neuron'
            viz_dict['label'] = 'MLI'
        elif save_neuron_label[n_ind] in ["putative_mf", "MF"]:
            viz_dict['spike_indices__'] = np.uint32(spike_indices[neuron_labels == label]) + 1
            viz_dict['type__'] = 'Neuron'
            viz_dict['label'] = 'MF'
        elif save_neuron_label[n_ind] in ["putative_golgi", "GC"]:
            viz_dict['spike_indices__'] = np.uint32(spike_indices[neuron_labels == label]) + 1
            viz_dict['type__'] = 'Neuron'
            viz_dict['label'] = 'GC'
        elif save_neuron_label[n_ind] in ["putative_ubc", "UBC"]:
            viz_dict['spike_indices__'] = np.uint32(spike_indices[neuron_labels == label]) + 1
            viz_dict['type__'] = 'Neuron'
            viz_dict['label'] = 'UBC'
        elif save_neuron_label[n_ind] in ["putative_stellate", "SC"]:
            viz_dict['spike_indices__'] = np.uint32(spike_indices[neuron_labels == label]) + 1
            viz_dict['type__'] = 'Neuron'
            viz_dict['label'] = 'SC'
        elif save_neuron_label[n_ind] in ["putative_granule", "GR"]:
            viz_dict['spike_indices__'] = np.uint32(spike_indices[neuron_labels == label]) + 1
            viz_dict['type__'] = 'Neuron'
            viz_dict['label'] = 'GR'
        elif save_neuron_label[n_ind] is None:
            viz_dict['spike_indices__'] = np.uint32(spike_indices[neuron_labels == label]) + 1
            viz_dict['type__'] = 'Neuron'
            viz_dict['label'] = None
        else:
            raise ValueError("Unrecognized neuron label {0}.".format(save_neuron_label[n_ind]))

        for nv_key in nv_keys:
            if nv_key == 'filename__':
                viz_dict['filename__'] = filename
            elif nv_key == 'channel_id__':
                viz_dict['channel_id__'] = np.array(channel_ids[n_ind], dtype=np.uint16) + 1
            elif nv_key == 'sampling_rate__':
                viz_dict['sampling_rate__'] = sampling_rate
            elif nv_key == 'spike_indices_channel__':
                viz_dict['spike_indices_channel__'] = (channel_ids[n_ind] + 1) * np.ones(spike_indices.shape, dtype=np.uint16)
            else:
                raise ValueError("Unrecognized key", nv_key, "for NeuroViz dictionary")
            
        nv_neurons.append(viz_dict)

    # Need to save protocol 3 to be compatiblel with Julia
    with open(save_fname, 'wb') as fp:
        pickle.dump(nv_neurons, fp, protocol=3)
    print("Saved NeuroViz file:", save_fname)


def find_CS_ISI_violations(cs_label, neuron_labels, spike_indices, refractory_inds=80):
    """ Goes through all CS spike indices and finds any subsequent CS spikes that occur
    within less than "refrectory_inds". Assumes spike_indices is IN SORTED ORDER!
    """
    cs_select = neuron_labels == cs_label
    cs_spike_indices = spike_indices[cs_select]
    if cs_spike_indices.shape[0] < 2:
        # Can't be violations with under 2 spikes
        return np.zeros((spike_indices.shape[0], ), dtype='bool')
    bad_bool = np.zeros((spike_indices.shape[0], ), dtype='bool')
    bad_cs_bool = np.zeros((cs_spike_indices.shape[0], ), dtype='bool')
    cs_ind = 0
    while cs_ind < (cs_spike_indices.shape[0] - 1):
        check_cs_ind = cs_ind + 1
        while check_cs_ind < (cs_spike_indices.shape[0]):
            if cs_spike_indices[check_cs_ind] < cs_spike_indices[cs_ind]:
                raise ValueError("Input complex spikes are NOT IN SORTED ORDER!")
            if (cs_spike_indices[check_cs_ind] - cs_spike_indices[cs_ind]) < refractory_inds:
                # Violation
                bad_cs_bool[check_cs_ind] = True
                check_cs_ind += 1
            else:
                break
        cs_ind = check_cs_ind

    bad_bool[cs_select] = bad_cs_bool

    return bad_bool


def merge_labels(labels_to_merge, neuron_labels):
    """
    """
    if len(labels_to_merge) <= 1:
        print("Need more than 1 label to merge")
        return neuron_labels
    labels_to_merge = sorted(labels_to_merge)
    for label in labels_to_merge[1:]:
        # Label everything into lowest label number
        neuron_labels[neuron_labels == label] = labels_to_merge[0]
    return neuron_labels


def plot_pairwise_ccgs(neuron_labels, spike_indices, spike_waves):
    total_labels = np.unique(neuron_labels)
    print(f"Total labels: ")
    print(total_labels)
    bin_edges = np.linspace(spike_indices[0], spike_indices[-1], 101)
    for r_ind in range(0, total_labels.size):
        ref_unit = total_labels[r_ind]
        for u_ind in range(r_ind, total_labels.size):
            unit_num = total_labels[u_ind]
            print(f"CCG with ref unit {ref_unit} vs. unit {unit_num}")
            counts, bins = np.histogram(spike_indices[neuron_labels == ref_unit], bins=bin_edges)
            counts = 40000 * counts / (bin_edges[1] - bin_edges[0])
            plt.plot(bins[1:], counts)
            counts, bins = np.histogram(spike_indices[neuron_labels == unit_num], bins=bin_edges)
            counts = 40000 * counts / (bin_edges[1] - bin_edges[0])
            plt.plot(bins[1:], counts)
            plt.show()
            
            plt.plot(np.mean(spike_waves[neuron_labels == ref_unit, :], axis=0))
            plt.plot(np.mean(spike_waves[neuron_labels == unit_num, :], axis=0))
            plt.show()
            counts, time_axis = zero_symmetric_ccg(spike_indices[neuron_labels == ref_unit],
                                                                            spike_indices[neuron_labels == unit_num], 
                                                                            50*40, 40)
            center_ind = np.nonzero([time_axis == 0])[1][0]
            if ref_unit == unit_num:
                counts[center_ind] = 0
            plt.bar(time_axis, counts, width=1)
            plt.axvline(0, color='k')
            plt.axvline(10, color='k')
            plt.axvline(-10, color='k')
            plt.axvline(5, color='k')
            plt.axvline(-5, color='k')
            ag = plt.gcf()
            
            ag.set_size_inches(20, 15)
            plt.show()


def plot_ref_ccgs(ref_unit, neuron_labels, spike_indices, spike_waves, comp_labels=None):
    if comp_labels is None:
        total_labels = np.unique(neuron_labels)
    else:
        total_labels = comp_labels
    print(f"Ref unit has {np.count_nonzero(neuron_labels == ref_unit)} spikes")
    bin_edges = np.linspace(spike_indices[0], spike_indices[-1], 101)
    for u_ind in range(0, len(total_labels)):
        unit_num = total_labels[u_ind]
        print(f"CCG with ref unit {ref_unit} vs. unit {unit_num}")
        counts, bins = np.histogram(spike_indices[neuron_labels == ref_unit], bins=bin_edges)
        counts = 40000 * counts / (bin_edges[1] - bin_edges[0])
        plt.plot(bins[1:], counts)
        counts, bins = np.histogram(spike_indices[neuron_labels == unit_num], bins=bin_edges)
        counts = 40000 * counts / (bin_edges[1] - bin_edges[0])
        plt.plot(bins[1:], counts)
        plt.show()
        
        plt.plot(np.mean(spike_waves[neuron_labels == ref_unit, :], axis=0))
        plt.plot(np.mean(spike_waves[neuron_labels == unit_num, :], axis=0))
        plt.show()
        counts, time_axis = zero_symmetric_ccg(spike_indices[neuron_labels == ref_unit],
                                                                        spike_indices[neuron_labels == unit_num], 
                                                                        50*40, 40)
        center_ind = np.nonzero([time_axis == 0])[1][0]
        if ref_unit == unit_num:
            counts[center_ind] = 0
        plt.bar(time_axis, counts, width=1)
        plt.axvline(0, color='k')
        plt.axvline(10, color='k')
        plt.axvline(-10, color='k')
        plt.axvline(5, color='k')
        plt.axvline(-5, color='k')
        ag = plt.gcf()
        
        ag.set_size_inches(20, 15)
        plt.show()

    
def find_closest_units(neuron_labels, spike_indices, spike_waves):
    total_labels = np.unique(neuron_labels)
    bin_edges = np.linspace(spike_indices[0], spike_indices[-1], 101)
    lookup_closest = []
    bin_edges = np.linspace(spike_indices[0], spike_indices[-1], 101)
    for r_ind in range(0, total_labels.size):
        ref_unit = total_labels[r_ind]
        for u_ind in range(r_ind+1, total_labels.size):
            unit_num = total_labels[u_ind]
            template_r = np.mean(spike_waves[neuron_labels == ref_unit, :], axis=0)
            template_u = np.mean(spike_waves[neuron_labels == unit_num, :], axis=0)
            
            # norm_r = np.correlate(template_r, template_r, "full")
            # norm_u = np.correlate(template_u, template_u, "full")
            cross_r_u = np.correlate(template_r, template_u, "full")
            norm_factor = np.linalg.norm(template_r) * np.linalg.norm(template_u)
            norm_similarity = -1*np.amax(cross_r_u) / norm_factor

            # norm_similarity = -1*np.amax(cross_r_u)

            # counts_r, bins = np.histogram(spike_indices[neuron_labels == ref_unit], bins=bin_edges)
            # counts_r = 40000 * counts_r / (bin_edges[1] - bin_edges[0])
            # counts_u, bins = np.histogram(spike_indices[neuron_labels == unit_num], bins=bin_edges)
            # counts_u = 40000 * counts_u / (bin_edges[1] - bin_edges[0])

            # dist_r_u = np.sum( (template_r - template_u) ** 2)
            # counts_r_u = np.sum( (counts_r - counts_u) ** 2)
            # mean_fr_r = np.mean(counts_r[counts_r > 0])
            # mean_fr_u = np.mean(counts_u[counts_u > 0])
            # means_r_u = (mean_fr_r - mean_fr_u) ** 2

            # norm_similarity = dist_r_u 

            # All scores positive, closest to 0 is most similar
            # norm_similarity = np.abs(1 - (2*np.amax(cross_r_u) / (np.amax(norm_r) + np.amax(norm_u))))

            lookup_closest.append((norm_similarity, set([ref_unit, unit_num])))

    lookup_closest = sorted(lookup_closest, key=lambda x: x[0])
    return lookup_closest


def add_old_plx_events(maestro_data, purk_mat_fname, maestro_pl2_chan_offset=1, remove_bad_inds=False):

    mat_data_list = matlab_purkinje_maestro_struct_2_python(purk_mat_fname)
    print("Syncing old PLX events for file {0} with {1}.".format(maestro_data[0]['filename'].rsplit("/")[-1].rsplit(".")[0], purk_mat_fname))

    # Not sure if this will ever happen
    if len(mat_data_list) != len(maestro_data):
        raise ValueError(f"Maestro data has different number of trials than the matlab data!")

    # Sort the array of event channel numbers and event times by timestamps and set initial index to 0
    remove_ind = []
    for trial in range(0, len(maestro_data)):
        _, maestro_data[trial]['pl2_file'] = os.path.split(purk_mat_fname)
        maestro_data[trial]['pl2_synced'] = False
        # Check if Plexon strobe and Maestro file names match
        if mat_data_list[trial]['TrialName'] in maestro_data[trial]['header']['name']:
            # First make numpy array of all event numbers and their times for this trial
            # and sort it according to time of events
            maestro_events_times = [[], []]
            for p in range(0, len(maestro_data[trial]['events'])):
                maestro_events_times[0].extend(maestro_data[trial]['events'][p])
                maestro_events_times[1].extend([p + 1] * len(maestro_data[trial]['events'][p]))
            maestro_events_times = np.array(maestro_events_times)
            maestro_events_times = maestro_events_times[:, np.argsort(maestro_events_times[0, :])]
            try:
                # Now build a matching event array with Plexon events.  Include 2 extra event
                # slots for the start and stop XS2 events, which are not in Maestro file.  This
                # array will be used for searching the Plexon data to find correct output
                old_plx_trial_events = np.zeros((2, len(maestro_events_times[1])))
                old_plx_trial_events[0, :] = mat_data_list[trial]['DIOtherTimes'][:, 1]
                old_plx_trial_events[1, :] = mat_data_list[trial]['DIOtherTimes'][:, 0]
            except:
                print(trial)
                print(old_plx_trial_events.shape)
                print(mat_data_list[trial]['DIOtherTimes'][:, 1].shape)
                print(mat_data_list[trial]['DIOtherTimes'][:, 1])
                raise

            # Check trial duration according to plexon XS2 and Maestro file
            # old_plx_duration = mat_data_list[trial]['TimeSpikeChanStop'][0] - mat_data_list[trial]['TimeSpikeChanStart'][0]
            old_plx_duration = mat_data_list[trial]['TimeSpikeChanStop'] - mat_data_list[trial]['TimeSpikeChanStart']
            maestro_duration = maestro_data[trial]['header']['_num_saved_scans']
            if np.abs(old_plx_duration - maestro_duration) > 40.0:
                print("WARNING: difference between recorded plx trial duration {0} and maestro trial duration {1} is over 40 ms. This could mean XS2 inital pulse was delayed and unreliable.".format(pl2_duration, maestro_duration))
            elif np.abs(old_plx_duration - maestro_duration) > 2.0:
                # This happens due to the shitty REB system on some Yoda files. XS2 stop is correct so just set PL2 start based on this and Maestro duration
                # mat_data_list[trial]['TimeSpikeChanStart'][0] = mat_data_list[trial]['TimeSpikeChanStop'][0] - maestro_duration
                mat_data_list[trial]['TimeSpikeChanStart'] = mat_data_list[trial]['TimeSpikeChanStop'] - maestro_duration
            # Save start and stop for output
            # maestro_data[trial]['plexon_start_stop'] = (mat_data_list[trial]['TimeSpikeChanStart'][0], mat_data_list[trial]['TimeSpikeChanStop'][0])
            maestro_data[trial]['plexon_start_stop'] = (mat_data_list[trial]['TimeSpikeChanStart'], mat_data_list[trial]['TimeSpikeChanStop'])

            # Again, only include Plexon events that were observed in Maestro by looking
            # through all Maestro events and finding their counterparts in Plexon
            # according to the mapping defined in maestro_pl2_chan_offset
            maestro_data[trial]['plexon_events'] = [[] for _ in range(0, len(maestro_data[trial]['events']))]
            for event_ind, event_num in enumerate(maestro_events_times[1, :]):
                event_num = np.int64(event_num)
                
                if old_plx_trial_events[1, event_ind] != (event_num + maestro_pl2_chan_offset):
                    raise RuntimeError(f"Trial event number {event_num} in Maestro data does not match plx data on trial {trial}")

                # Compare Maestro and Plexon inter-event times, Need to convert Maestro times to ms
                if event_ind == 0:
                    # Check first event against the XS2 trial start event
                    aligment_difference = abs(1000 * (maestro_events_times[0, event_ind]) - old_plx_trial_events[0, 0])
                else:
                    aligment_difference = abs(1000 * (maestro_events_times[0, event_ind] - maestro_events_times[0, event_ind-1]) -
                                                     (old_plx_trial_events[0, event_ind] - old_plx_trial_events[0, event_ind-1]))
                
                if ( (aligment_difference > 0.1) and (event_num != 1) ):
                    # Event 1 is not aligned in Yoda because the REB system didn't work
                    remove_ind.append(trial)
                    print("Plexon and Maestro inter-event intervals do not match within 0.1 ms for trial {0} and event number {1}.".format(trial, event_num))
                    print(f"Alignment difference = {aligment_difference}")
                    maestro_data[trial]['pl2_synced'] = False
                    break
                    # raise ValueError("Plexon and Maestro inter-event intervals do not match within 0.1 ms for trial {0} and event number {1}.".format(trial, event_num))
                else:
                    maestro_data[trial]['pl2_synced'] = True

                # Re-stack plexon events for output by lists of channel number so they match Maestro data in maestro_data[trial]['events']
                if event_num > 0:
                    maestro_data[trial]['plexon_events'][event_num - 1].append(old_plx_trial_events[0, event_ind])
                else:
                    # This is probably an error
                    print('FOUND A START STOP CODE IN TRIAL EVENTS!? PROBABLY AN ERROR')
        else:
            raise ValueError(f"Trial names do not match on trial {trial}")

    if ( (len(remove_ind) > 0) and (remove_bad_inds) ):
        # Want these inds in reverse order
        remove_ind.reverse()
        if remove_bad_inds:
            for index in remove_ind:
                print("Trial {} did not have matching Plexon and Maestro events and was removed".format(index))
                del maestro_data[index]

    return maestro_data


def maestro_plx_sync_file(raw_maestro_dir, fname, mat_file, fname_csv):
    """
    """
    mat_data_list = matlab_purkinje_maestro_struct_2_python(mat_file)
    trial_files = sorted(os.listdir(os.path.join(raw_maestro_dir, fname)))
    _, extension = os.path.splitext(fname_csv)
    if extension != ".csv":
        fname_csv = fname_csv + ".csv"
    # Open a CSV file in write mode
    with open(fname_csv, "w", newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Write header
        csv_writer.writerow(["Maestro Filename", "Plexon Alignment Time"])
        
        for trial, trial_file in zip(mat_data_list, trial_files):
            if trial['MaestroFilename'] != trial_file:
                raise ValueError(f"Trial name {trial['MaestroFilename']} does not match the indexed maestro file {trial_file}!")
            csv_writer.writerow([trial_file, trial['TimeSpikeChanStart']])

    print(f"Saved file {fname_csv}")
