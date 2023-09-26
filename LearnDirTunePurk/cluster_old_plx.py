import numpy as np
import umap
from spikesorting_fullpursuit import sort, preprocessing
from spikesorting_fullpursuit.parallel.spikesorting_parallel import branch_pca_2_0



def branch_umap_cluster(neuron_labels, umap_scores, p_value_cut_thresh=0.01, n_random=100):
    """
    """
    neuron_labels_copy = np.copy(neuron_labels)
    clusters_to_check = [ol for ol in np.unique(neuron_labels_copy)]
    next_label = int(np.amax(clusters_to_check) + 1)
    while len(clusters_to_check) > 0:
        curr_clust = clusters_to_check.pop()
        curr_clust_bool = neuron_labels_copy == curr_clust
        clust_scores = umap_scores[curr_clust_bool, :]
        if clust_scores.shape[0] <= 1:
            # Only one spike so don't try to sort
            continue
        median_cluster_size = min(100, int(np.around(clust_scores.shape[0] / 1000)))

        # Re-cluster and sort using only clips from current cluster
        clust_labels = sort.initial_cluster_farthest(clust_scores, median_cluster_size, n_random=n_random)
        clust_labels = sort.merge_clusters(clust_scores, clust_labels,
                                            p_value_cut_thresh=p_value_cut_thresh)
        new_labels = np.unique(clust_labels)
        if new_labels.size > 1:
            # Found at least one new cluster within original so reassign labels
            for nl in new_labels:
                temp_labels = neuron_labels_copy[curr_clust_bool]
                temp_labels[clust_labels == nl] = next_label
                neuron_labels_copy[curr_clust_bool] = temp_labels
                clusters_to_check.append(next_label)
                next_label += 1

    return neuron_labels_copy


def umap_and_cluster(clips, p_value_cut_thresh=0.01, n_random=100):
    """
    """
    umap_scores = umap.UMAP().fit_transform(clips)
    umap_scores = np.float64(umap_scores)
    median_cluster_size = min(100, int(np.around(umap_scores.shape[0] / 1000)))
    neuron_labels = sort.initial_cluster_farthest(umap_scores, median_cluster_size, n_random=n_random)
    neuron_labels = sort.merge_clusters(umap_scores, neuron_labels,
                        split_only = False,
                        merge_only = False,
                        p_value_cut_thresh=p_value_cut_thresh)
    
    neuron_labels = branch_umap_cluster(neuron_labels, umap_scores, p_value_cut_thresh=p_value_cut_thresh)
    neuron_labels = sort.merge_clusters(umap_scores, neuron_labels,
                                        split_only = False,
                                        merge_only = False,
                                        p_value_cut_thresh=p_value_cut_thresh)

    sort.reorder_labels(neuron_labels)
    return neuron_labels, umap_scores


def cluster_clips(clips, settings={}):
    
    if 'add_peak_valley' not in settings:
        settings['add_peak_valley'] = False
    if 'use_rand_init' not in settings:
        settings['use_rand_init'] = True
    if 'check_components' not in settings:
        settings['check_components'] = 20
    if 'max_components' not in settings:
        settings['max_components'] = 2
    if 'p_value_cut_thresh' not in settings:
        settings['p_value_cut_thresh'] = 0.01
    if 'verbose' not in settings:
        settings['verbose'] = True
    if 'do_branch_PCA' not in settings:
        settings['do_branch_PCA'] = True

    median_cluster_size = min(100, int(np.around(clips.shape[0] / 1000)))
    if clips.shape[0] > 1:
        # MUST SLICE curr_chan_inds to get a view instead of copy
        scores = preprocessing.compute_pca(clips,
                    settings['check_components'], settings['max_components'], add_peak_valley=settings['add_peak_valley'],
                    curr_chan_inds=np.arange(0, clips.shape[1]))
        n_random = max(100, np.around(clips.shape[0] / 100)) if settings['use_rand_init'] else 0
        neuron_labels = sort.initial_cluster_farthest(scores, median_cluster_size, n_random=n_random)
        neuron_labels = sort.merge_clusters(scores, neuron_labels,
                            split_only = False,
                            p_value_cut_thresh=settings['p_value_cut_thresh'])

        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        if settings['verbose']: print("After first sort", curr_num_clusters.size, "different clusters", flush=True)
    else:
        neuron_labels = np.zeros(1, dtype=np.int64)
        curr_num_clusters = np.zeros(1, dtype=np.int64)
    if settings['verbose']: print("Currently", curr_num_clusters.size, "different clusters", flush=True)

    # Remove deviant clips before doing branch PCA to avoid getting clusters
    # of overlaps or garbage
    # keep_clips = preprocessing.cleanup_clusters(clips, neuron_labels)
    # clips = clips[keep_clips, :]

    # Single channel branch
    if curr_num_clusters.size > 1 and settings['do_branch_PCA']:
        neuron_labels = branch_pca_2_0(neuron_labels, clips,
                            np.arange(0, clips.shape[1]),
                            p_value_cut_thresh=settings['p_value_cut_thresh'],
                            add_peak_valley=settings['add_peak_valley'],
                            check_components=settings['check_components'],
                            max_components=settings['max_components'],
                            use_rand_init=settings['use_rand_init'],
                            method='pca')
        curr_num_clusters, n_per_cluster = np.unique(neuron_labels, return_counts=True)
        if settings['verbose']: print("After SINGLE BRANCH", curr_num_clusters.size, "different clusters", flush=True)

    neuron_labels = sort.merge_clusters(scores, neuron_labels,
                            split_only = False,
                            p_value_cut_thresh=settings['p_value_cut_thresh'])
    return neuron_labels
