# script contains function that source reconstruct 62 channel EEG data
# using MNE methods (e.g. eLORETA) or beamforming for time-frequency
# includes function for statistical analysis in source space: permutation t-test or cluster permutation test in source space
# also includes a function to plot contrasts in source space


# Author: Carina Forster
# email: forster@cbs.mpg.de

# load packages
import os.path as op
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy


from mne.datasets import fetch_fsaverage
import subprocess

# Specify the file path for which you want the last commit date
file_path = "D:\expecon_ms\\analysis_code\\eeg\\source\\source_reco.py"

last_commit_date = subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", file_path]).decode("utf-8").strip()

print("Last Commit Date for", file_path, ":", last_commit_date)

# load source space files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'

_oct = '6'

fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')
src_fname = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')

# Read the source space and the forward solution
src = mne.read_source_spaces(src_fname)
fwd = mne.read_forward_solution(fwd_dir)

# datapaths
dir_cleanepochs = Path('D:/expecon_ms/data/eeg/prepro_ica/clean_epochs_corr')
behavpath = Path('D:/expecon_ms/data/behav/behav_df/')

# save paths for beamforming
beamformer_path = Path('D:/expecon_ms/data/eeg/source/high_low_pre_beamformer')
figs_dics_path = Path('D:/expecon_ms/figs/manuscript_figures/figure6_tfr_contrasts/source')
# save paths for mne
mne_path = Path('D:/expecon_ms/data/eeg/source/high_low_pre_eLORETA')
save_dir_cluster_output = Path('D:/expecon_ms/figs/eeg/sensor/cluster_permutation_output')

IDlist = ['007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034','035', '036',
          '037', '038', '039','040', '041', '042', '043', '044','045', '046', '047', '048', '049']


def run_source_reco(dics=1, path=beamformer_path, drop_bads=True):

    """ run source reconstruction on epoched EEG data using eLoreta or DICS beamforming
    for frequency source analysis.
    input: dics: 1 for DICS beamforming, 0 for eLoreta
    path: path to save source estimates
    drop_bads: if True, bad epochs are dropped
    output: .stc files for each hemisphere that contain sourece reconstruction for 
    each participant: shape: verticesxtimepoints
    """

    for idx, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)

        # load cleaned epochs (after ica component rejection)
        epochs = mne.read_epochs(f'{dir_cleanepochs}'
                                 f'/P{subj}_epochs_after_ica_0.1Hzfilter-epo.fif')

        ids_to_delete = [10, 12, 13, 18, 26, 30, 32, 32, 39, 40, 40, 30]
        blocks_to_delete = [6, 6, 4, 3, 4, 3, 2, 3, 3, 2, 5, 6]

        # Check if the participant ID is in the list of IDs to delete
        if pd.unique(epochs.metadata.ID) in ids_to_delete:

            # Get the corresponding blocks to delete for the current participant
            participant_blocks_to_delete = [block for id_, block in
                                            zip(ids_to_delete, blocks_to_delete)
                                            if id_ == pd.unique(epochs.metadata.ID)]
            
            # Drop the rows with the specified blocks from the dataframe
            epochs = epochs[~epochs.metadata.block.isin(participant_blocks_to_delete)]
            
        # remove trials with rts >= 2.5 (no response trials) and trials with rts < 0.1
        epochs = epochs[epochs.metadata.respt1 >= 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # load behavioral data
        data = pd.read_csv(f'{behavpath}//prepro_behav_data.csv')

        subj_data = data[data.ID == idx+7]

        # get drop log from epochs
        drop_log = epochs.drop_log

        search_string = 'IGNORED'

        indices = [index for index, tpl in enumerate(drop_log) if tpl and search_string not in tpl]

        # drop bad epochs (too late recordings)
        if indices:
            epochs.metadata = subj_data.reset_index().drop(indices)
        else:
            epochs.metadata = subj_data

        # drop bad epochs
        if drop_bads:
            # drop epochs with abnormal strong signal (> 200 mikrovolts)
            epochs.drop_bad(reject=dict(eeg=200e-6))

        # epochs for high and low probability condition
        epochs_high = epochs[epochs.metadata.cue == 0.75]
        epochs_low = epochs[epochs.metadata.cue == 0.25]

        if dics == 1:
                
            # We are interested in the beta band
            freqs = np.arange(15, 25, 1)

            # Computing the cross-spectral density matrix for the beta frequency band, for
            # different time intervals.
            # csd for all epochs
            csd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-0.4, tmax=0)
            # csd for high prob.trials only
            csd_a = mne.time_frequency.csd_morlet(epochs_high, freqs, tmin=-0.4, tmax=0)
            # csd for low prob trials only
            csd_b = mne.time_frequency.csd_morlet(epochs_low, freqs, tmin=-0.4, tmax=0)
            #csd_baseline = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-1, tmax=-0.5)

            info = epochs.info

            # To compute the source power for a frequency band, rather than each frequency
            # separately, we average the CSD objects across frequencies.
            csd_a = csd_a.mean()
            csd_b = csd_b.mean()
            #csd_baseline = csd_baseline.mean()

            # Computing DICS spatial filters using the CSD that was computed on the entire
            # timecourse.

            filters = mne.beamformer.make_dics(info, fwd, csd, noise_csd=None,
                                pick_ori='max-power', reduce_rank=True, real_filter=True)

            # Applying DICS spatial filters separately to the CSD computed using the
            # baseline and the CSD computed during the ERS activity.
            source_power_a, freqs = mne.beamformer.apply_dics_csd(csd_a, filters)
            source_power_b, freqs = mne.beamformer.apply_dics_csd(csd_b, filters)

            source_power_a.save(f'{path}\high_beta_{subj}')
            source_power_b.save(f'{path}\low_beta_{subj}')

        else:

            # average epochs for mne
            evokeds_high = epochs_high.average()
            evokeds_low = epochs_low.average()

            # create noise covariance with a bias of data length

            noise_cov = create_noise_cov(evokeds_high.data.shape, evokeds_high.info)

            #mne.write_cov('covariance_prestim.cov', noise_cov)

            inv_op = mne.minimum_norm.make_inverse_operator(evokeds_high.info, fwd, noise_cov)

            # loose=1.0, fixed=False
            evokeds_high.set_eeg_reference(projection=True)  # needed for inverse modeling

            conditiona_stc = mne.minimum_norm.apply_inverse(evokeds_high, inv_op, lambda2=0.05,
                                                            method='eLORETA', pick_ori='normal')
            
            inv_op = mne.minimum_norm.make_inverse_operator(evokeds_low.info, fwd, noise_cov)

            evokeds_low.set_eeg_reference(projection=True)  # needed for inverse modelingS

            conditionb_stc = mne.minimum_norm.apply_inverse(evokeds_low, inv_op, lambda2=0.05,
                                                            method='eLORETA', pick_ori='normal')
            
            conditiona_stc.save(f'{path}/high_{subj}', overwrite=True)
            conditionb_stc.save(f'{path}/low_{subj}', overwrite=True)


def create_source_contrast_array(path=beamformer_path):

    """function loads source estimates per participant and contrasts them, 
    beforestoring the contrast in a numpy array.
    input: path to source estimates
    output: contrast in
    shape of numpy array:
    participantsxverticesxtimepoints
    """

    stc_all, stc_high_all, stc_low_all = [], [], []

    for subj in IDlist:

        stc_high = mne.read_source_estimate(f'{path}\\high_beta_{subj}')
        stc_low = mne.read_source_estimate(f'{path}\\low_beta_{subj}')

        stc_diff = stc_high.data-stc_low.data

        stc_high_all.append(stc_high.data)
        stc_low_all.append(stc_low.data)
        stc_all.append(stc_diff)

    stc_low = np.array(stc_low_all)
    stc_high = np.array(stc_high_all)
    stc_array = np.array(stc_all)

    data = stc_array

    return stc_array


def extract_timecourse_from_label():

    """function that extracts the timecourse from a label in source space"""

    # Get labels for FreeSurfer 'aparc' cortical parcellation with 75 labels/hemi
    labels_parc = mne.read_labels_from_annot('fsaverage', parc='aparc.a2009s', subjects_dir=subjects_dir)

    stc_all, stc_high_all, stc_low_all = [], [], []

    for idx, subj in enumerate(IDlist):

        stc_high = mne.read_source_estimate(f'{path}high_{subj}')
        stc_low = mne.read_source_estimate(f'{path}low_{subj}')

        for stc in [stc_high, stc_low]:

            # extract activity in from source label
            # S1
            postcentral_gyrus = mne.extract_label_time_course(
                [stc], labels_parc[55], src, allow_empty=True)
            # S2
            G_front_inf_Opercular_rh = mne.extract_label_time_course(
                [stc], labels_parc[25], src, allow_empty=True)
            # ACC
            G_cingul_Post_dorsal_rh = mne.extract_label_time_course(
                [stc], labels_parc[19], src, allow_empty=True)


def spatio_temporal_source_test(data=None, n_perm=10000, jobs=-1, 
                                save_path_source_figs=figs_dics_path):

    """function that runs a cluster based permutation test over space and time
    data: 3D numpy array: participantsxspacextime
    n_perm: how many permutations for cluster test
    jobs: how many parallel GPUs should be used
    out: cluster output"""

    print('Computing adjacency.')

    adjacency = mne.spatial_src_adjacency(src)

    # Note that X needs to be a multi-dimensional array of shape
    # observations (subjects) × time × space, so we permute dimensions

    # read data in for expecon 2
    data = np.load("D:\expecon_ms\data\eeg\source\high_low_pre_beamformer\expecon2\source_beta_highlow.npy")

    X = np.transpose(data, [0, 2, 1])

    X_mean = np.mean(X[:,:,:], axis=1)

    # mean over time and permutation test to get sign. vertices
    t,p,h = mne.stats.permutation_t_test(X_mean)

    # mean over time and participants and plot contrast in source space
    X_avg = np.mean(X[:,:,:], axis=(0,1))

    # put contrast or p values in source space
    fsave_vertices = [s['vertno'] for s in src]
    stc = mne.SourceEstimate(X_avg, tmin=-0.5, tstep=0.0001, vertices = fsave_vertices, subject='fsaverage')

    brain = stc.plot(
        hemi='rh', views='medial', subjects_dir=subjects_dir,
        subject = 'fsaverage', time_viewer=True,
        background='white')

    brain.save_image(f'{save_path_source_figs}/t_values_high_low_rh_beamformer_dics.png')

    # Here we set a cluster forming threshold based on a p-value for
    # the cluster based permutation test.
    # We use a two-tailed threshold, the "1 - p_threshold" is needed
    # because for two-tailed tests we must specify a positive threshold.

    p_threshold = 0.05
    df = (len(IDlist)-2) - 1  # degrees of freedom for the test
    t_threshold = scipy.stats.distributions.t.ppf(1 - p_threshold / 2, df=df)

    # Now let's actually do the clustering. This can take a long time...

    print('Clustering.')

    T_obs, clusters, cluster_p_values, H0 = clu = \
        mne.stats.spatio_temporal_cluster_1samp_test(X[:,:,:], adjacency=adjacency, 
                                                     threshold=t_threshold,
                                                     n_jobs=jobs, n_permutations=n_perm)

    return clu


def plot_cluster_output(clu=None):

    """function that plots the cluster output of the cluster permutation test
    input: cluster output
    output: plot of cluster output"""

    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_idx = np.where(clu[2] < 0.05)[0]
    good_clusters = [clu[1][idx] for idx in good_clusters_idx]
    print(min(clu[2]))

    print('Visualizing clusters.')

    # Now let's build a convenient representation of our results, where consecutive
    # cluster spatial maps are stacked in the time dimension of a SourceEstimate
    # object. This way by moving through the time dimension we will be able to see
    # subsequent cluster maps.
    fsave_vertices = [s['vertno'] for s in src]

    stc_all_cluster_vis = mne.stats.summarize_clusters_stc(clu,
                                                vertices=fsave_vertices,
                                                subject='fsaverage', p_thresh=0.1)
    
    # Let's actually plot the first "time point" in the SourceEstimate, which
    # shows all the clusters, weighted by duration.

    # blue blobs are for condition A < condition B, red for A > B

    brain = stc_all_cluster_vis.plot(
        hemi='rh', views='lateral', subjects_dir=subjects_dir,
        time_label='temporal extent (ms)', size=(800, 800),
        smoothing_steps=5, time_viewer=False,
        background='white', transparent=True, colorbar=False)
    
    brain.save_image("D:\expecon_ms\data\eeg\source\cluster_rh_lateral.png")


################################ Helper functions ################################


def create_noise_cov(data_size, data_info):
    """
    Computes identity noise covariance with a bias of data length
    Method is by Mina Jamshidi Idaji (minajamshidi91@gmail.com)
    :param tuple data_size: size of original data (dimensions - 1D)
    :param mne.Info data_info: info that corresponds to the original data
    :returns: (mne.Covariance) - noise covariance for further source reconstruction
    """

    data1 = np.random.normal(loc=0.0, scale=1.0, size=data_size)
    raw1 = mne.io.RawArray(data1, data_info)

    return mne.compute_raw_covariance(raw1, tmin=0, tmax=None)