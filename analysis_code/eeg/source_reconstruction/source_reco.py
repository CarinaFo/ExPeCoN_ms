# script contains function that source reconstruct 62 channel EEG data
# using MNE or beamforming for time-frequency
# includes function for statistical analysis in source space: permutation, cluster permutation
# also includes a function to plot contrasts in source space


# Author: Carina Forster
# last update: 14.07.2023

import os
# load packages
import os.path as op

# plotting
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
# freesurfer
from mne.datasets import fetch_fsaverage
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

save_dir_cluster_output = r"D:\expecon_ms\figs\eeg\sensor\cluster_permutation_output"
dir_cleanepochs = "D:\\expecon_ms\\data\\eeg\\prepro_ica\\clean_epochs_iclabel"
behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'

IDlist = ['007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034','035', '036',
          '037', '038', '039','040', '041', '042', '043', '044','045', '046', '047', '048', '049']


def run_source_reco(dics=1):

    """ run source reconstruction on epoched EEG data using eLoreta or DICS beamforming
    for frequency source analysis
    output: .stc files for each hemisphere that contain sourece reconstruction for 
    each participant: shape: verticesxtimepoints
    """

    for idx, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)

        epochs = mne.read_epochs(f'{dir_cleanepochs}\P{subj}_epochs_after_ic-label-epo.fif')

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

        #high vs. low condition
        epochs_high = epochs[epochs.metadata.cue == 0.75]
        epochs_low = epochs[epochs.metadata.cue == 0.25]

        # hit vs. miss trials
        #epochs_high = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1))]
        #epochs_low = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 0))]
        
        #average and crop in prestimulus window
        evokeds_high = epochs_high.average()
        evokeds_low = epochs_low.average()

        if dics == 1:
                
            # We are interested in the beta band. Define a range of frequencies, using a
            # log scale, from 12 to 30 Hz.

            freqs = np.logspace(np.log10(12), np.log10(30), 8)

            # Computing the cross-spectral density matrix for the beta frequency band, for
            # different time intervals.
            csd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-0.4, tmax=0)
            csd_a = mne.time_frequency.csd_morlet(epochs_high, freqs, tmin=-0.4, tmax=0)
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

            os.chdir("D:\expecon_ms\data\eeg\source\high_low_pre_beamformer")

            source_power_a.save('\high_beta_' + subj)
            source_power_b.save('\low_beta_' + subj)

        else:

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
            
            os.chdir("D:\expecon_ms\data\eeg\source\high_low_pre")

            conditiona_stc.save('high_' + subj, overwrite=True)
            conditionb_stc.save('low_' + subj, overwrite=True)

def create_source_contrast_array():

    """function loads source estimates per participant and contrasts them, before
    storing the contrast in a numpy array: 
    shape participantsxverticesxtimepoints"""

    # Get labels for FreeSurfer 'aparc' cortical parcellation with 75 labels/hemi
    labels_parc = mne.read_labels_from_annot('fsaverage', parc='aparc.a2009s', subjects_dir=subjects_dir)

    stc_all, stc_high_all, stc_low_all = [], [], []

    for idx, subj in enumerate(IDlist):

        # skip those participants
        if subj == '040' or subj == '045':
            continue

        os.chdir("D:\expecon_ms\data\eeg\source\high_low_pre_eLoreta")
        #os.chdir("D:\expecon_ms\data\eeg\source\high_low_pre_beamformer")
        #os.chdir("D:\expecon_ms\data\eeg\source\hit_miss_pre_beamformer")

        stc_high = mne.read_source_estimate('high_' + subj)
        stc_low = mne.read_source_estimate('low_' + subj)

        # extract activity in from source label
        # S1
        #postcentral_gyrus = mne.extract_label_time_course(
         #   [stc], labels_parc[55], src, allow_empty=True)
        # S2
        #G_front_inf_Opercular_rh = mne.extract_label_time_course(
          #  [stc], labels_parc[25], src, allow_empty=True)
        # ACC
        #G_cingul_Post_dorsal_rh = mne.extract_label_time_course(
         #   [stc], labels_parc[19], src, allow_empty=True)

        #stc_high = mne.read_source_estimate('high_fixed_' + subj)
        #stc_low = mne.read_source_estimate('low_fixed_' + subj)

        stc_diff = stc_high.data-stc_low.data

        stc_high_all.append(stc_high.data)
        stc_low_all.append(stc_low.data)
        stc_all.append(stc_diff)

    stc_low = np.array(stc_low_all)
    stc_high = np.array(stc_high_all)
    stc_array = np.array(stc_all)

    data = stc_array

    return stc_array

def spatio_temporal_source_test(data=None, n_perm=10000, jobs=10):
    """function that runs a cluster based perutation test over space and time
    data: 3D numpy array: participantsxspacextime
    n_perm: how many permutations for cluster test
    jobs: how many parallel GPUs should be used
    out: cluster output"""

    print('Computing adjacency.')

    adjacency = mne.spatial_src_adjacency(src)

    # Note that X needs to be a multi-dimensional array of shape
    # observations (subjects) × time × space, so we permute dimensions

    X = np.transpose(data, [0, 2, 1])

    X_mean = np.mean(X[:,:,:], axis=1)

    # mean over time and permutation test to get sign. vertices
    H,p,t = mne.stats.permutation_t_test(X_mean)

    # mean over time and participants and plot contrast in source space
    X_avg = np.mean(X[:,:,:], axis=(0,1))

    # put contrast or p values in source space
    fsave_vertices = [s['vertno'] for s in src]
    stc = mne.SourceEstimate(p, tmin=-0.5, tstep=0.0001, vertices = fsave_vertices, subject='fsaverage')

    brain = stc.plot(
        hemi='rh', views='medial', subjects_dir=subjects_dir,
        subject = 'fsaverage', time_viewer=False,
        background='grey')

    brain.save_image(r"D:\expecon_ms\figs\eeg\source\avg_high_low_dics_perm_beta_rh_medial.png")

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
        
    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_idx = np.where(clu[2] < 0.05)[0]
    good_clusters = [clu[1][idx] for idx in good_clusters_idx]
    print(min(cluster_p_values))

    print('Visualizing clusters.')

    # Now let's build a convenient representation of our results, where consecutive
    # cluster spatial maps are stacked in the time dimension of a SourceEstimate
    # object. This way by moving through the time dimension we will be able to see
    # subsequent cluster maps.
    fsave_vertices = [s['vertno'] for s in src]

    stc_all_cluster_vis = mne.stats.summarize_clusters_stc(clu,
                                                vertices=fsave_vertices,
                                                subject='fsaverage', p_thresh=0.05)
    
    # Let's actually plot the first "time point" in the SourceEstimate, which
    # shows all the clusters, weighted by duration.

    # blue blobs are for condition A < condition B, red for A > B

    brain = stc_all_cluster_vis.plot(
        hemi='rh', views='lateral', subjects_dir=subjects_dir,
        time_label='temporal extent (ms)', size=(800, 800),
        smoothing_steps=5, time_viewer=False,
        background='white', transparent=True, colorbar=False)
    
    brain.save_image("D:\expecon_ms\data\eeg\source\cluster_rh_lateral.png")


def cluster_perm_space_time(perm=10000, tmin=-0.5, tmax=0):

    """ this function runs a cluster permutation test over electrodes and timepoints
    in sensor space and plots the output and saves it
    input: perm: how many permutations for cluster test
    tmin: crop the epochs at this time in seconds
    tmax: crop the data until this time in seconds 
    out: saves the cluster figures as svg and png files """

    all_trials, trials_removed = [], []

    evokeds_low_all, evokeds_high_all = [], []

    for idx, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)
        # skip those participants
        if subj == '040' or subj == '045':
            continue

        # load cleaned epochs
        os.chdir("D:\expecon_ms\data\eeg\prepro_ica\clean_epochs")

        epochs = mne.read_epochs('P' + subj + '_epochs_after_ica-epo.fif')

        # Remove 7 blocks with hitrates < 0.2 or > 0.8

        if subj == '010':
            epochs = epochs[epochs.metadata.block != 6]
        if subj == '012':
            epochs = epochs[epochs.metadata.block != 6]
        if subj == '026':
            epochs = epochs[epochs.metadata.block != 4]
        if subj == '030':
            epochs = epochs[epochs.metadata.block != 3]
        if subj == '032':
            epochs = epochs[epochs.metadata.block != 2]
            epochs = epochs[epochs.metadata.block != 3]
        if subj == '039':
            epochs = epochs[epochs.metadata.block != 3]
        
        # remove trials with rts >= 2.5 (no response trials) and trials with rts < 0.1
        before_rt_removal = len(epochs.metadata)

        epochs = epochs[epochs.metadata.respt1 > 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]
        # some weird trigger stuff going on?
        epochs = epochs[epochs.metadata.trial != 1]
        
        #save n trials per participant
        all_trials.append(len(epochs.metadata))

        trials_removed.append(before_rt_removal - len(epochs.metadata))

        #high vs. low condition
        epochs_high = epochs[(epochs.metadata.cue == 0.75)]
        epochs_low = epochs[(epochs.metadata.cue == 0.25)]

        #average and crop in prestimulus window
        evokeds_high = epochs_high.average().crop(tmin, tmax)
        evokeds_low = epochs_low.average().crop(tmin, tmax)

        evokeds_high_all.append(evokeds_high)
        evokeds_low_all.append(evokeds_low)

    # get grand average over all subjects for plotting the results later

    a_gra = mne.grand_average(evokeds_high_all)
    b_gra = mne.grand_average(evokeds_low_all)

    high = np.array([h.data for h in evokeds_high_all])
    low = np.array([l.data for l in evokeds_low_all])

    X = [h.data-l.data for h,l in zip(evokeds_high_all, evokeds_low_all)]

    X = np.transpose(X, [0, 2, 1])

    ch_adjacency,_ = mne.channels.find_ch_adjacency(epochs.info, ch_type='eeg')

    #threshold_tfce = dict(start=0, step=0.1)

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X[:,:,:], n_permutations=perm,
                                                                                    adjacency=ch_adjacency,
                                                                                    )

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0] # times where something significant happened

    print(len(good_cluster_inds))
    print(cluster_p_values)

    # now plot the significant cluster(s)
    a = 'high'
    b= 'low'

    # configure variables for visualization
    colors = {a: "crimson", b: 'steelblue'}

    # organize data for plotting
    # instead of grand average we could use the evoked data per subject so that we can plot CIs
    grand_average = {a: a_gra, b: b_gra}

    # # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for t stat
        t_map = T_obs[time_inds, ...].mean(axis=0)

        # get signals at the sensors contributing to the cluster
        sig_times = a_gra.times[time_inds]

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

        # plot average test statistic and mark significant sensors
        t_evoked = mne.EvokedArray(t_map[:, np.newaxis], a_gra.info, tmin=0)

        t_evoked.plot_topomap(times=0, mask=mask, axes=ax_topo,
                            show=False,
                                colorbar=False, mask_params=dict(markersize=10))
        image = ax_topo.images[0]

        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel('Averaged t-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes('right', size='300%', pad=1.2)
        title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
        if len(ch_inds) > 1:
            title += "s (mean)"

        mne.viz.plot_compare_evokeds(grand_average, title=title, picks=ch_inds, axes=ax_signals,
                                        colors=colors, show=False, ci=True,
                                        split_legend=True, legend='lower right', truncate_yaxis='auto')

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                                    color='green', alpha=0.3)

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        os.chdir(save_dir_cluster_output)
        plt.savefig('cluster' + str(i_clu) + '.svg')
        plt.savefig('cluster' + str(i_clu) + '.png')
        plt.show()


def extract_cluster_and_plot_source_contrast(clusters=None, good_cluster_inds=None,
                                             evoked_dataa=None, evoked_datab=None,
                                             evokeds=None):

    """extract cluster timepoints and channels from cluster test in sensor space and put contrast into source space
    using eLoreta"""

    IDlist = ['007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021', '022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '041', '042', '043', '044', '046',
          '047', '048', '049']

    # extract cluster channels and timepoints from significant clusters in sensor space and store difference per participant
    
    cluster_zero = clusters[good_clusters_idx[1]] # TPJ
    cluster_one = clusters[good_clusters_idx[2]] # ACC/frontal pole

    # store unique timepoints and channels per cluster

    timepoint_idx = np.unique(cluster_zero[0])
    channel_idx = np.unique(cluster_zero[1])

    timepoint_idx1 = np.unique(cluster_one[0])
    channel_idx1 = np.unique(cluster_one[1])

    # extract only significant timepoints from evoked data
    high_zero = [h.data[:,timepoint_idx] for h in evoked_dataa]
    low_zero = [h.data[:,timepoint_idx] for h in evoked_datab]
    # calculate difference for both conditions
    diff_zero = [h-l for h,l in zip(high_zero, low_zero)]

    high_one = [h.data[:,timepoint_idx1] for h in evoked_dataa]
    low_one = [h.data[:,timepoint_idx1] for h in evoked_datab]

    diff_one = [h-l for h,l in zip(high_one, low_one)]

    # convert back to EvokedArray for source reconstruction for both cluster
    diff_evo_zero = [mne.EvokedArray(z, evokeds.info) for z in diff_zero]
    diff_evo_one = [mne.EvokedArray(z, evokeds.info) for z in diff_one]

    # source reconstruct the contrast

    for idx, subj in enumerate(IDlist):

        noise_cov = mne.read_cov('D:\expecon_ms\data\eeg\source\cluster_source\covariance_prestim.cov')

        inv_op = mne.minimum_norm.make_inverse_operator(diff_evo_zero[0].info, fwd, noise_cov)

        diff_evo_zero[idx].set_eeg_reference(projection=True)  # needed for inverse modeling

        cluster_0 = mne.minimum_norm.apply_inverse(diff_evo_zero[idx], inv_op, lambda2=0.05,
                                                    method='eLORETA', pick_ori='normal')
    
        os.chdir("D:\expecon_ms\data\eeg\source\cluster_source")

        cluster_0.save('cluster0_' + subj)
    
    # load and plot the stc contrast

    stc_all = []

    for idx, subj in enumerate(IDlist):

        os.chdir("D:\expecon_ms\data\eeg\source\cluster_source")

        conditiona_stc = mne.read_source_estimate('cluster1_' + subj)

        stc_all.append(conditiona_stc)

    stc_array = np.array([np.array(s.data) for s in stc_all])

    # average over time and participants
    X_avg = np.mean(stc_array[:,:,:], axis=(0))

    fsave_vertices = [s['vertno'] for s in src]

    # create source estimate

    stc = mne.SourceEstimate(X_avg, tmin=-0.5, tstep=0.0001, vertices = fsave_vertices)

    pos, latency = stc.get_peak(hemi='rh', vert_as_index=True, time_as_index=True)

    peak_vertex_surf = stc.rh_vertno[pos]

    # plot source contrast
    brain = stc.plot(
        hemi='both', views='medial', subjects_dir=subjects_dir,
        subject = 'fsaverage', time_viewer=False, background='white')
    # save source contrast image
    brain.save_image()


def extract_diff_per_sub():

    # extract cluster channels and timepoints from significant clusters in sensor space and store difference per participant

    cluster_zero = clusters[good_cluster_inds[0]]
    cluster_one = clusters[good_cluster_inds[1]]

    # store unique timepoints and channels per cluster

    timepoint_idx = np.unique(cluster_zero[0])
    channel_idx = np.unique(cluster_zero[1])

    timepoint_idx1 = np.unique(cluster_one[0])
    channel_idx1 = np.unique(cluster_one[1])

    # extract data only from evokeds

    high = [h.data for h in evokeds_high_all]
    low = [h.data for h in evokeds_low_all]

    high_czero = [h[channel_idx,:] for h in high]
    low_czero = [h[channel_idx,:] for h in low]

    high_zero = [h[:,timepoint_idx] for h in high_czero]
    low_zero = [h[:,timepoint_idx] for h in low_czero]

    # calculate difference for both conditions
    diff_zero = [h-l for h,l in zip(high_zero, low_zero)]

    high_cone = [h[channel_idx1,:] for h in high]
    low_cone = [h[channel_idx1,:] for h in low]

    high_one = [h[:,timepoint_idx1] for h in high_cone]
    low_one = [h[:,timepoint_idx1] for h in low_cone]

    diff_one = [h[:,:]-l[:,:] for h,l in zip(high_one, low_one)]

    alpha_one = mne.filter.filter_data(diff_one, 250, 7, 13)
    beta_one = mne.filter.filter_data(diff_one, 250, 30, 39)

    alpha_zero = mne.filter.filter_data(diff_zero, 250, 7, 13)
    beta_zero = mne.filter.filter_data(diff_zero, 250, 15, 25)

    diff_zero_sub = np.mean(np.array(diff_zero), axis=(1,2))
    diff_one_sub = np.mean(np.array(diff_one), axis=(1,2))
    
    diff_a_one = np.mean(np.array(alpha_one**2), axis=(1,2))
    diff_b_one = np.mean(np.array(beta_one**2), axis=(1,2))

    diff_a_zero = np.mean(np.array(alpha_zero**2), axis=(1,2))
    diff_b_zero = np.mean(np.array(beta_zero**2), axis=(1,2))

    # read in criterion difference

    c_diff = pd.read_csv('D:\\expecon_ms\\data\\behav\\diff_c.csv')

    # calculate TFR for alpha and beta band in both clusters and store difference per participant

    scipy.stats.pearsonr(diff_one, c_diff.iloc[:,1])

    # correlate with delta criterion per participant (positive correlation?)


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