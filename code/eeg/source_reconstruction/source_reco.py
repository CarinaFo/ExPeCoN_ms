import os.path as op
import os
import numpy as np
import mne
from mne.datasets import fetch_fsaverage
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd
from mpl_toolkits.axes_grid1 import make_axes_locatable

#mne.viz.set_3d_backend("notebook")
print(__doc__)

# Reading the raw data and creating epochs:

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'

_oct = '6'

fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')
src_fname = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')

# Read the source space and the forward solution
src = mne.read_source_spaces(src_fname)
fwd = mne.read_forward_solution(fwd_dir)

save_dir_cluster_output = "D:\expecon_ms\data\eeg\cluster_test_output"

IDlist = ['007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021', '022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046',
          '047', '048', '049']
          
def run_source_reco(dics=0):

    trials_removed, all_trials = [], []

    for idx, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)
        # skip those participants
        if subj == '040' or subj == '045':
            continue

        # load cleaned epochs
        os.chdir("D:\expecon_ms\data\eeg\prepro_ica\clean_epochs")

        epochs = mne.read_epochs('P' + subj + '_epochs_after_ica-epo.fif')

        # Remove 6 blocks with hitrates < 0.2 or > 0.8

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

        #save n trials per participant
        all_trials.append(len(epochs.metadata))

        trials_removed.append(before_rt_removal - len(epochs.metadata))

        #high vs. low condition
        epochs_high = epochs[epochs.metadata.cue == 0.75]
        epochs_low = epochs[epochs.metadata.cue == 0.25]

        #average and crop in prestimulus window
        evokeds_high = epochs_high.average().crop(-0.5,0)
        evokeds_low = epochs_low.average().crop(-0.5,0)

        if dics == 1:
                
            # We are interested in the beta band. Define a range of frequencies, using a
            # log scale, from 12 to 30 Hz.

            freqs = np.logspace(np.log10(15), np.log10(30), 9)

            # Computing the cross-spectral density matrix for the beta frequency band, for
            # different time intervals.
            csd = csd_morlet(epochs, freqs, tmin=-1, tmax=1)
            csd_a = csd_morlet(epochs_high, freqs, tmin=-0.5, tmax=0)
            csd_b = csd_morlet(epochs_low, freqs, tmin=-0.5, tmax=0)
            csd_baseline = csd_morlet(epochs, freqs, tmin=-1, tmax=-0.5)

            info = epochs.info

            # To compute the source power for a frequency band, rather than each frequency
            # separately, we average the CSD objects across frequencies.
            csd_a = csd_a.mean()
            csd_b = csd_b.mean()
            csd_baseline = csd_baseline.mean()

            # Computing DICS spatial filters using the CSD that was computed on the entire
            # timecourse.

            filters = make_dics(info, fwd, csd, noise_csd=csd_baseline,
                                pick_ori='max-power', reduce_rank=True, real_filter=True)

            # Applying DICS spatial filters separately to the CSD computed using the
            # baseline and the CSD computed during the ERS activity.

            source_power_a, freqs = apply_dics_csd(csd_a, filters)
            source_power_b, freqs = apply_dics_csd(csd_b, filters)

            os.chdir("D:\expecon_ms\data\eeg\source_dics\high_low_pre")

            source_power_a.save('high_' + subj)
            source_power_b.save('low_' + subj)

        else:

            # create noise covariance with a bias of data length
            noise_cov = create_noise_cov(evokeds_high.data.shape, evokeds_high.info)

            mne.write_cov('covariance_prestim.cov', noise_cov)

            inv_op = mne.minimum_norm.make_inverse_operator(evokeds_high.info, fwd, noise_cov,
                                                                loose=1.0, fixed=False)

            evokeds_high.set_eeg_reference(projection=True)  # needed for inverse modeling

            conditiona_stc = mne.minimum_norm.apply_inverse(evokeds_high, inv_op, lambda2=0.05,
                                                            method='eLORETA', pick_ori='normal')
            
            inv_op = mne.minimum_norm.make_inverse_operator(evokeds_low.info, fwd, noise_cov,
                                                                loose=1.0, fixed=False)

            evokeds_low.set_eeg_reference(projection=True)  # needed for inverse modeling

            conditionb_stc = mne.minimum_norm.apply_inverse(evokeds_low, inv_op, lambda2=0.05,
                                                            method='eLORETA', pick_ori='normal')
            os.chdir("D:\expecon_ms\data\eeg\source\high_low_pre")

            conditiona_stc.save('high_' + subj, overwrite=True)
            conditionb_stc.save('low_' + subj, overwrite=True)

def source_contrast():

            
    stc_all = []

    for idx, subj in enumerate(IDlist):

        # skip those participants
        if subj == '040' or subj == '045':
            continue

        os.chdir("D:\expecon_ms\data\eeg\source\high_low_pre")

        stc_high = mne.read_source_estimate('high_' + subj)
        stc_low = mne.read_source_estimate('low_' + subj)

        stc_diff = stc_high.data/stc_low.data
        
        #stc_diff = stc_high.data-stc_low.data

        #stc_diff.save('diff_' + subj)

        stc_all.append(stc_diff)

    stc_array = np.array(stc_all)

    stc_array.shape # participants x vertices x timepoints

    return stc_array

def spatio_temporal_source_test():
        
    print('Computing adjacency.')

    adjacency = mne.spatial_src_adjacency(src)

    # Note that X needs to be a multi-dimensional array of shape
    # observations (subjects) × time × space, so we permute dimensions

    X = np.transpose(stc_array, [0, 2, 1])

    X_mean = np.mean(X[:,100:,:], axis=1)

    H,p,t = mne.stats.permutation_t_test(X_mean)

    X_avg = np.mean(stc_array[:,:,100:], axis=(0,2))

    stc = mne.SourceEstimate(p, tmin=-0.5, tstep=0.0001, vertices = fsave_vertices)

    brain = stc.plot(
        hemi='rh', views='lateral', subjects_dir=subjects_dir,
        subject = 'fsaverage', time_viewer=False)

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
                                                     n_jobs=10, n_permutations=perm)

    return T_obs, clusters, cluster_p_values, H0

def plot_cluster_output():
        
    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_idx = np.where(cluster_p_values < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]

    print('Visualizing clusters.')

    # Now let's build a convenient representation of our results, where consecutive
    # cluster spatial maps are stacked in the time dimension of a SourceEstimate
    # object. This way by moving through the time dimension we will be able to see
    # subsequent cluster maps.
    stc_all_cluster_vis = mne.stats.summarize_clusters_stc(clu, tstep=tstep,
                                                vertices=fsave_vertices,
                                                subject='fsaverage', p_thresh=0.05)

    # Let's actually plot the first "time point" in the SourceEstimate, which
    # shows all the clusters, weighted by duration.

    fsave_vertices = [s['vertno'] for s in src]
    # blue blobs are for condition A < condition B, red for A > B
    brain = stc_all_cluster_vis.plot(
        hemi='rh', views='lateral', subjects_dir=subjects_dir,
        time_label='temporal extent (ms)', size=(800, 800),
        smoothing_steps=5)

    # We could save this via the following:
    # brain.save_image('clusters.png')
    #message = 'DICS source power in the 12-30 Hz frequency band'

    #brain = stc.plot(hemi='rh', views='axial', subjects_dir=subjects_dir,
    #                subject=subject, time_label=message)

def cluster_perm_space_time(csp=0, perm=10000):
        
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

        if csp:

            epochs.crop(-0.5,0)

            labels = epochs.metadata.cue

            csp = mne.decoding.CSP(n_components=2, reg=0.5, log=True, norm_trace=False)

            # plot CSP patterns estimated on full data for visualization
            csp.fit_transform(epochs.get_data(), labels)

            csp.plot_patterns(epochs.info, ch_type='eeg', units='Patterns (AU)', size=1.5)

        #epochs = epochs.filter(1,4)

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
        epochs = epochs[epochs.metadata.trial != 1]
        
        #save n trials per participant
        all_trials.append(len(epochs.metadata))

        trials_removed.append(before_rt_removal - len(epochs.metadata))

        #high vs. low condition
        epochs_high = epochs[(epochs.metadata.cue == 0.75)]
        epochs_low = epochs[(epochs.metadata.cue == 0.25)]

        #average and crop in prestimulus window
        evokeds_high = epochs_high.average().crop(-0.5,0)
        evokeds_low = epochs_low.average().crop(-0.5,0)

        evokeds_high_all.append(evokeds_high)
        evokeds_low_all.append(evokeds_low)

    # get grand average over all subjects for plotting the results later

    a_gra = mne.grand_average(evokeds_high_all)
    b_gra = mne.grand_average(evokeds_low_all)

    X = [h.data-l.data for h,l in zip(evokeds_high_all, evokeds_low_all)]

    X = np.transpose(X, [0, 2, 1])

    ch_adjacency,_ = mne.channels.find_ch_adjacency(epochs.info, ch_type='eeg')

    threshold_tfce = dict(start=0, step=0.1)
    
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X[:,:,:], n_permutations=perm,
                                                                                    adjacency=ch_adjacency)

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0] # times where something significant happened

    print(len(good_cluster_inds))
    print(cluster_p_values)

    # this seemed to work, now plot the significant cluster

    a = 'high'
    b= 'low'

    # configure variables for visualization
    colors = {a: "crimson", b: 'steelblue'}
    #
    # # organize data for plotting
    #instead of grand average we use the evoked data per subject so that we can plot CIs

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
        plt.show()


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


def extract_cluster_and_plot_source_contrast():

    """extract cluster timepoints and channels from cluster test in sensor space and put contrast into source space
    using eLoreta"""

    IDlist = ['007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021', '022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '041', '042', '043', '044', '046',
          '047', '048', '049']

    # extract cluster channels and timepoints from significant clusters in sensor space and store difference per participant

    cluster_zero = clusters[good_cluster_inds[0]]
    cluster_one = clusters[good_cluster_inds[1]]

    # store unique timepoints and channels per cluster

    timepoint_idx = np.unique(cluster_zero[0])
    channel_idx = np.unique(cluster_zero[1])

    timepoint_idx1 = np.unique(cluster_one[0])
    channel_idx1 = np.unique(cluster_one[1])

    # extract only significant timepoints from evoked data
    high_zero = [h.data[:,timepoint_idx] for h in evokeds_high_all]
    low_zero = [h.data[:,timepoint_idx] for h in evokeds_low_all]
    # calculate difference for both conditions
    diff_zero = [h-l for h,l in zip(high_zero, low_zero)]

    high_one = [h.data[:,timepoint_idx1] for h in evokeds_high_all]
    low_one = [h.data[:,timepoint_idx1] for h in evokeds_low_all]

    diff_one = [h-l for h,l in zip(high_one, low_one)]
    # convert back to EvokedArray for source reconstruction for both cluster
    diff_evo_zero = [mne.EvokedArray(z, evokeds_high.info) for z in diff_zero]
    diff_evo_one = [mne.EvokedArray(z, evokeds_high.info) for z in diff_one]

    # source reconstruct the contrast

    for idx, subj in enumerate(IDlist):

        noise_cov = mne.read_cov('covariance_prestim.cov')

        inv_op = mne.minimum_norm.make_inverse_operator(diff_evo_one[0].info, fwd, noise_cov)

        #inv_op = mne.minimum_norm.make_inverse_operator(diff_evo_one[0].info, fwd, noise_cov, loose=1.0, fixed=False)

        diff_evo_one[idx].set_eeg_reference(projection=True)  # needed for inverse modeling

        cluster_1 = mne.minimum_norm.apply_inverse(diff_evo_one[idx], inv_op, lambda2=0.05,
                                                    method='eLORETA', pick_ori='normal')
    
        os.chdir("D:\expecon_ms\data\eeg\source\cluster_source")

        cluster_1.save('cluster1_loose_' + subj)
    
    # source reconstruct the contrast

    stc_all = []

    for idx, subj in enumerate(IDlist):

        conditiona_stc = mne.read_source_estimate('cluster1_loose_' + subj)

        stc_all.append(conditiona_stc)

    stc_array = np.array([np.array(s.data) for s in stc_all])

    # average over time and participants
    X_avg = np.mean(stc_array[:,:,:], axis=(0))

    # create source estimate
    stc = mne.SourceEstimate(X_avg, tmin=-0.5, tstep=0.0001, vertices = fsave_vertices)
    # plot source contrast
    brain = stc.plot(
        hemi='rh', views='lateral', subjects_dir=subjects_dir,
        subject = 'fsaverage', time_viewer=True)

    # now extract only cluster channel and timepoints per epoch

    high_zero = [h.data[:,timepoint_idx] for h in evokeds_high_all]
    high_one = [h.data[clusters[good_cluster_inds[1]][1],clusters[good_cluster_inds[1]][0]] for h in evokeds_high_all]


    low_one = [h[ch_cluster_zero,:] for h in evokeds_low_all]
    low_zero = [h[ch_cluster_zero,:] for h in evokeds_low_all]

    # calculate TFR for alpha and beta band in both clusters and store difference per participant


    # correlate with delta criterion per participant (positive correlation?)



def extract_cluster_per_trial():

    # extract cluster in timeseries and power per trial and store per participant for mixed model analys

    