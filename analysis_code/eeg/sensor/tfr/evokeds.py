########################################evoked analysis######################################


# Author: Carina Forster
# Date: 2023-04-03

# This script plots grand average evoked responses for high and low cue conditions
# and runs a paired ttest between the two conditions with a cluster based permutation test
# to correct for multiple comparisons

# Functions used:
# cluster_perm_space_time: runs a cluster permutation test over electrodes and timepoints
# in sensor space and plots the output and saves it

import os.path as op

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyvistaqt  # for proper 3D plotting
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import scipy.stats as stats

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# set paths
dir_cleanepochs = 'D:\\expecon_ms\\data\\eeg\prepro_ica\\clean_epochs'
behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'

# figure path
savedir_figs = 'D:\\expecon_ms\\figs\\manuscript_figures\\Figure3'

# participant index
IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')


def create_contrast(tmin=-0.5, tmax=0, cond='highlow',
                    cond_a='high', cond_b='low', laplace=True,
                    reject_criteria=dict(eeg=200e-6),
                    flat_criteria=dict(eeg=1e-6), induced=True):

    """ This function creates a contrast between two conditions for epoched data in a specified time window.
    It returns the evoked responses for the two conditions and the contrast between them.

    Parameters
    ----------
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    cond : str
        Condition to be contrasted.
    cond_a : str
        First condition to be contrasted.
    cond_b : str
        Second condition to be contrasted.
    laplace : bool
        If True, data is laplacian transformed.
    reject_criteria : dict
        Criteria for rejecting trials.
    flat_criteria : dict
        Criteria for rejecting trials.
    
    Returns
    -------
    evokeds_a : mne.evoked
        Evoked response for condition a.
    evokeds_b : mne.evoked
        Evoked response for condition b.
    cond_a : str
        Name of condition a.
    cond_b : str
        Name of condition b.
    cond: str
        Name of contrast.
    """

    all_trials, trials_removed = [], []

    evokeds_a_all, evokeds_b_all = [], []

    for counter, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)

        # skip those participants
        if subj == '040' or subj == '045' or subj == '032' or subj == '016':
            continue

        # load cleaned epochs
        epochs = mne.read_epochs(f"{dir_cleanepochs}"
                                 f"/P{subj}_epochs_after_ica-epo.fif")

        # Remove 5 blocks with hitrates < 0.2 or > 0.8
        if subj == '010':
            epochs = epochs[epochs.metadata.block != 6]
        if subj == '012':
            epochs = epochs[epochs.metadata.block != 6]
        if subj == '026':
            epochs = epochs[epochs.metadata.block != 4]
        if subj == '030':
            epochs = epochs[epochs.metadata.block != 3]
        if subj == '039':
            epochs = epochs[epochs.metadata.block != 3]
        
        # remove trials with rts >= 2.5 (no response trials) 
        # and trials with rts < 0.1
        epochs = epochs[epochs.metadata.respt1 > 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # remove first trial of each block (trigger delays)
        epochs = epochs[epochs.metadata.trial != 1]

        # load behavioral data
        data = pd.read_csv(f"{behavpath}//prepro_behav_data.csv")

        subj_data = data[data.ID == counter+7]

        if ((counter == 5) or (counter == 13) or
           (counter == 21) or (counter == 28)):  # first epoch has no data
            epochs.metadata = subj_data.iloc[1:, :]
        elif counter == 17:
            epochs.metadata = subj_data.iloc[3:, :]
        else:
            epochs.metadata = subj_data

        # drop bad epochs
        epochs.drop_bad(reject=reject_criteria, flat=flat_criteria)

        # create contrasts
        if cond == 'highlow':
            epochs_a = epochs[(epochs.metadata.cue == 0.75)]
            epochs_b = epochs[(epochs.metadata.cue == 0.25)]
        if cond == 'hitmiss': # sign in prestim window
            epochs_a = epochs[((epochs.metadata.isyes == 1)
                              & (epochs.metadata.sayyes == 1))]
            epochs_b = epochs[((epochs.metadata.isyes == 1)
                              & (epochs.metadata.sayyes == 0))]
        elif cond == 'confunconf_hits': # sign in prestim window
            epochs_a = epochs[((epochs.metadata.isyes == 1)
                              & (epochs.metadata.sayyes == 1)
                              & (epochs.metadata.conf == 1))]
            epochs_b = epochs[((epochs.metadata.isyes == 1)
                              & (epochs.metadata.sayyes == 1)
                              & (epochs.metadata.conf == 0))]
        elif cond == 'prevchoice': # no sign. cluster in prestim window
            epochs_a = epochs[(epochs.metadata.prevsayyes == 1)]
            epochs_b = epochs[(epochs.metadata.prevsayyes == 0)]
        elif cond == 'prevstim':   # no sign cluster in prestim window
            epochs_a = epochs[(epochs.metadata.previsyes == 1)]
            epochs_b = epochs[(epochs.metadata.previsyes == 0)]
        elif cond == 'prevchoice_yes': # no sign. cluster in prestim window
            epochs_a = epochs[((epochs.metadata.cue == 0.75)
                              & (epochs.metadata.prevsayyes == 1))]
            epochs_b = epochs[((epochs.metadata.cue == 0.25)
                              & (epochs.metadata.prevsayyes == 1))]
        elif cond == 'prevchoice_no':  # sign. cluster, similar to highlow in prestim window
            epochs_a = epochs[(epochs.metadata.cue == 0.75)
                              & (epochs.metadata.prevsayyes == 0)]
            epochs_b = epochs[(epochs.metadata.cue == 0.25)
                              & (epochs.metadata.prevsayyes == 0)]
        elif cond == 'highlow_prevno':  # 2 sign. clusters in prestim window
            epochs_a = epochs[((epochs.metadata.prevsayyes == 0)
                              & (epochs.metadata.isyes == 1)
                              & (epochs.metadata.sayyes == 1)
                              & (epochs.metadata.cue == 0.75))]
            epochs_b = epochs[((epochs.metadata.prevsayyes == 0)
                              & (epochs.metadata.isyes == 1)
                              & (epochs.metadata.sayyes == 1)
                              & (epochs.metadata.cue == 0.25))]
        elif cond == 'highlow_prevyes':  # n.s. in poststim window
            epochs_a = epochs[((epochs.metadata.prevsayyes == 1)
                              & (epochs.metadata.isyes == 1)
                              & (epochs.metadata.sayyes == 1)
                              & (epochs.metadata.cue == 0.75))]
            epochs_b = epochs[((epochs.metadata.prevsayyes == 1)
                              & (epochs.metadata.isyes == 1)
                              & (epochs.metadata.sayyes == 1)
                              & (epochs.metadata.cue == 0.25))]
        elif cond == 'highlow_cr':   # n.s. in poststim window
            epochs_a = epochs[((epochs.metadata.isyes == 0)
                              & (epochs.metadata.sayyes == 0)
                              & (epochs.metadata.cue == 0.75))]
            epochs_b = epochs[((epochs.metadata.isyes == 0)
                              & (epochs.metadata.sayyes == 0)
                              & (epochs.metadata.cue == 0.25))]
        elif cond == 'highlow_hits':   # 1 sign. cluster in poststim window
            epochs_a = epochs[((epochs.metadata.isyes == 1)
                              & (epochs.metadata.sayyes == 1)
                              & (epochs.metadata.cue == 0.75))]
            epochs_b = epochs[((epochs.metadata.isyes == 1)
                              & (epochs.metadata.sayyes == 1)
                              & (epochs.metadata.cue == 0.25))]
        
        if induced:
            epochs_a = epochs_a.subract_evoked()
            epochs_b = epochs_b.subract_evoked()

        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        # apply laplace transform
        if laplace == True:
            epochs_a = mne.preprocessing.compute_current_source_density(epochs_a)
            epochs_b = mne.preprocessing.compute_current_source_density(epochs_b)

        # average and crop in defined time window
        evokeds_a = epochs_a.average().crop(tmin, tmax)
        evokeds_b = epochs_b.average().crop(tmin, tmax)

        evokeds_a_all.append(evokeds_a)
        evokeds_b_all.append(evokeds_b)
        
        # save n_trials per participant and trials removed to a csv file
        pd.DataFrame(trials_removed).to_csv(f'{behavpath}//trials_removed.csv')
        pd.DataFrame(all_trials).to_csv(f'{behavpath}//trials_per_subject.csv')

    return evokeds_a_all, evokeds_b_all, cond_a, cond_b, cond

def plot_psd(data_a=None, data_b=None, cond_a=None, cond_b=None,
             fmin=1, fmax=40, picks=['C4']):

    """Plot PSD for grand average data."""

    gra_high = mne.grand_average(data_a)
    gra_low = mne.grand_average(data_b)

    psd_low = gra_low.compute_psd(fmin=fmin, fmax=fmax, picks=picks)
    psd_high = gra_high.compute_psd(fmin=fmin, fmax=fmax, picks=picks)

    psd_low_d = psd_low.get_data()
    psd_high_d = psd_high.get_data()

    freqs = psd_low.freqs

    plt.plot(freqs, np.log10(psd_high_d[0]), label=cond_a)
    plt.plot(freqs, np.log10(psd_low_d[0]), label=cond_b)
    plt.legend()

    plt.savefig(f'psd_{cond_a}_{cond_b}_laplace.png')
    plt.show()


def load_all_epochs(tmin=-0.3, tmax=-0.2, laplace=True,
                    reject_criteria=dict(eeg=200e-6),
                    flat_criteria=dict(eeg=1e-6),
                    channel_list = ['C3', 'CP5', 'CP1', 'Pz', 'CP2', 'C4', 'T8', 'FC6','FC2',
                                    'C1','CP3', 'P1', 'P2', 'CPz','CP4','C6','C2','FC4']):

    """ This function loads all epochs per participant and averages over time and channels

    Parameters
    ----------
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    laplace : bool
        If True, data is laplacian transformed.
    reject_criteria : dict
        Criteria for rejecting trials.
    flat_criteria : dict
        Criteria for rejecting trials.
    
    Returns
    -------
    evokeds_a : mne.evoked
        Evoked response for condition a.
    """
    epochs_all, df = [], []

    for counter, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)

        # skip those participants
        if subj == '040' or subj == '045' or subj == '032' or subj == '016':
            continue

        # load cleaned epochs
        epochs = mne.read_epochs(f"{dir_cleanepochs}"
                                 f"/P{subj}_epochs_after_ica-epo.fif")

        # Remove 5 blocks with hitrates < 0.2 or > 0.8
        if subj == '010':
            epochs = epochs[epochs.metadata.block != 6]
        if subj == '012':
            epochs = epochs[epochs.metadata.block != 6]
        if subj == '026':
            epochs = epochs[epochs.metadata.block != 4]
        if subj == '030':
            epochs = epochs[epochs.metadata.block != 3]
        if subj == '039':
            epochs = epochs[epochs.metadata.block != 3]
        
        # remove trials with rts >= 2.5 (no response trials) 
        # and trials with rts < 0.1
        epochs = epochs[epochs.metadata.respt1 > 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # remove first trial of each block (trigger delays)
        epochs = epochs[epochs.metadata.trial != 1]

        # load behavioral data
        data = pd.read_csv(f"{behavpath}//prepro_behav_data.csv")

        subj_data = data[data.ID == counter+7]

        if ((counter == 5) or (counter == 13) or
           (counter == 21) or (counter == 28)):  # first epoch has no data
            epochs.metadata = subj_data.iloc[1:, :]
        elif counter == 17:
            epochs.metadata = subj_data.iloc[3:, :]
        else:
            epochs.metadata = subj_data

        # drop bad epochs
        epochs.drop_bad(reject=reject_criteria, flat=flat_criteria)

        # apply laplace transform
        if laplace == True:
            epochs = mne.preprocessing.compute_current_source_density(epochs)

        # average and crop in defined time window
        epochs = epochs.crop(tmin, tmax)
        epochs = epochs.pick_channels(channel_list)

        metadata = epochs.metadata

        df.append(metadata)

        epochs = epochs.get_data().mean(axis=(1, 2))
    
        epochs_all.append(epochs)
        
    return epochs_all, df

def cluster_perm_space_time_plot(perm=10000, channels=['C4', 'CP4', 'C6', 'CP6'],
                                 average_over_channels=False):
    
    """Permutation test for cluster-based permutation test over space and time.
    First creates contrasts and then performs the permutation test over channels 
    and time points (p=.05 for both cluster and significance level).
    Plots the evoked signal for the significant cluster and a topomap of the
    t-values.
    Parameters
    ----------
    perm : int
        Number of permutations.
    channels : list
        List of channels to average over.
    average_over_channels : bool
        If True, average over channels.
    Returns
    -------
    t : array
        T-values.
    p : array   
        P-values.
    h : array
    Boolean array of significant sensors.
    """


    # create contrasts
    evokeds_a_all, evokeds_b_all, cond_a, cond_b, cond = create_contrast()

    a_gra = mne.grand_average(evokeds_a_all)
    b_gra = mne.grand_average(evokeds_b_all)

    if average_over_channels is True:

        # select channels and convert to array
        a = np.array([ax.copy().pick_channels(channels).data 
                      for ax in evokeds_a_all])
        b = np.array([bx.copy().pick_channels(channels).data 
                      for bx in evokeds_b_all])

        # take mean over channels
        a = np.mean(a, axis=1)
        b = np.mean(b, axis=1)

        # test if difference is sign. different from zero
        X = a-b

        t, p, h = mne.stats.permutation_t_test(X, n_permutations=perm, tail=0,
                                               n_jobs=-1)

        print(min(p))

    # create contrast
    X = [ax.data-bx.data for ax, bx in zip(evokeds_a_all, evokeds_b_all)]

    # change the axis for cluster test (channels should be last axis)
    X = np.transpose(X, [0, 2, 1])

    # load cleaned epochs for one participant to get channel adjacency
    subj = '007'
    epochs = mne.read_epochs(f"{dir_cleanepochs}/P{subj}_epochs_after_ica-epo.fif")

    ch_adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, ch_type='eeg')

    # threshold_tfce = dict(start=0, step=0.1)

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X[:,:,:], n_permutations=perm,
                                   adjacency=ch_adjacency,
                                   #threshold=threshold_tfce,
                                   tail=0, n_jobs=-1)

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]  # find significant clusters

    print(len(good_cluster_inds))
    print(cluster_p_values[good_cluster_inds])

    # now plot the significant cluster(s)
    a = cond_a
    b = cond_b

    # configure variables for visualization
    colors = {a: "#ff2a2aff", b: '#2a95ffff'}

    # organize data for plotting
    # instead of grand average we could use the evoked data per subject so 
    # that we can plot CIs
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
                              colorbar=False,
                              mask_params=dict(markersize=10))
        
        image = ax_topo.images[0]

        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel('Averaged t-map ({:0.3f} - {:0.3f} s)'
                           .format(*sig_times[[0, -1]]))

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes('right', size='300%', pad=1.2)
        title = 'Cluster #{0}, {1} sensor'.format(i_clu + 1, len(ch_inds))
        if len(ch_inds) > 1:
            title += "s (mean)"

        mne.viz.plot_compare_evokeds(grand_average, title=title, picks=ch_inds, 
                                     axes=ax_signals,
                                     colors=colors, show=False, ci=True,
                                     split_legend=True, legend='lower right', 
                                     truncate_yaxis='auto')

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                                  color='lightgrey', alpha=0.3)

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)

        # save figure as svg and png file
        plt.savefig(f'{savedir_figs}//cluster_nolaplace_{cond}_{str(i_clu)}.svg')
        plt.savefig(f'{savedir_figs}//cluster_nolaplace_{cond}_{str(i_clu)}.png')
        plt.show()

def cluster_to_source_space(tmin=-0.4, tmax=0, lambda2=0.05, method='eLORETA'):

    evokeds_a_all, evokeds_b_all, cond_a, cond_b, cond = out

    stc_conda, stc_condb = [], []
    inv_op = mne.minimum_norm.read_inverse_operator('D:\\expecon_ms\\data\\eeg\\source\\fsaverage-6oct-inv.fif')

    for a, b in zip(evokeds_a_all, evokeds_b_all):
            
            a.set_eeg_reference(projection=True).crop(tmin, tmax)
            b.set_eeg_reference(projection=True).crop(tmin, tmax)

            condition1_stc, condition1_residual = mne.minimum_norm.apply_inverse(a, inv_op, lambda2,
                                                                                method=method, return_residual=True)

            condition2_stc, condition2_residual = mne.minimum_norm.apply_inverse(b, inv_op, lambda2,
                                                                                method=method, return_residual=True)
            stc_conda.append(condition1_stc)
            stc_condb.append(condition2_stc)

def permutation_test():
    
    src_dir = 'D:\\expecon_ms\\data\eeg\\source\\fsaverage-6oct-src.fif'

    src = mne.read_source_spaces(src_dir)
    fsave_vertices = [s["vertno"] for s in src]

    print("Computing adjacency.")

    adjacency = mne.spatial_src_adjacency(src)

    X = np.array([a.data-b.data for a,b in zip(stc_conda, stc_condb)])

    X.shape

    X = np.transpose(X, [0, 2, 1])

    # should be participantsxtimepointsxvertices
    # Here we set a cluster forming threshold based on a p-value for
    # the cluster based permutation test.
    # We use a two-tailed threshold, the "1 - p_threshold" is needed
    # because for two-tailed tests we must specify a positive threshold.
    
    # Now let's actually do the clustering. This can take a long time...
    print("Clustering.")
    T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(
                                X,
                                adjacency=adjacency,
                                n_jobs=None,
                                buffer_size=None,
                                verbose=True,
                                n_permutations=1000
                                )

    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_idx = np.where(cluster_p_values < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]

    print("Visualizing clusters.")

    tstep = stc_conda[0].tstep*1000 # convert to ms

    # Now let's build a convenient representation of our results, where consecutive
    # cluster spatial maps are stacked in the time dimension of a SourceEstimate
    # object. This way by moving through the time dimension we will be able to see
    # subsequent cluster maps.
    stc_all_cluster_vis = mne.stats.summarize_clusters_stc(
        clu, vertices=fsave_vertices, subject="fsaverage", tmin=0,
        tstep=tstep
    )

    # Let's actually plot the first "time point" in the SourceEstimate, which
    # shows all the clusters, weighted by duration.
    subject = 'fsaverage'

    # blue blobs are for condition A < condition B, red for A > B
    brain = stc_all_cluster_vis.plot(
        hemi="both",
        views="lateral",
        subject=subject,
        time_label="temporal extent (ms)",
        size=(800, 800),
        smoothing_steps=5,
        backend='pyvistaqt',
        background='w',
    )

    # We could save this via the following:
    brain.save_image('cluster_highlow.png')

def plot_evoked_source():

    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    subjects_dir = op.dirname(fs_dir)
    subject = 'fsaverage'

    # convert list to array with dimensions: subjects,vertices,timepoints
    a_array = np.mean(np.array([s.data for s in stc_conda]), axis=0)

    # convert back to source estimate
    a_stc = mne.SourceEstimate(a_array, stc_conda[0].vertices, tmin=-0.4, tstep=0.004)

    mne.viz.plot_source_estimates(a_stc, subject=subject, subjects_dir=subjects_dir, hemi='both')

    # convert to array and take mean over participants
    b_array = np.mean(np.array([s.data for s in stc_condb]), axis=0)

    # convert back to source estimate
    b_stc = mne.SourceEstimate(b_array, stc_condb[0].vertices, tmin=-0.4, tstep=0.004)

    mne.viz.plot_source_estimates(b_stc, subject=subject, subjects_dir=subjects_dir)

    diff_arr = a_array - b_array

    diff_stc = mne.SourceEstimate(diff_arr, stc_conda[0].vertices, tmin=-0.4, tstep=0.004)

    mne.viz.plot_source_estimates(diff_stc, subject=subject, subjects_dir=subjects_dir, hemi='both')
