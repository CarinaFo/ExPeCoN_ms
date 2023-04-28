########################################evoked analysis######################################


# Author: Carina Forster
# Date: 2023-04-03

# This script plots grand average evoked responses for high and low cue conditions
# and runs a paired ttest between the two conditions with a cluster based permutation test
# to correct for multiple comparisons

# Functions used:
# cluster_perm_space_time: runs a cluster permutation test over electrodes and timepoints
# in sensor space and plots the output and saves it

import mne
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import os
import pandas as pd

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# set paths
dir_cleanepochs = r"D:\expecon_ms\data\eeg\prepro_ica\clean_epochs"

# save cluster figures as svg and png files
save_dir_cluster_output = r"D:\expecon_ms\figs\eeg\sensor\evokeds"

# participant index
IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')


def create_contrast(perm=10000, tmin=0, tmax=0.5, cond='highsignal', cond_a=
                            'high', cond_b='low', laplace=True):

    """ this function runs a cluster permutation test over electrodes and timepoints
    in sensor space and plots the output and saves it
    input: 
    perm: how many permutations for cluster test
    tmin: crop the epochs at this time in seconds
    tmax: crop the data until this time in seconds
    cond: condition to be analyzed (highlow, signalnoise, highsignal, highnoise)
    cond_a: name of condition a
    cond_b: name of condition b
    laplace: apply laplace transform to data
    output:
    saves the cluster figures as svg and png files """

    all_trials, trials_removed = [], []

    evokeds_a_all, evokeds_b_all = [], []

    for idx, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)
        # skip those participants
        if subj == '040' or subj == '045' or subj == '032':
            continue

        # load cleaned epochs
        epochs = mne.read_epochs(f"{dir_cleanepochs}/P{subj}_epochs_after_ica-epo.fif")

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
        
        # remove trials with rts >= 2.5 (no response trials) and trials with rts < 0.1
        before_rt_removal = len(epochs.metadata)
        epochs = epochs[epochs.metadata.respt1 > 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # remove first trial of each block (trigger delays)
        epochs = epochs[epochs.metadata.trial != 1]
        
        #save n_trials per participant
        all_trials.append(len(epochs.metadata))

        trials_removed.append(before_rt_removal - len(epochs.metadata))

        #high vs. low condition
        if cond == 'highlow':
            epochs_a = epochs[(epochs.metadata.cue == 0.75)]
            epochs_b = epochs[(epochs.metadata.cue == 0.25)]
        elif cond == 'signalnoise':
            epochs_a = epochs[(epochs.metadata.isyes == 1)]
            epochs_b = epochs[(epochs.metadata.isyes == 0)]
        elif cond == 'highsignal':
            epochs_a = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.isyes == 1))]
            epochs_b = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.isyes == 1))]
        elif cond == 'highnoise':
            epochs_a = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.isyes == 0))]
            epochs_b = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.isyes == 0))]

        # apply laplace transform
        if laplace == True:
            epochs_a = mne.preprocessing.compute_current_source_density(epochs_a)
            epochs_b = mne.preprocessing.compute_current_source_density(epochs_b)

        #average and crop in prestimulus window
        evokeds_a = epochs_a.average().crop(tmin, tmax)
        evokeds_b = epochs_b.average().crop(tmin, tmax)

        evokeds_a_all.append(evokeds_a)
        evokeds_b_all.append(evokeds_b)
        
        # save trial number and trials removed to csv file

        pd.DataFrame(trials_removed).to_csv('trials_removed.csv')
        pd.DataFrame(all_trials).to_csv('trials_per_subject.csv')

    return evokeds_a_all, evokeds_b_all, all_trials, trials_removed, cond_a, cond_b

def cluster_perm_space_time_plot(evokeds_a_all, evokeds_b_all,
                                 cond_a, cond_b, perm=10000, tmin=-0.5, tmax=0, cond='highsignal', laplace=True):

    # get grand average over all subjects for plotting the results later

    a_gra = mne.grand_average(evokeds_a_all)
    b_gra = mne.grand_average(evokeds_b_all)

    a = np.array([ax.data for ax in evokeds_a_all])
    b = np.array([bx.data for bx in evokeds_b_all])

    X = [ax.data-bx.data for ax,bx in zip(evokeds_a_all, evokeds_b_all)]

    X = np.transpose(X, [0, 2, 1])

    # load cleaned epochs
    subj='007'
    epochs = mne.read_epochs(f"{dir_cleanepochs}/P{subj}_epochs_after_ica-epo.fif")

    ch_adjacency,_ = mne.channels.find_ch_adjacency(epochs.info, ch_type='eeg')

    threshold_tfce = dict(start=0, step=0.1)

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X[:,:,:], n_permutations=perm,
                                                                                    adjacency=ch_adjacency, 
                                                                                    #threshold=threshold_tfce, 
                                                                                    tail=0, n_jobs=-1
                                                                                    )

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0] # times where something significant happened

    print(len(good_cluster_inds))
    print(cluster_p_values)

    # now plot the significant cluster(s)
    a = cond_a
    b= cond_b

    # configure variables for visualization
    colors = {a: "#ff2a2aff", b: '#2a95ffff'}

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
                                    color='darkgreen', alpha=0.3)

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        os.chdir(save_dir_cluster_output)
        plt.savefig('cluster_highnoise_post_' + str(i_clu) + '.svg')
        plt.savefig('cluster_highnoise_post_' + str(i_clu) + '.png')
        plt.show()
