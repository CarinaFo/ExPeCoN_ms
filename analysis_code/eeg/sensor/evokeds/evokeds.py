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

# set font to Arial and font size to 14
plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# set paths
dir_cleanepochs = r'D:\\expecon_ms\data\eeg\prepro_ica\\clean_epochs_iclabel'
behavpath = r'D:\expecon_ms\data\behav\behav_df'

# save cluster figures as svg and png files
save_dir_cluster_output = r"D:\expecon_ms\figs\manuscript_figures\Figure4_evokeds"

# participant index
IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')


def create_contrast(cond='highlow', cond_a='high',
                    cond_b='low', fmin=7, fmax=13, 
                    envelope=False,
                    laplace=False,
                    subtract_evoked=False,
                    save_drop_log=False,
                    reject_criteria=dict(eeg=200e-6),
                    flat_criteria=dict(eeg=1e-6)):

    """ this function runs a cluster permutation test over electrodes and timepoints
    in sensor space and plots the output and saves it
    input: 
    cond: condition to be analyzed (highlow, signalnoise, highsignal, highnoise)
    cond_a: name of condition a
    cond_b: name of condition b
    laplace: apply laplace transform to data
    subtract_evoked: boolean, subtract evoked signal from each epoch
    save_drop_log: save drop log output
    output:
    None """

    perc = []

    all_trials, trials_removed = [], []

    evokeds_a_all, evokeds_b_all, epochs_all = [], [], []

    for idx, subj in enumerate(IDlist):

        # print participant idx
        print(f'Participant {str(idx)}')

        # load cleaned epochs
        epochs = mne.read_epochs(f"{dir_cleanepochs}/P{subj}_epochs_after_ic-label-epo.fif")
        
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
        before_rt_removal = len(epochs.metadata)
        epochs = epochs[epochs.metadata.respt1 >= 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # save n_trials per participant
        all_trials.append(len(epochs.metadata))

        trials_removed.append(before_rt_removal - len(epochs.metadata))

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
        #epochs.drop_bad(reject=reject_criteria, flat=flat_criteria)

        if save_drop_log:
            # percentage of trials removed
            perc.append(epochs.drop_log_stats())

            epochs.plot_drop_log(show=False)

            plt.savefig(f'D:\expecon_ms\data\eeg\prepro_ica\droplog\P{subj}_drop_log.png', dpi=300)

        if laplace:
            epochs = mne.preprocessing.compute_current_source_density(epochs)

        if cond == 'highlow_hits':
            epochs_a = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1))]
            epochs_b = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1))]
        elif cond == 'highlow_prevyes':
            epochs_a = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.prevsayyes == 1))]
            epochs_b = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.prevsayyes == 1))]
        elif cond == 'highlow_prevno':
            epochs_a = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.prevsayyes == 0))]
            epochs_b = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.prevsayyes == 0))]
        elif cond == 'highlow':
            epochs_a = epochs[(epochs.metadata.cue == 0.75)]
            epochs_b = epochs[(epochs.metadata.cue == 0.25)]
        elif cond == 'prevchoice':
            epochs_a = epochs[epochs.metadata.prevsayyes == 1]
            epochs_b = epochs[epochs.metadata.prevsayyes == 0]
        elif cond == 'highlow_prevchoiceyes_highconf':
            epochs_a = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.prevsayyes == 1) & (epochs.metadata.conf == 1))]
            epochs_b = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.prevsayyes == 1) & (epochs.metadata.conf == 1))]
        elif cond == 'signalnoise':
            epochs_a = epochs[(epochs.metadata.isyes == 1)]
            epochs_b = epochs[(epochs.metadata.isyes == 0)]
        elif cond == 'hitmiss':
            epochs_a = epochs[(epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1)]
            epochs_b = epochs[(epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 0)]
        elif cond == 'hit_prevchoice':
            epochs_a = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1) & (epochs.metadata.prevsayyes == 1))]
            epochs_b = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1) & (epochs.metadata.prevsayyes == 0))]
        elif cond == 'all_epochs':
            epochs = epochs.subtract_evoked()
            epochs.apply_hilbert(envelope=True)
            epochs_all.append(epochs)

        if cond != 'all_epochs':
            if subtract_evoked:
                epochs_a = epochs_a.subtract_evoked()
                epochs_b = epochs_b.subtract_evoked()

            if envelope:
                # apply broadband filter
                epochs_a = epochs_a.filter(fmin, fmax)       
                epochs_b = epochs_b.filter(fmin, fmax)

                # get analytic signal (envelope)
                epochs_a.apply_hilbert(envelope=True)
                epochs_b.apply_hilbert(envelope=True)

            mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

            # average and crop in prestimulus window
            evokeds_a = epochs_a.average()
            evokeds_b = epochs_b.average()

            evokeds_a_all.append(evokeds_a)
            evokeds_b_all.append(evokeds_b)
        
        # save trial number and trials removed to csv file
        pd.DataFrame(trials_removed).to_csv(f'D:\expecon_ms\data\eeg\prepro_ica\droplog\\trials_removed.csv')
        pd.DataFrame(all_trials).to_csv(f'D:\expecon_ms\data\eeg\prepro_ica\droplog\\trials_per_subject.csv')

    return evokeds_a_all, evokeds_b_all, all_trials, trials_removed, cond_a, cond_b, perc, cond


def compute_psd():

    tmin = -1
    tmax = -0.1
    fmin = 7
    fmax = 30

    # compute PSD of evoked data
    high_psd = [a.compute_psd(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, n_jobs=-1).get_data() for a in evokeds_a_all]
    low_psd = [a.compute_psd(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, n_jobs=-1).get_data() for a in evokeds_b_all]

    # convert to array
    high_psd = np.array(high_psd)
    low_psd = np.array(low_psd)

    # extract PSD for C4 electrode
    high_psd = high_psd[:, 22, :]
    low_psd = low_psd[:, 22, :]

    spectrum = evokeds_a_all[0].compute_psd(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, n_jobs=-1)
    freqs = spectrum.freqs


    plt.plot(freqs, np.log10(np.mean(high_psd, axis=0)), label='0.75')
    plt.plot(freqs, np.log10(np.mean(low_psd, axis=0)), label='0.25')
    plt.legend()
    plt.xlabel('Freq (Hz)')
    plt.ylabel('Power (log10)')



def cluster_perm_space_time_plot(perm=10000, contrast='signalnoise'):

    evokeds_a_all, evokeds_b_all, all_trials, trials_removed, cond_a, cond_b, perc = create_contrast()

    # get grand average over all subjects for plotting the results later
    
    a = [ax.copy().crop(tmin, tmax) for ax in evokeds_a_all]
    b = [bx.copy().crop(tmin, tmax) for bx in evokeds_b_all]

    a = [ax.copy().pick_channels(['C4']).crop(tmin, tmax) for ax in evokeds_a_all]
    b = [bx.copy().pick_channels(['C4']).crop(tmin, tmax) for bx in evokeds_b_all]

    a_gra = mne.grand_average(a)
    b_gra = mne.grand_average(b)

    fig = mne.viz.plot_compare_evokeds({'high': a_gra, 'low': b_gra}, combine='mean', picks=['C4'], show_sensors=True)
    fig[0].savefig(f'D:\expecon_ms\\figs\manuscript_figures\Figure4_evokeds\{cond}.svg')

    diff = mne.combine_evoked([a_gra,b_gra], weights=[1,-1])
    topo = diff.plot_topo()
    topo.savefig(f'D:\expecon_ms\\figs\manuscript_figures\Figure5_envelope\\topo_hitmiss_alpha.svg')
   
    X = np.array([ax.data-bx.data for ax,bx in zip(a,b)])

    X = np.transpose(X, [0, 2, 1]) # channels should be last dimension

    # load cleaned epochs
    subj='007'
    epochs = mne.read_epochs(f"{dir_cleanepochs}/P{subj}_epochs_after_ic-label-epo.fif")

    ch_adjacency,_ = mne.channels.find_ch_adjacency(epochs.info, ch_type='eeg')

    threshold_tfce = dict(start=0, step=0.01)

    # 2D cluster test
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X[:,:,:], n_permutations=10000,
                                                                                    adjacency=ch_adjacency, 
                                                                                    tail=0, 
                                                                                    n_jobs=-1)
    
    good_cluster_inds = np.where(cluster_p_values < 0.05) # times where something significant happened

    print(len(good_cluster_inds))
    print(cluster_p_values) # 1D cluster test
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(np.squeeze(X[:,:,:]), n_permutations=10000,
                                                                                    tail=0, n_jobs=-1
                                                                                    )

   

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
                                        colors=colors, show=False, ci=True, combine='mean',
                                        split_legend=True, legend='lower right', truncate_yaxis='auto')

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1],
                                    color='darkgreen', alpha=0.3)

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)
        plt.savefig(f'D:\\expecon_ms\\figs\\manuscript_figures\\Figure4_evokeds\\cluster_{cond}_prev_{str(i_clu)}_.svg')
        plt.show()
