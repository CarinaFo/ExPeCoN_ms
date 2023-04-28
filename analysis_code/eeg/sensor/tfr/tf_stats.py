################cluster based permutation tests for time-frequency data##############################



# Functions:

# prepare_tfcontrasts: calculate time-frequency estimates using multitaper or morlet wavelets
# run_ttest: runs a paired ttest between 2 conditions and plots t values in sensor space over
#               specified time and frequency range
# cluster_test: runs a cluster based permutation test over channels,time and frequencies
# plot_cluster_output: plot significant cluster in time-frequency space for one channel,
# a mean over channels or separately for each channel
# plot_3D_cluster: plots cluster in one time-frequency plot with significant channels
# F test (4 conditions) and ANOVA not checked yet


# run out = cluster_test() to compute TF contrasts, plot topo tmap and run cluster based permutation test
# you can then plot the output running plot_cluster_output()

# load packages

import os
import mne
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, ScalarFormatter, FixedFormatter, StrMethodFormatter
from os.path import join as opj
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import stats


# set paths
epochs_dir = r"D:\expecon_ms\data\eeg\prepro_ica\clean_epochs"
savepath_tfr = r"D:\expecon_ms\data\eeg\sensor"
savepath_TF_trial = r'D:\expecon_EEG_112021\power_per_trial'
save_dir_tf = r'D:\expecon_EEG\prepro_data_tf'
save_dir = r'D:\expecon_EEG\fig\cb'
savedir_epochs = 'D:/expecon/data/eeg/epochs_after_ica_cleaning'

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')

# participant index of participants that changed their criterion according to the probabilities (higher criterion
# in low expectation condition)

# Functions

def prepare_tfcontrasts(tmin=-0.5, tmax=0, ncycles=4.0, fmin=7, fmax=30,
                        baseline=0, baseline_interval=(-0.8, -0.5), mode='mean', sfreq=250,
                        save=1, laplace=0, method='tfr_morlet', cond="hitmiss", zero_pad=1):

    """this function runs morlet wavelets or multitaper on clean,epoched data
    it allows to extract epochs based on a attribute if metadata is attached to the dataframe
    subtraction of the evoked potential is included
    UPDATE: included zero-padding (symmetric) to exclude edge and post-stimulus artifacts
    see zero_pad_data() function docstring for more infos on zero padding

    output: list of TF output per subject (saved as h5 file)
    PSD shape: n_subjects,n_epochs, n_channels, n_freqs for power spectral density
    (can be used for single trial analysis)"""

    freqs = np.arange(fmin, fmax+1,1) # define frequencies of interest
    n_cycles = freqs / ncycles  # different number of cycles per frequency

    all_tfr_a, all_tfr_b = [], []

    # loop over participants
    for counter, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)

        # skip those participants
        if subj == '040' or subj == '045':
            continue

        # load cleaned epochs
        #os.chdir("D:\expecon_ms\data\eeg\prepro_ica\clean_epochs")

        #epochs = mne.read_epochs('P' + subj + '_epochs_after_ica-epo.fif')

        epochs = mne.read_epochs(savedir_epochs + '\P' + subj + '_after_ica-epo.fif')

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
        epochs = epochs[epochs.metadata.respt1 > 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]
        # some weird trigger stuff going on?
        epochs = epochs[epochs.metadata.trial != 1]

        # create experimental conditions
        if cond == "hitmiss":

            epochs_a = epochs.__getitem__((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1))
            epochs_b = epochs.__getitem__((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 0))

            cond_a = "hit"
            cond_b = "miss"

        elif cond == "highlow":

            epochs_a = epochs[(epochs.metadata.cue == 0.75)]
            epochs_b = epochs[(epochs.metadata.cue == 0.25)]

            cond_a = "high"
            cond_b = "low"

        elif cond == "signalnoise":

            epochs_a = epochs.__getitem__(epochs.metadata.isyes == 1)
            epochs_b = epochs.__getitem__(epochs.metadata.isyes == 0)

            cond_a = "signal"
            cond_b = "noise"

        elif cond == "highlownoise":

            epochs_a = epochs.__getitem__((epochs.metadata.cue == 0.75) & (epochs.metadata.isyes == 0))
            epochs_b = epochs.__getitem__((epochs.metadata.cue == 0.25) & (epochs.metadata.isyes == 0))

            cond_a = "high_noise"
            cond_b = "low_noise"

        elif cond == "highlowsignal":

            epochs_a = epochs.__getitem__((epochs.metadata.cue == 0.75) & (epochs.metadata.isyes == 1))
            epochs_b = epochs.__getitem__((epochs.metadata.cue == 0.25) & (epochs.metadata.isyes == 1))

            cond_a = "high_signal"
            cond_b = "low_signal"

        elif cond == "confidence":

            epochs_a = epochs.__getitem__(epochs.metadata.conf == 1)
            epochs_b = epochs.__getitem__(epochs.metadata.conf == 0)

            cond_a = "confident"
            cond_b = "unconfident"

        elif cond == "confidencehits":

            epochs_a = epochs.__getitem__((epochs.metadata.conf == 1) & (epochs.metadata.sayyes == 1) & (epochs.metadata.isyes==1))
            epochs_b = epochs.__getitem__((epochs.metadata.conf == 0) & (epochs.metadata.sayyes == 1) & (epochs.metadata.isyes==1))

            cond_a = "confidenthits"
            cond_b = "unconfidenthits"

        elif cond == 'congruency':

            epochs_a = epochs.__getitem__((epochs.metadata.cue == 0.25) & (epochs.metadata.isyes == 0) |
                                          (epochs.metadata.cue == 0.75) & (epochs.metadata.isyes == 1))
            epochs_b = epochs.__getitem__(
                (epochs.metadata.cue == 0.25) & (epochs.metadata.isyes == 1) |
                (epochs.metadata.cue == 0.75) & (epochs.metadata.isyes == 0))

            cond_a = "congruent"
            cond_b = "incongruent"

        elif cond == 'correct':

            epochs_a = epochs.__getitem__((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1) |
                                          (epochs.metadata.isyes == 0) & (epochs.metadata.sayyes == 0))
            epochs_b = epochs.__getitem__(
                (epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 0) |
                (epochs.metadata.isyes == 0) & (epochs.metadata.isyes == 1))

            cond_a = "correct"
            cond_b = "incorrect"

        # induced activity in the time window of interest
        #epochs_a = epochs_a.subtract_evoked()
        #epochs_b = epochs_b.subtract_evoked()

        #crop the data in the desired time window
        epochs_a.crop(tmin, tmax)
        epochs_b.crop(tmin, tmax)

        # make sure there is the same amount of trials in both conditions
        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        # apply laplace filter if desired
        if laplace == 1:

            epochs_a = mne.preprocessing.compute_current_source_density(epochs_a)
            epochs_b = mne.preprocessing.compute_current_source_density(epochs_b)

        # extract the data from the epochs structure
        dataa = epochs_a.get_data()
        datab = epochs_b.get_data()  #epochs*channels*times

        # zero pad the data on both ends to avoid leakage and edge artifacts
        if zero_pad == 1:

            dataa = zero_pad_data(dataa)
            datab = zero_pad_data(datab)

            #put back into epochs structure

            epochs_a = mne.EpochsArray(dataa, epochs_a.info, tmin=tmin*2)

            epochs_b = mne.EpochsArray(datab, epochs_b.info, tmin=tmin*2)

        # tfr estimation using morlet wavelets or multitiaper
        if method == 'morlet':

            tfr_a = mne.time_frequency.tfr_morlet(epochs_a, freqs=freqs, n_cycles=n_cycles, return_itc=False,
                                                         n_jobs=15, output='power', average=True, use_fft=True)

            tfr_b = mne.time_frequency.tfr_morlet(epochs_b, freqs=freqs, n_cycles=n_cycles, return_itc=False,
                                                    n_jobs=15, output='power', average=True, use_fft=True)

            if baseline == 1:

                tfr_a.apply_baseline(mode=mode, baseline=baseline_interval)
                tfr_b.apply_baseline(mode=mode, baseline=baseline_interval)


            # this saves the TF representation

            all_tfr_a.append(tfr_a)
            all_tfr_b.append(tfr_b)

        else:

            tfr_a = mne.time_frequency.tfr_multitaper(epochs_a, n_cycles=ncycles, freqs=freqs, return_itc=False,
                                                      n_jobs=15)
            tfr_b = mne.time_frequency.tfr_multitaper(epochs_b, n_cycles=ncycles, freqs=freqs, return_itc=False,
                                                      n_jobs=15)

            if baseline == 1:

                tfr_a.apply_baseline(mode=mode, baseline=baseline_interval)
                tfr_b.apply_baseline(mode=mode, baseline=baseline_interval)

            # save tfr array per participant in a list and save to disk using the h5 format
            all_tfr_a.append(tfr_a)
            all_tfr_b.append(tfr_b)

    mne.time_frequency.write_tfrs(opj(savepath_tfr, method, cond_a + '-tfr.h5'), all_tfr_a,
                overwrite=True)

    mne.time_frequency.write_tfrs(opj(savepath_tfr, method, cond_b + '-tfr.h5'), all_tfr_b,
                                    overwrite=True)

    return all_tfr_a, all_tfr_b, fmin, fmax, freqs, tmin, tmax, cond_a, cond_b, epochs_a, epochs_b, method

def run_ttest():

    """run a simple t test between conditions without correcting for multiple comparisions and plot the values"""

    a, b, fmin, fmax, freqs, tmin, tmax, cond_a, cond_b, epochs_a_padded, epochs_b_padded, tf_method = prepare_tfcontrasts()

    #cluster_inp = np.load(savepath_results, cluster_inp.npy)

    acrop = [ax.copy().crop(tmin, tmax) for ax in a]
    bcrop = [bx.copy().crop(tmin, tmax) for bx in b]

    # no baseline correction of TF data

    #acrop = [a.apply_baseline((None,None), mode="mean") for a in acrop]
    #bcrop = [a.apply_baseline((None, None), mode="mean") for a in bcrop]

    info = acrop[0].info
    times = acrop[0].times

    power_a_array = np.array([a.data for a in acrop])
    power_b_array = np.array([b.data for b in bcrop])

    #zscore

    #power_a_array = [a - a.mean() / a.std() for a in power_a_array]
    #power_b_array = [a - a.mean() / a.std() for a in power_b_array]

    n_subs, n_chan, n_freqs, n_timepoints = power_a_array.shape

    stat_cond, pval_cond = scipy.stats.ttest_rel(power_a_array[:,:,:,:], power_b_array[:,:,:,:], axis=0)

    Conab = mne.time_frequency.AverageTFR(info, stat_cond, times, freqs[:], nave=len(IDlist))

    mne.viz.plot_tfr_topomap(Conab, colorbar=True, size=11, tmin=times[0], tmax=times[-1], show_names=True,
                             sphere='eeglab', show=False)

    plt.savefig(f'topomap_' + cond_a + '_' + cond_b + '_' + str(tmin) + '_' + str(tmax) + '.svg')

    return power_a_array, power_b_array,n_subs, n_chan,n_freqs, n_timepoints, stat_cond, freqs,tmin,tmax, cond_a, cond_b,\
           times, info, tf_method, acrop, a


def cluster_time_frequency(jobs=15, n_perm=10000):
    """runs a cluster permutation test over time and frequency space (2D), select channel or average over specified channels
    (ROI), uses threshold free cluster enhancement to determine cluster threshold, channels are picked based on topo plot (see above)
    Plots cluster output on top of T-map, saves figure as svg
    input array: 3D: participants, channels, timepoints"""


    info = info.copy().pick_channels(channels)

    acrop[20].pick_channels(channels)

    n_ch = len(channels)

    spec_channel_list = []

    for i, channel in enumerate(channels):
        spec_channel_list.append(a[15].ch_names.index(channel))
    spec_channel_list

    mean_over_channels = np.mean(power_a_array[:, spec_channel_list, :, :], axis=1)
    mean_over_channels_b = np.mean(power_b_array[:, spec_channel_list, :, :], axis=1)

    X = mean_over_channels - mean_over_channels_b

    threshold_tfce = dict(start=0, step=0.1)

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        X,
        n_jobs=15, n_permutations=n_perm, out_type='mask',  tail=0, threshold=threshold_tfce)

    cluster_p = cluster_p_values.reshape(24,126)
    # Apply the mask to the image
    masked_img = T_obs.copy()
    masked_img[np.where(cluster_p > 0.05)] = 0

    vmax = np.max(T_obs)
    vmin = np.min(T_obs)

    # add time on the x axis 
    x = np.linspace(-0.5,0,126)
    y = np.arange(7,31,1)

    # Plot the original image with lower transparency
    plt.imshow(T_obs, alpha=1, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto',
    vmin=vmin, vmax=vmax)

    # Plot the masked image on top
    plt.imshow(masked_img, alpha=0.2, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
    aspect='auto', vmin=vmin, vmax=vmax)

    # Add x and y labels
    plt.xlabel('time (s)')
    plt.ylabel('Freq (Hz)')

    plt.savefig('hit-miss_alpha.svg')
    # Show the plot
    plt.show()

def cluster_test(n_perm=5000, mean_over_channels=0, selected_channels=0, all_ch=1):

    """implement cluster based permutation test over channels, frequencies and time as implemented in mne"""

    power_a_array, power_b_array, n_subs, n_chan, n_freqs, n_timepoints, stat_cond, freqs, tmin, tmax, cond_a, cond_b,\
    times, info, tf_method, acrop, a = run_ttest()

    # right hemisphere channels (contralateral to stimulus)

    channels = ['C2', 'C4', 'C6', 'CP2', 'CP4', 'CP6']

    channels = ['CP6', 'CP4']


    # left hemisphere channels (contralateral to motor response)

    #channels = ['C1', 'C3', 'C5', 'CP1', 'CP3', 'CP5']

    #update info structure

    if all_ch == 1:

        n_ch = 62

        ch_matrix, ch_list = mne.channels.find_ch_adjacency(info, ch_type='eeg')

        combined_adjacency = mne.stats.combine_adjacency(ch_matrix, n_freqs)

        # our adjacency is square with each dim matching the data size
        assert combined_adjacency.shape[0] == combined_adjacency.shape[1] == \
               n_ch * n_freqs

    else:

        info = info.copy().pick_channels(channels)

        acrop[20].pick_channels(channels)

        n_ch = len(channels)

        spec_channel_list = []

        for i, channel in enumerate(channels):
            spec_channel_list.append(a[15].ch_names.index(channel))
        spec_channel_list

        ch_matrix, ch_list = mne.channels.find_ch_adjacency(info, ch_type='eeg')

        # make sure that the order matches the order of the TF array
        # subsxchannelsxfrequenciesxtimepoints

        combined_adjacency = mne.stats.combine_adjacency(ch_matrix, n_freqs, n_timepoints)
        
        # our adjacency is square with each dim matching the data size
        assert combined_adjacency.shape[0] == combined_adjacency.shape[1] == \
               n_ch * n_freqs * n_timepoints

        if mean_over_channels == 1:

            mean_over_channels = np.mean(power_a_array[:, spec_channel_list, :, :], axis=1)
            mean_over_channels_b = np.mean(power_b_array[:, spec_channel_list, :, :], axis=1)

            T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                mean_over_channels-mean_over_channels_b,
                n_jobs=15, n_permutations=n_perm, out_type='mask',  tail=0)

        elif selected_channels == 1:

            T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                power_a_array[:, ch_inds, :, :] - power_b_array[:, ch_inds, :, :],
                n_jobs=15, n_permutations=n_perm, tail=0,
                adjacency=combined_adjacency)
        else:

            T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                power_a_array[:,:, :, :] - power_b_array[:, :, :, :],
                n_jobs=15, n_permutations=n_perm, tail=0,
                adjacency=combined_adjacency, threshold=2.5)

            for i_clu in good_cluster_inds:
                # unpack cluster information, get unique indices
                ch_inds, freq_inds, time_inds = clusters[i_clu]
                ch_inds = np.unique(ch_inds)
                freq_inds = np.unique(freq_inds)
                time_inds = np.unique(time_inds)

                # get topography for t stat
                t_map = np.mean(T_obs[:, :, :], axis=(1, 2))
                # create spatial mask
                mask = np.zeros((t_map.shape[0], 1), dtype=bool)
                mask[ch_inds, :] = True
                t_evoked = mne.EvokedArray(t_map[:, np.newaxis], info, tmin=tmin)
                t_evoked.plot_topomap(times=times[0], mask=mask, cmap='Reds',
                                      vmin=np.min, vmax=np.max, show=True,
                                      colorbar=False, mask_params=dict(markersize=10))
                plt.show()

    times = times[:]

    n_timepoints = len(times)

    freqs=freqs[:]

    n_freqs = len(freqs)

    # n_comparisons = len(power_a_array)  # L auditory vs L visual stimulus
    # n_conditions = power_a_array.shape[0]  # 55 epochs per comparison
    # threshold = stats.distributions.t.ppf(
    #     1 - p_accept, n_comparisons - 1, n_conditions - 1)

    mean_over_time_a = np.mean(power_a_array[:, :, :, :], axis=3)
    mean_over_time_b = np.mean(power_b_array[:, :, :, :], axis=3)

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        mean_over_time_a - mean_over_time_b,
        n_jobs=15, n_permutations=n_perm, tail=0, adjacency=combined_adjacency, threshold=2.5)

        # where is the cluster located (which channels)
        # Create new stats image with only significant clusters

    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]

    plt.imshow(T_obs_plot.T)
    plt.show()

    good_cluster_inds = np.where(cluster_p_values < p_accept)[0]

    for i_clu in good_cluster_inds:
        # unpack cluster information, get unique indices
        ch_inds, freq_inds = clusters[i_clu]
        ch_inds = np.unique(ch_inds)
        freq_inds = np.unique(freq_inds)

        # get topography for t stat

        t_map = T_obs[:, freq_inds].mean(axis=1)
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True
        t_evoked = mne.EvokedArray(t_map[:, np.newaxis], info, tmin=tmin)
        t_evoked.plot_topomap(times=times[0], mask=mask,
                              vmin=np.min, vmax=np.max, show=False,
                              colorbar=False, mask_params=dict(markersize=10))

        plt.savefig("topo.svg", dpi=600)
        plt.show()

    print(cluster_p_values[cluster_p_values < 0.05])

    return T_obs, clusters, cluster_p_values, H0, times, acrop, channels, freqs, stat_cond, tf_method, cond_a, cond_b, tmin, tmax

def plot_cluster_output(average=1, plot_one_channel=0, specific_channel=0, channel_to_plot="C4",
                        channels=['C2', 'C4', 'C6', 'CP2', 'CP4', 'CP6']):
    '''
    plot tfr figure for one specific channel, the channel with the highest
    t-value or for a cluster of channels inspired by Sebastian Speers code (he used the t-values
    from the ttest and masks the significant cluster by changing the luminance value alpha
    '''

    T_obs, clusters, cluster_p_values, H0, times, acrop, channels, freqs, stat_cond, tf_method, cond_a, cond_b, tmin, tmax = out

    if plot_one_channel == 1:

        if average == 1:

            plt.figure()

            # Create new stats image with only significant clusters
            T_obs_plot = np.nan * np.ones_like(T_obs)
            for c, p_val in zip(clusters, cluster_p_values):
                if p_val <= 0.05:
                    T_obs_plot[c] = T_obs[c]

            vmax = np.max(np.abs(T_obs))
            vmin = -vmax
            plt.subplot(2, 1, 1)
            plt.imshow(T_obs, cmap=plt.cm.gray,
                       extent=[times[0], times[-1], freqs[0], freqs[-1]],
                       aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            plt.imshow(T_obs_plot, cmap=plt.cm.RdBu_r,
                       extent=[times[0], times[-1], freqs[0], freqs[-1]],
                       aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.xlabel('Time (ms)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'power averaged over centro-parietal channels')
            plt.show()

        else:

            plt.figure()

            # Create new stats image with only significant clusters
            T_obs_plot = np.nan * np.ones_like(T_obs)
            for c, p_val in zip(clusters, cluster_p_values):
                if p_val <= 0.05:
                    T_obs_plot[c] = T_obs[c]

            # Just plot one channel's data (with the strongest effect)

            ch_idx, f_idx, t_idx = np.unravel_index(
                np.nanargmax(np.abs(T_obs_plot)), acrop[15].data.shape)

            if specific_channel == 1:

                ch_idx = acrop[10].ch_names.index(channel_to_plot)  # to show a specific one

            vmax = np.max(np.abs(T_obs))
            vmin = -vmax
            plt.subplot(2, 1, 1)
            plt.imshow(T_obs[ch_idx], cmap=plt.cm.RdBu_r,
                       extent=[times[0], times[-1], freqs[0], freqs[-1]],
                       aspect='auto', origin='lower', vmin=vmin, vmax=vmax, alpha=0.5)
            plt.imshow(T_obs_plot[ch_idx], cmap=plt.cm.RdBu_r,
                       extent=[times[0], times[-1], freqs[0], freqs[-1]],
                       aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
            plt.colorbar()
            plt.xlabel('Time (ms)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'Induced power ({acrop[15].ch_names[ch_idx]})')
            plt.show()

    else:
        # for this method of plotting outtype == mask is requiered
        #T_obs, clusters, cluster_p_values, H0,times, epochs, channels, freqs, stat_cond = cluster_test()

        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(9, 5))

        ax = ax.ravel()

        fig.text(0.5, 0.04, 'Time', ha='center', fontsize=12)
        fig.text(0.04, 0.5, 'Frequency', va='center', rotation='vertical', fontsize=12)

        cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
        fig.subplots_adjust(right=0.9)

        # set x ticks
        num_ticks = 3
        # the index of the position of yticks
        xticks = np.linspace(0, len(times) - 1, num_ticks, dtype=np.int64)
        # the content of labels of these yticks
        xticklabels = [times[idx] for idx in xticks]
        xticklabels = [i for i in xticklabels]

        for i, channel in enumerate(channels_new):

            print(str(i))

            ch_idx = acrop[0].ch_names.index(channel)  # to show a specific one

            stat_channel = stat_cond[:,:,:][ch_idx]

            print(ch_idx)

            # Create new stats image with only significant clusters

            T_obs_plot = np.nan * np.ones_like(T_obs)
            for c, p_val in zip(clusters, cluster_p_values):
                if p_val <= 0.05:
                    T_obs_plot[c] = T_obs[c]

            vmax = np.max(np.abs(T_obs))
            vmin = -vmax
            majorLocator = MultipleLocator(150)

            # all values
            df = pd.DataFrame(stat_channel, columns=times, index=np.round(freqs[:], 1))
            #sns.set(font_scale=1.3)
            sns.heatmap(df, cmap='coolwarm', vmin=vmin, vmax=vmax, xticklabels=xticklabels, yticklabels=2, alpha=0.5,
                        ax=ax[i],
                        cbar=i == 0,
                        cbar_ax=None if i else cbar_ax)

            ax[i].invert_yaxis()

            # significant values
            df_tstat = pd.DataFrame(T_obs_plot[i], columns=times, index=np.round(freqs[:], 1))

            sns.heatmap(df_tstat, cmap='coolwarm', vmin=vmin, vmax=vmax, xticklabels=xticklabels,yticklabels=2,  ax=ax[i],
                        cbar=i == 0,
                        cbar_ax=None if i else cbar_ax)

            plt.xticks(rotation=0.45)  # Rotates X-Axis Ticks by 45-degrees
            ax[i].invert_yaxis()
            ax[i].set_xticks(xticks)
            ax[i].set_title(channel, fontsize=12)

        plt.show(block=False) # if you don't block the figure gets destroyed
        #plt.ion() #this would also work (check figure before saving)

        os.chdir(save_dir)

        fig.savefig('cbPerm_' + tf_method + '_' + cond_a + '_' + cond_b + '_' + str(tmin) + '_' + str(tmax) + '.svg', format='svg', dpi=300)

        print("saved figure succesfully in " + save_dir)

def plot_3dcluster(p_accept=0.05):

    """this function plots significant electrodes and TF plots for 3D cluster based permutation test
    works only if out_type != mask"""

    good_cluster_inds = np.where(cluster_p_values < p_accept)[0]

    for i_clu in good_cluster_inds:
        # unpack cluster information, get unique indices
        ch_inds, freq_inds, time_inds = clusters[i_clu]
        ch_inds = np.unique(ch_inds)
        time_inds = np.unique(time_inds)
        freq_inds = np.unique(freq_inds)

        # get topography for t stat
        t_map = T_obs[:,freq_inds,:].mean(axis=1)
        t_map = t_map[:,time_inds].mean(axis=1)

        # get signals at the sensors contributing to the cluster
        sig_times = times[time_inds]

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # plot average test statistic and mark significant sensors
        t_evoked = mne.EvokedArray(t_map[:, np.newaxis], info, tmin=tmin)
        t_evoked.plot_topomap(times=times[0], mask=mask, axes=ax_topo,
                              show=False,
                              colorbar=False, mask_params=dict(markersize=10))
        image = ax_topo.images[0]

        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            'Averaged t-map ({:0.3f} - {:0.3f} s)'.format(*sig_times[[0, -1]]))

        # add new axis for spectrogram
        ax_spec = divider.append_axes('right', size='300%', pad=1.2)
        title = 'Cluster #{0}, {1} spectrogram'.format(i_clu + 1, len(ch_inds))
        if len(ch_inds) > 1:
            title += " (max over channels)"
        T_obs_plot = T_obs[ch_inds].max(axis=0)
        T_obs_plot_sig = np.zeros(T_obs_plot.shape) * np.nan
        T_obs_plot_sig[tuple(np.meshgrid(freq_inds, time_inds))] = \
            T_obs_plot[tuple(np.meshgrid(freq_inds, time_inds))]

        for f_image, cmap in zip([T_obs_plot, T_obs_plot_sig], ['gray', 'autumn']):
            c = ax_spec.imshow(f_image, aspect='auto', origin='lower', cmap=cmap,
                               extent=[times[0], times[-1],
                                       freqs[0], freqs[-1]])
        ax_spec.set_xlabel('Time (ms)')
        ax_spec.set_ylabel('Frequency (Hz)')
        ax_spec.set_title(title)

        # add another colorbar
        ax_colorbar2 = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(c, cax=ax_colorbar2)
        ax_colorbar2.set_ylabel('t-stat')

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=.05)

        plt.show(block=False) # if you don't block the figure gets destroyed
        #plt.ion() #this would also work (check figure before saving)

        fig.savefig('cbPerm_' + cond_a + '_' + cond_b + '_' + str(tmin) + '_' + str(tmax) + '.svg', format='svg', dpi=600)


def permutation_anova(idlist=IDlist, a='low_noise', b='high_noise', c='low_signal', d='high_signal',
                      tmin=-1, tmax=0, n_perm=5000, baseline_epochs=(-0.3, -0.01),
                      threshold_tfce=dict(start=0, step=0.1), ch_name='CP4', tail=1, fmin = 7, fmax=35,
                      ncycles=4.0):

    """Non-parametric cluster-level F test for spatio-temporal data. Allows you to compare more than 2 conditions.
    This function provides a convenient wrapper for mne.stats.permutation_cluster_test.
    for use with data organized in the form (observations(subjects or trials) × time × space) per array.
    Arrays per condition are then stored in a list"""



    # Factor to down-sample the temporal dimension of the TFR computed by
    # tfr_morlet.

    decim = 2

    freqs = np.arange(fmin, fmax+1,1)
    n_cycles = freqs / ncycles  # different number of cycle per frequency

    a_evoked, b_evoked, c_evoked, d_evoked = [], [], [], []
    time_range = int(abs(tmin) + tmax)

    power_all = np.empty((43, 4, len(freqs), 51))

    cond_list = [a, b, c, d]

    for counter, id in enumerate(idlist):
        epochs_power = list()

        epochs_cond = list()

        for counter_cond, cond in enumerate(cond_list):
            print('Processing subject' + id + cond)

            os.chdir(savepath_epochs)

            epochs = mne.read_epochs('P' + id + '_epoch_' + cond + '-epo.fif')

            epochs.crop(0, 0.5)

            epochs.pick_channels([ch_name])

            epochs_cond.append(epochs)

        mne.epochs.equalize_epoch_counts([epochs_cond[0], epochs_cond[1], epochs_cond[2], epochs_cond[3]])

        for counter_cond, cond in enumerate(cond_list):

            this_tfr = mne.time_frequency.tfr_multitaper(epochs_cond[counter_cond], freqs, n_cycles=n_cycles,
                                                     decim=decim, average=True, return_itc=False)

            this_tfr.crop(tmin, tmax)

            this_power = this_tfr.data[0, :, :]  # shape is 8 (freq) x 313 (timepoints)

            epochs_power.append(this_power)

        power_arr = np.array([elem for elem in epochs_power])  # 4x8x313, conditions,freq,timepoints

        epochs_power[:] = []

        power_all[counter, :, :, :] = power_arr

        cluster_inp = np.split(power_all, 4, 1)

        cluster_inp = [c[:, 0, :, :] for c in cluster_inp]

        print(cluster_inp[1].shape)

        # Now set up Anova

    n_conditions = len(cond_list)
    n_replications = 43

    factor_levels = [2, 2]  # number of levels in each factor
    effects = 'A*B'  # this is the default signature for computing all effects
    # Other possible options are 'A' or 'B' for the corresponding main effects
    # or 'A:B' for the interaction effect only (this notation is borrowed from the
    # R formula language)
    n_freqs = len(freqs)
    times = 1e3 * this_tfr.times
    print(times)
    n_times = len(times)

    print(power_all.shape)

    # reshape last two dimensions in one mass-univariate observation-vector
    data = power_all.reshape(n_replications, n_conditions, n_freqs * n_times)

    # so we have replications * conditions * observations:
    print(data.shape)

    fvals, pvals = f_mway_rm(data, factor_levels, effects=effects)

    effect_labels = ['signal_noise', 'cons_lib', 'signal by condition']

    # let's visualize our effects by computing f-images
    for effect, sig, effect_label in zip(fvals, pvals, effect_labels):
        plt.figure()
        # show naive F-values in gray
        plt.imshow(effect.reshape(n_freqs, n_times), cmap=plt.cm.gray, extent=[times[0],
                                                                               times[-1], freqs[0], freqs[-1]],
                   aspect='auto',
                   origin='lower')
        # create mask for significant Time-frequency locations
        effect[sig >= 0.05] = np.nan
        plt.imshow(effect.reshape(n_freqs, n_times), cmap='RdBu_r', extent=[times[0],
                                                                            times[-1], freqs[0], freqs[-1]],
                   aspect='auto',
                   origin='lower')
        plt.colorbar()
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        plt.title(r"Time-locked response for '%s' (%s)" % (effect_label, ch_name))
        plt.show()

        # Implement ANOVA cluster based permutation test (expect as input a list with the 4 conditions and shape of
        # n_observationsxfrequenciesxtime
        #
        effects = 'A:B'

        #
        def stat_fun(*args):
            return f_mway_rm(np.swapaxes(args, 1, 0), factor_levels=factor_levels,
                             effects=effects, return_pvals=False)[0]

        #
        # # The ANOVA returns a tuple f-values and p-values, we will pick the former.
        pthresh = 0.05  # set threshold rather high to save some time
        f_thresh = f_threshold_mway_rm(n_replications, factor_levels, effects,
                                       pthresh)

        T_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
            cluster_inp, stat_fun=stat_fun, threshold=f_thresh, tail=tail, n_jobs=1,
            n_permutations=n_perm, buffer_size=None, out_type='mask')

        good_clusters = print(np.where(cluster_p_values < .05)[0])
        T_obs_plot = T_obs.copy()
        T_obs_plot[~clusters[np.squeeze(good_clusters)]] = np.nan

        plt.figure()
        for f_image, cmap in zip([T_obs, T_obs_plot], [plt.cm.gray, 'RdBu_r']):
            plt.imshow(f_image, cmap=cmap, extent=[times[0], times[-1],
                                                   freqs[0], freqs[-1]], aspect='auto',
                       origin='lower')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        plt.title("Time-locked response for 'signal vs condition' (%s)\n"
                  " cluster-level corrected (p <= 0.05)" % ch_name)
        plt.show()

        #
        #
        # #FDR corrected
        #
        mask, _ = fdr_correction(pvals[2])
        T_obs_plot2 = T_obs.copy()
        T_obs_plot2[~mask.reshape(T_obs_plot.shape)] = np.nan

        plt.figure()
        for f_image, cmap in zip([T_obs, T_obs_plot2], [plt.cm.gray, 'RdBu_r']):
            if np.isnan(f_image).all():
                continue  # nothing to show
            plt.imshow(f_image, cmap=cmap, extent=[times[0], times[-1],
                                                   freqs[0], freqs[-1]], aspect='auto',
                       origin='lower')

        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency (Hz)')
        plt.title("Time-locked response for 'modality by location' (%s)\n"
                  " FDR corrected (p <= 0.05)" % ch_name)
        plt.show()

########################################################## Helper functions

def zero_pad_data(data):
    '''
    :param data: data array with the structure channelsxtime
    :return: array with zeros padded to both sides of the array with length = data.shape[2]
    '''

    zero_pad = np.zeros(data.shape[2])

    padded_list=[]

    for epoch in range(data.shape[0]):
        ch_list = []
        for ch in range(data.shape[1]):
            ch_list.append(np.concatenate([zero_pad,data[epoch][ch],zero_pad]))
        padded_list.append([value for value in ch_list])

    return np.squeeze(np.array(padded_list))
