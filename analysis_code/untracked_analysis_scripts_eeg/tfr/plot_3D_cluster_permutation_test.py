# plot_3D_cluster: plots cluster in one time-frequency plot with significant channels
# F test (4 conditions) and ANOVA not checked yet

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

def plot_3dcluster(p_accept=0.05):

    """this function plots significant electrodes and TF plots for 3D cluster based permutation test
    works only if out_type != mask
    based on Code by Sebastian Speer"""

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
