# this script provides functions that compute time-frequency representations (TFRs) for each trial

# author: Carina Forster
# email: forster@cbs.mpg.de


# import packages
import os
import random
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.stats import f_mway_rm, f_threshold_mway_rm, fdr_correction

# Specify the file path for which you want the last commit date
file_path = "D:\expecon_ms\\analysis_code\\eeg\\sensor\\tfr_contrasts.py"

last_commit_date = subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", file_path]).decode("utf-8").strip()
print("Last Commit Date for", file_path, ":", last_commit_date)

# add path to sys.path.append() if package isn't found
sys.path.append('D:\\expecon_ms\\analysis_code')

os.chdir('D:\\expecon_ms\\analysis_code')
from permutation_tests import cluster_correlation

# for plots in new windows
# %matplotlib qt

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif', 
                     'font.sans-serif': 'Arial'})
    
# datapaths
dir_cleanepochs = Path('D:/expecon_ms/data/eeg/prepro_ica/clean_epochs_corr')
behavpath = Path('D:/expecon_ms/data/behav/behav_df')

# save figure path
savedir_figure6 = Path('D:/expecon_ms/figs/manuscript_figures/figure6_tfr_contrasts')

IDlist = ['007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021', '022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046',
          '047', '048', '049']


def compute_tfr(tmin=-1, tmax=0.5, fmax=35, fmin=7, laplace=0,
                induced=False, mirror_data=0, psd=0, drop_bads=0):

    '''calculate time-frequency representations per trial
      (induced power) using multitaper method.
      Data is then saved in a tfr object per subject and stored to disk as
      a .h5 file.

        Parameters
        ----------
        tmin : float 
        tmax : float
        fmin: float
        fmax: float
        laplace: boolean, info: apply current source density transform
        induced : boolean, info: subtract evoked response from each epoch
        mirror_data : boolean, info: mirror the data on both sides to avoid edge artifacts
        psd : boolean, info: calculate power spectral density
        drop_bads : boolean, 
            info: drop epochs with abnormal strong signal (> 200 mikrovolts)
        Returns
        -------
        None
        '''

    # Define frequencies and cycles for multitaper method
    freqs = np.arange(fmin, fmax, 1)
    cycles = freqs/4.0
    
    # store behavioral data and spectra
    df_all, spectra_all = [], []

    # now loop over participants
    for idx, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)

        # load cleaned epochs (after ica component rejection)
        epochs = mne.read_epochs(f'{dir_cleanepochs}'
                                 f'/P{subj}_epochs_after_ica_0.1Hzfilter-epo.fif')

        # kick out blocks with too high or low hitrates and false alarm rates
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
            
        # remove trials with rts >= 2.5 (no response trials) and trials with reaction times < 0.1
        epochs = epochs[epochs.metadata.respt1 >= 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # load behavioral data
        data = pd.read_csv(f'{behavpath}//prepro_behav_data.csv')

        # behavioral data for current participant
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

        if drop_bads:
            # drop epochs with abnormal strong signal (> 200 mikrovolts)
            epochs.drop_bad(reject=dict(eeg=200e-6))

        # crop epopchs in desired time window
        epochs.crop(tmin=tmin, tmax=tmax)

        # apply CSD to the data (less point spread)
        if laplace:
            epochs = mne.preprocessing.compute_current_source_density(epochs)

        # avoid leakage and edge artifacts by zero padding the data
        if mirror_data:
            
            metadata = epochs.metadata
            # zero pad the data on both sides to avoid leakage and edge artifacts
            data = epochs.get_data()

            data = zero_pad_or_mirror_data(data, zero_pad=False)

            # put back into epochs structure
            epochs = mne.EpochsArray(data, epochs.info, tmin=tmin * 2)

            # add metadata back
            epochs.metadata = metadata

        # subtract evoked response from each epoch
        if induced:
            epochs = epochs.subtract_evoked()

        # compute power spectral density
        if psd:
            # calculate prestimulus power per trial
            spectrum = epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax, 
                                        tmin=tmin, tmax=tmax, n_jobs=-1,
                                        n_per_seg=256)
                
            spec_data = spectrum.get_data()
            spectra_all.append(spectrum)
            spectrum_path = Path('D:/expecon_ms/data/eeg/sensor/psd/')

            # save the spectra to disk
            np.save(f'{spectrum_path}{Path("/")}{subj}_psd.npy', spec_data)
            
        else:
    
            # Assign a sequential count for each row within each 'blocks' and 'subblock' group
            epochs.metadata['trial_count'] = epochs.metadata.groupby(['block', 'subblock']).cumcount()

            df_all.append(epochs.metadata)

            # create conditions
            epochs_a = epochs[(epochs.metadata.cue == 0.75)]
            epochs_b = epochs[(epochs.metadata.cue == 0.25)]

            # set tfr path
            tfr_path = Path('D:/expecon_ms/data/eeg/sensor/tfr')

            if os.path.exists(f'{tfr_path}{Path("/")}{subj}_high-tfr.h5'):
                    print('TFR already exists')
            else:
                tfr_a = mne.time_frequency.tfr_multitaper(epochs_a, 
                                                            freqs=freqs,
                                                            n_cycles=cycles,
                                                            return_itc=False,
                                                            n_jobs=-1, 
                                                            average=True)
                    
                tfr_b = mne.time_frequency.tfr_multitaper(epochs_b, 
                                                            freqs=freqs,
                                                            n_cycles=cycles,
                                                            return_itc=False,
                                                            n_jobs=-1, 
                                                            average=True)
                    
                tfr_a.save(f'{tfr_path}{Path("/")}{subj}_high_yes-tfr.h5', 
                        overwrite=True)
                tfr_b.save(f'{tfr_path}{Path("/")}{subj}_high_no-tfr.h5', 
                        overwrite=True)
                    
            if os.path.exists(f'{tfr_path}{Path("/")}{subj}_single_trial_power-tfr.h5'):
                print('TFR already exists')
            else:
                tfr = mne.time_frequency.tfr_multitaper(epochs, 
                                                            freqs=freqs,
                                                            n_cycles=cycles,
                                                            return_itc=False,
                                                            n_jobs=-1, 
                                                            average=False,
                                                            decim=2)
                    
                tfr.save(f'{tfr_path}{Path("/")}{subj}_single_trial_power-tfr.h5',
                            overwrite=True)

    return 'Done with tfr/erp computation'


def load_tfr_conds(cond_a='high', cond_b='low', mirror=0):

    '''load tfr data for two conditions
    Parameters
    ----------
    cond_a : str
        which condition tfr to load 
    cond_b : str
        which condition tfr to load
    mirror : boolean
            whether to load mirrored data
    Returns
    -------
    tfr_a_all: list
        list of tfr objects for condition a
    tfr_b_all: list
        list of tfr objects for condition b
    '''

    tfr_a_all, tfr_b_all = [], []

    for idx, subj in enumerate(IDlist):

        # load tfr data
        tfr_path = Path('D:/expecon_ms/data/eeg/sensor/tfr')
        if mirror:
            tfr_a = mne.time_frequency.read_tfrs(f'{tfr_path}{Path("/")}{subj}_{cond_a}_mirror-tfr.h5', condition=0)
            tfr_b = mne.time_frequency.read_tfrs(f'{tfr_path}{Path("/")}{subj}_{cond_b}_mirror-tfr.h5', condition=0)
   
        else:
            tfr_a = mne.time_frequency.read_tfrs(f'{tfr_path}{Path("/")}{subj}_{cond_a}-tfr.h5', condition=0)
            tfr_b = mne.time_frequency.read_tfrs(f'{tfr_path}{Path("/")}{subj}_{cond_b}-tfr.h5', condition=0)
   
        tfr_a_all.append(tfr_a)
        tfr_b_all.append(tfr_b)

    return tfr_a_all, tfr_b_all

def plot_tfr_cluster_test_output(channel_names=['CP4'], fmin=7, fmax=35):

    '''plot cluster permutation test output for tfr data (time and frequency cluster)
    Parameters
    channel_names : list of char 
        channels to analyze
    fmin : int
        minimum frequency to plot
    fmax : int
        maximum frequency to plot
    ----------
    Returns
    -------
    None
    '''

    # load tfr data
    tfr_a_all, tfr_b_all  = load_tfr_conds()

    # average over participants
    gra_a = mne.grand_average(tfr_a_all) # high
    gra_b = mne.grand_average(tfr_b_all) # low

    # difference between conditions (2nd level)
    diff = gra_a - gra_b
    diff.data = diff.data*10**11

    # Create a 2x3 grid of plots (2 rows, 3 columns)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # which time windows to plot
    time_windows = [(-1, -0.4), (-0.4, 0)]
    # for mirrored data
    #time_windows = [(-1, -0.2)]

    # which axes to plot the time windows
    axes_first_row = [(0,0), (0,1)]
    axes_second_row = [(1,0), (1,1)]

    # now populate first row with tfr contrasts
    for t, a in zip(time_windows, axes_first_row):

        tfr_fig = diff.copy().crop(tmin=t[0], tmax=t[1]).plot(picks=channel_names,
                                                     cmap = plt.cm.bwr,
                                                     axes=axs[a[0], a[1]], show=False)[0]

    # now plot cluster permutation output in the second row
    # pick channels
    ch_index = [tfr_a_all[0].ch_names.index(c) for c in channel_names]

    for t, a in zip(time_windows, axes_second_row):

        # contrast data
        X = np.array([h.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names)
                      .data - l.copy().
                      crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names)
                      .data for h, l in zip(tfr_a_all, tfr_b_all)])

        if len(channel_names) > 1:
            # pick channel
            X = np.mean(X, axis=1)
        else:
            X = np.squeeze(X)

        print(X.shape) # should be participants x frequencies x timepoints
        
        threshold_tfce = dict(start=0, step=0.1)

        # run cluster test over time and frequencies (no need to define adjacency)
        T_obs, clusters, cluster_p, H0 = mne.stats.permutation_cluster_1samp_test(
                                X,
                                n_jobs=-1,
                                n_permutations=10000,
                                threshold=threshold_tfce,
                                tail=0)

        if len(cluster_p) > 0:

            print(f'The minimum p-value is {min(cluster_p)}')

            good_cluster_inds = np.where(cluster_p < 0.05)
            
            if len(good_cluster_inds[0]) > 0:

                # Find the index of the overall minimum value
                min_index = np.unravel_index(np.argmin(T_obs), T_obs.shape)

                freqs = np.arange(fmin, fmax, 1)
                times = tfr_a_all[0].copy().crop(t[0], t[1]).times

        # run function to plot significant cluster in time and frequency space
        plot_cluster_test_output(T_obs = T_obs, cluster_p_values = cluster_p, clusters=clusters, fmin=7, fmax=35,
                                data_cond=tfr_a_all, tmin=t[0], tmax=t[1], ax0=a[0], ax1=a[1])

    # finally save figure
    plt.savefig(f'{savedir_figure6}{Path("/")}fig6_{cond_a}_{cond_b}_tfr_{channel_names[0]}.svg', dpi=300, format='svg')
    plt.savefig(f'{savedir_figure6}{Path("/")}fig6_{cond_a}_{cond_b}_tfr_{channel_names[0]}.png', dpi=300, format='png')


def run_3D_cluster_test():

    """run a 3D cluster permutation test (time, frequency and channels) on the difference between conditions"""

    # pick 9 channels as our cluster of channels we are interested in
    [c.pick_channels(['C6', 'C4', 'C2', 'CP2', 'CP4', 'CP6']) for c in tfr_a_all]
    [c.pick_channels(['C6', 'C4', 'C2', 'CP2', 'CP4', 'CP6']) for c in tfr_b_all]
    
    # definde adjaceny matrix for cluster permutation test
    ch_adjacency = mne.channels.find_ch_adjacency(tfr_a_all[0].info,
                                                  ch_type='eeg')
    
    # contrast data
    X = np.array([h.copy().crop(tmin,tmax).data - l.copy().crop(tmin,tmax).data for h, l in zip(tfr_a_all, tfr_b_all)])

    # frequency, time and channel adjacency
    com_adjacency = mne.stats.combine_adjacency(X.shape[2],
                                                X.shape[3], ch_adjacency[0])
    
    # change axes to match the format of the function
    X = np.transpose(X, [0, 3, 2, 1])

    # threshold free cluster enhancement
    threshold_tfce = dict(start=0, step=0.1)

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X,
                                                           n_permutations=100,
                                                           adjacency=com_adjacency,
                                                           n_jobs=-1)
    
    print(f'The minimum p-value is {min(cluster_p_values)}')

    good_cluster_inds = np.where(cluster_p_values < 0.05)

    # Find the index of the overall minimum value
    min_index = np.unravel_index(np.argmin(T_obs), T_obs.shape)

    print("Index of the overall minimum value:", min_index)

    return T_obs, clusters, cluster_p_values, H0


def plot_cluster_test_output(T_obs=None, cluster_p_values=None, clusters=None, fmin=7, fmax=35,
                             data_cond=None, tmin=0, tmax=0.5, ax0=0, ax1=0):
    
    """Plot the cluster test output in time-frequency space
    Parameters
    ----------
    T_obs : array
    T-statistic from cluster permutation test
    cluster_p_values : array
    p-values from cluster permutation test
    clusters : array
    clusters from cluster permutation test
    fmin : int
    minimum frequency to plot
    fmax : int
    maximum frequency to plot
    data_cond : list
    list of tfr objects for each condition
    tmin : int
    minimum time to plot
    tmax : int
    maximum time to plot
    ax0 : int
    index of the axes in the first row
    ax1 : int
    index of the axes in the second row
    ----------
    Returns
    -------
    None
    """

    # frequency and time range
    freqs = np.arange(fmin, fmax, 1)

    times = 1e3 * data_cond[0].copy().crop(tmin,tmax).times

    # define significant T-values (part of significant cluster)
    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]
    
    # define vmin and vmax
    min_v = np.max(np.abs(T_obs))

    vmin = -min_v
    vmax = min_v

    # plot T  values
    fig1 = axs[ax0, ax1].imshow(
        T_obs,
        cmap=plt.cm.bwr,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )

    # plot significant clusters
    fig2 = axs[ax0, ax1].imshow(
        T_obs_plot,
        cmap=plt.cm.bwr,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    # Add colorbar for each subplot
    fig.colorbar(fig2, ax = axs[ax0, ax1])

    # add axes labels
    axs[ax0, ax1].set_xlabel("Time (ms)")
    axs[ax0, ax1].set_ylabel("Frequency (Hz)")

    
def run_cluster_correlation_2D(channel_names=['CP4']):

    """run a 2D cluster correlation as implemented by Magda (correlate criterion with every voxel in TFR)
    Parameters
    ----------
    channel_names : list of char
        channels to analyze
    ----------
    Returns
    -------
    None
    """

    # load random effects from glmmer model
    re = pd.read_csv("D:\expecon_ms\data\\behav\mixed_models\\brms\\brms_betaweights.csv")
    
    # contrast data (high - low)
    X = np.array([h.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names)
                      .data - l.copy().
                      crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names)
                      .data for h, l in zip(tfr_a_all, tfr_b_all)])
    
    out = cluster_correlation.permutation_cluster_correlation_test(np.squeeze(X), 
                                                             re.cue,
                                                             test='pearson')

########################################################## Helper functions ################################################################

def zero_pad_or_mirror_data(data, zero_pad=False):
    '''
    :param data: data array with the structure epochsxchannelsxtime
    :return: array with mirrored data on both ends = data.shape[2]
    '''

    if zero_pad:
        padded_list=[]
        
        zero_pad = np.zeros(data.shape[2])
        # loop over epochs
        for epoch in range(data.shape[0]):
            ch_list = []
            # loop over channels
            for ch in range(data.shape[1]):
                # zero padded data at beginning and end
                ch_list.append(np.concatenate([zero_pad, data[epoch][ch], zero_pad]))
            padded_list.append([value for value in ch_list])
    else:
        padded_list=[]

        # loop over epochs
        for epoch in range(data.shape[0]):
            ch_list = []
            # loop over channels
            for ch in range(data.shape[1]):
                # mirror data at beginning and end
                ch_list.append(np.concatenate([data[epoch][ch][::-1], data[epoch][ch], data[epoch][ch][::-1]]))
            padded_list.append([value for value in ch_list])

    return np.squeeze(np.array(padded_list))


def permute_trials(n_permutations=500, power_a=None, power_b=None):

    """ Permute trials between two conditions and equalize trial counts,
    then average across permutations. store in TFR object."""

    # store permutations
    power_a_perm, power_b_perm = [], [] 

    for i in range(n_permutations):
        if power_b.data.shape[0] > power_a.data.shape[0]:
            random_sample = power_a.data.shape[0]
            idx_list = list(range(power_b.data.shape[0]))

            power_b_idx = random.sample(idx_list, random_sample)

            power_b.data = power_b.data[power_b_idx, :, :, :]
        else:
            random_sample = power_b.data.shape[0]
            idx_list = list(range(power_a.data.shape[0]))

            power_a_idx = random.sample(idx_list, random_sample)

            power_a.data = power_a.data[power_a_idx, :, :, :]

        evoked_power_a = power_a.average()
        evoked_power_b = power_b.average()

        power_a_perm.append(evoked_power_a)
        power_b_perm.append(evoked_power_b)

    # average across permutations
    evoked_power_a_perm_arr = np.mean(np.array([p.data for p in power_a_perm]), axis=0)
    evoked_power_b_perm_arr = np.mean(np.array([p.data for p in power_b_perm]), axis=0)

    # put back into TFR object
    evoked_power_a = mne.time_frequency.AverageTFR(power.info, evoked_power_a_perm_arr, 
                                                        power.times, power.freqs, power_a.data.shape[0])
    evoked_power_b = mne.time_frequency.AverageTFR(power.info, evoked_power_b_perm_arr, 
                                                        power.times, power.freqs, power_b.data.shape[0])

    return evoked_power_a, evoked_power_b


def equalize_epochs(array1, array2):

    """
    Equalizes the number of epochs between two arrays along the first dimension.

    Args:
        array1 (numpy.ndarray): First input array with shape (epochs1, channels, time, frequency).
        array2 (numpy.ndarray): Second input array with shape (epochs2, channels, time, frequency).

    Returns:
        numpy.ndarray, numpy.ndarray: Two arrays with equalized number of epochs.
    """
    # Calculate the number of epochs in each array
    num_epochs_array1 = array1.shape[0]
    num_epochs_array2 = array2.shape[0]

    # Determine which array has more epochs
    if num_epochs_array1 > num_epochs_array2:
        array_with_more_epochs = array1
        array_with_fewer_epochs = array2
    else:
        array_with_more_epochs = array2
        array_with_fewer_epochs = array1

    # Calculate the difference in the number of epochs
    num_epochs_diff = abs(num_epochs_array1 - num_epochs_array2)

    # Randomly select indices (epochs) from the array with more epochs to match the number of epochs in the other array
    selected_indices = np.random.choice(num_epochs_array_with_more_epochs, size=num_epochs_array_with_fewer_epochs, replace=False)

    # Use the selected indices to create a new array with the same number of epochs as the other array
    new_array_with_more_epochs = array_with_more_epochs[selected_indices, :, :, :]

    # Return the arrays with equalized number of epochs
    if array_with_more_epochs is array1:
        return new_array_with_more_epochs, array_with_fewer_epochs
    else:
        return array_with_fewer_epochs, new_array_with_more_epochs


def prepare_for_anova(channel_names=['CP4']):

    """prepare data for ANOVA analysis (function not used in EXPECON paper)"""

    n_conditions = 4
    n_replications = 43

    factor_levels = [2, 2]  # number of levels in each factor
    effects = "A*B"  # this is the default signature for computing all effects
    # Other possible options are 'A' or 'B' for the corresponding main effects
    # or 'A:B' for the interaction effect only (this notation is borrowed from the
    # R formula language)
    freqs = tfr_a_all[0].freqs
    n_freqs = len(tfr_a_all[0].freqs)

    # load all tfr or all evokeds
    all_tfr = [tfr_a_all, tfr_b_all, tfr_c_all, tfr_d_all]

    tf_all_subs = []
    for t in all_tfr:
        tf_all = []
        for tf in t:
            tf.data = tf.data*10**11
            tf = tf.copy().crop(-0.2, 0.2).data
            tf_all.append(tf)
        tf_all_subs.append(tf_all)

    tf_all_subs = []
    for t in all_evo:
        tf_all = []
        for tf in t:
            tf = tf.copy().crop(-0.4, 0).data
            tf_all.append(tf)
        tf_all_subs.append(tf_all)

    times = 1e3 * tfr_a_all[0].copy().crop(-0.2, 0.2).times
    times = all_evo[0][0].copy().crop(-0.4,0).times
    n_times = len(times)

    # conver to numpy array
    tfr_allcond = np.array(tf_all_subs)

    # pick channel
    ch_index = tfr_a_all[0].ch_names.index(channel_names[0])
    ch_index = all_evo[0][0].ch_names.index(channel_names[0])

    # dimensions: conditions, subjects, frequencies, timepoints
    tfr_allcond_cl = tfr_allcond[:,:,ch_index,:,:]

    # dimensions: conditions, subjects, channels, timepoints
    erp_allcond = tfr_allcond[:,:,ch_index,:]

    erp_allcond = np.swapaxes(erp_allcond, 1, 0)

    tfr_allcond = np.swapaxes(tfr_allcond_cl, 1, 0)

    # so we have replications × conditions × observations
    # where the time-frequency observations are freqs × times:

    fvals, pvals = f_mway_rm(tfr_allcond, factor_levels, effects=effects)

    effect_labels = ["stimulus probability", "previous choice", "stim_prob*prev_choice"]

    fig, axes = plt.subplots(3, 1, figsize=(6, 6))

    # let's visualize our effects by computing f-images
    for effect, sig, effect_label, ax in zip(fvals, pvals, effect_labels, axes):
        # show naive F-values in gray
        ax.imshow(
            effect,
            cmap="gray",
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
        )
        # create mask for significant time-frequency locations
        effect[sig >= 0.05] = np.nan
        c = ax.imshow(
            effect,
            cmap="autumn",
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
        )
        fig.colorbar(c, ax=ax)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Frequency (Hz)")
        ax.set_title(f'Time-locked response for "{effect_label}" ({channel_names})')

    fig.tight_layout()

    # cluster permutation test to account for multiple comparisons

    effects = "A"  # only interaction effect

    def stat_fun(*args):
        return f_mway_rm(
            np.swapaxes(args, 1, 0),
            factor_levels=factor_levels,
            effects=effects,
            return_pvals=False,
        )[0]


    # The ANOVA returns a tuple f-values and p-values, we will pick the former.
    pthresh = 0.05  # set threshold rather high to save some time
    f_thresh = f_threshold_mway_rm(n_replications, factor_levels, effects, pthresh)
    tail = 1  # f-test, so tail > 0

    F_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_test(
        tfr_allcond_cl,
        stat_fun=stat_fun,
        threshold=f_thresh,
        tail=tail,
        n_jobs=-1,
        n_permutations=10000,
        buffer_size=None,
        out_type="mask",
    )

    good_clusters = np.where(cluster_p_values < 0.15)[0]
    F_obs_plot = F_obs.copy()
    F_obs_plot[~clusters[np.squeeze(good_clusters)]] = np.nan

    fig, ax = plt.subplots(figsize=(6, 4))
    for f_image, cmap in zip([F_obs, F_obs_plot], ["gray", "autumn"]):
        c = ax.imshow(
            f_image,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
        )

    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        f'Time-locked response for "modality by location" ({ch_name})\n'
        "cluster-level corrected (p <= 0.05)"
    )
    fig.tight_layout()

    # use FDR correction
    mask, _ = fdr_correction(pvals[0])
    F_obs_plot2 = F_obs.copy()
    F_obs_plot2[~mask.reshape(F_obs_plot2.shape)] = np.nan

    fig, ax = plt.subplots(figsize=(6, 4))
    for f_image, cmap in zip([F_obs, F_obs_plot2], ["gray", "autumn"]):
        c = ax.imshow(
            f_image,
            cmap=cmap,
            aspect="auto",
            origin="lower",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
        )

    fig.colorbar(c, ax=ax)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(
        f'Time-locked response for "modality by location" ({channel_names})\n'
        "FDR corrected (p <= 0.05)"
    )
    fig.tight_layout()
