#################################################################################################
# investigate pre-stimulus power
##################################################################################################

# import packages
import os
import pickle
import random
import sys

# add path to sys.path.append() if package isn't found
sys.path.append('D:\\expecon_ms\\analysis_code')

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
from behavioral_data_analysis import figure1
from permutation_tests import cluster_correlation

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# datapaths
tfr_single_trial_power_dir = "D:\\expecon_ms\\data\\eeg\\sensor\\induced_tfr\\single_trial_power"
tfr_contrast_dir = "D:\\expecon_ms\\data\\eeg\\sensor\\induced_tfr\\condition_contrasts"
savedir_figure4 = 'D:\\expecon_ms\\figs\\manuscript_figures\\Figure4'
dir_cleanepochs = "D:\\expecon_ms\\data\\eeg\\prepro_ica\\clean_epochs"
behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')


def calculate_power_per_trial(tmin=-0.5, tmax=0,
                              zero_pad=1, reject_criteria=dict(eeg=200e-6),
                              flat_criteria=dict(eeg=1e-6)):

    '''calculate prestimulus time-frequency representations per trial
      (induced power) using multitaper method. Data is cut between tmin and
      tmax and zero padded on both ends to prevent edge artifacts and poststimulus
      leakage. Data is then saved in a tfr object per subject and stored to disk as
        a .h5 file.

        Parameters
        ----------
        tmin : float crop the data in the pre-stimulus window
        tmax : float
        zero_pad : int zero pad the data on both sides to avoid leakage and edge artifacts
        reject_criteria : dict reject bad epochs based on maximium absolute amplitude
        flat_criteria : dict    reject bad epochs based on minimum amplitude
        Returns

        -------
        None

        '''
    
    freq_list = np.logspace(*np.log10([6, 35]), num=12)
    n_cycles = freq_list / 2.0  # different number of cycle per frequency

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

        # subtract evoked response
        epochs = epochs.subtract_evoked()

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

        # save behavioral data after bad epochs rejection
        data = pd.read_csv(f"{behavpath}//prepro_behav_data_after_rejectepochs.csv")

        # crop the data in the pre-stimulus window
        epochs.crop(tmin, tmax)

        # zero pad the data on both sides to avoid leakage and edge artifacts
        data = epochs.get_data()

        if zero_pad == 1:

            data = zero_pad_data(data)

            # put back into epochs structure

            epochs = mne.EpochsArray(data, epochs.info, tmin=tmin * 2)

        # calculate prestimulus power per trial
        power = mne.time_frequency.tfr_multitaper(epochs, n_cycles=n_cycles, freqs=freq_list, return_itc=False,
                                                  n_jobs=-1, average=False)
        # save the tfr object to disk
        mne.time_frequency.write_tfrs(f"{tfr_single_trial_power_dir}//{subj}_power_per_trial-tfr.h5", power)


def contrast_conditions():
    
    '''calculate the difference between conditions for each subject
    average across trials and calculate grand average over participants
    Parameters
    ----------
    None
    Returns
    list containing the difference between conditions for each subject
    '''

    diff_all_subs_highlow, diff_all_subs_hitmiss = [], []
    evo_high_all, evo_low_all = [], []

    # load behavioral data 
    # (make sure has the same amount of trials as epochs)

    data = pd.read_csv(f"{behavpath}//prepro_behav_data_after_rejectepochs.csv")

    for counter, subj in enumerate(IDlist):

        # skip those participants
        if subj == '040' or subj == '045' or subj == '032' or subj == '016':
            continue

        power = mne.time_frequency.read_tfrs(f"{tfr_single_trial_power_dir}\\{subj}_power_per_trial-tfr.h5")[0]
        
        # add metadata
        subj_data = data[data.ID == counter+7]

        power.metadata = subj_data

        # get high and low expectation trials
        power_high = power[((power.metadata.cue == 0.75) &
                           (power.metadata.isyes == 0) &
                           (power.metadata.sayyes == 0))]
        power_low = power[((power.metadata.cue == 0.25) &
                          (power.metadata.isyes == 0) &
                          (power.metadata.sayyes == 0))]

        # randomly sample from low power trials to match number of trials
        #  in high power condition (equalize epoch counts mne not supported
        # for tfrepochs object (yet))
        random_sample = power_high.data.shape[0]
        idx_list = list(range(power_low.data.shape[0]))

        power_low_idx = random.sample(idx_list, random_sample)

        power_low.data = power_low.data[power_low_idx, :, :, :]

        # get hit and miss trials
        power_hit = power[((power.metadata.isyes == 1) &
                          (power.metadata.sayyes == 1))]
        power_miss = power[((power.metadata.isyes == 1) &
                           (power.metadata.sayyes == 0))]
        # average across trials
        evoked_power_hit = power_hit.average()
        evoked_power_miss = power_miss.average()

        evoked_power_high = power_high.average()
        evoked_power_low = power_low.average()

        evo_high_all.append(evoked_power_high)
        evo_low_all.append(evoked_power_low)

        # calculate the difference between conditions
        diff_highlow = evoked_power_low - evoked_power_high

        diff_hitmiss = evoked_power_hit - evoked_power_miss

        # save the difference between conditions for each subject
        diff_all_subs_highlow.append(diff_highlow)
        diff_all_subs_hitmiss.append(diff_hitmiss)

    # Save the list to a file
    with open(f'{tfr_contrast_dir}//diff_all_subs_lowhigh.pickle', 'wb') as file:
        pickle.dump(diff_all_subs_highlow, file)
        
    with open(f'{tfr_contrast_dir}//diff_all_subs_hitmiss.pickle', 'wb') as file:
        pickle.dump(diff_all_subs_hitmiss, file)

    return diff_all_subs_highlow, diff_all_subs_hitmiss

def plot_grand_average(cond='hitmiss',
                       tmin=-0.5, tmax=0,
                       channel_names=['CP4', 'CP6', 'C4', 'C6']):

    '''plot the grand average of the difference between conditions
    after loading the contrast data using pickle
    Parameters
    ----------
    cond : str
        condition to plot (hitmiss or highlow)
    tmin : float
    tmax : float
    channels : list of char (channel names)
    Returns
    -------
    None
    '''

    # Load the list from the file
    with open(f'{tfr_contrast_dir}\\'
              f'diff_all_subs_{cond}.pickle', 'rb') as file:
        contrast_data = pickle.load(file)

    diff_gra = mne.grand_average(contrast_data).apply_baseline(
                                 baseline=(-1, -0.8))

    diff_gra.copy().crop(tmin, tmax).plot_joint()

    topo = diff_gra.copy().crop(tmin, tmax).plot_topo()

    tfr = diff_gra.copy().crop(tmin, tmax).plot(
          picks=channel_names, combine='mean')[0]

    topo.savefig(f'{savedir_figure4}//{cond}_topo.svg')
    
    tfr.savefig(f'{savedir_figure4}//{cond}_tfr.svg')


def run_2D_cluster_perm_test(channel_names=['CP4', 'CP6', 'C4', 'C6'],
                             n_perm=10000):
    
    '''run a 2D cluster permutation test on the difference between conditions
    Parameters
    ----------
    channel_names : list of char (channel names)
    contrast_data : list
        list containing the difference between conditions for each subject
    n_perm : int (how many permutations for cluster test)
    Returns
    -------
    None
    '''

    spec_channel_list = []

    for i, channel in enumerate(channel_names):
        spec_channel_list.append(contrast_data[15].ch_names.index(channel))
    spec_channel_list

    X = np.array([d.crop(-0.5, 0).data for d in contrast_data])

    mean_over_channels = np.mean(X[:, spec_channel_list, :, :], axis=1)

    threshold_tfce = dict(start=0, step=0.05)

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                            mean_over_channels,
                            n_jobs=-1,
                            n_permutations=n_perm,
                            threshold=threshold_tfce,
                            tail=0,
                            seed=1234)
    
    print(f'The minimum p-value is {min(cluster_p_values)}')
    
    return T_obs, clusters, cluster_p_values, H0, mean_over_channels

def plot2D_cluster_output(cond='hitmiss', tmin=-0.5, tmax=0):

    '''plot the output of the 2D cluster permutation test
    Parameters
    ----------
    cond : str
        condition to plot (hitmiss or highlow)
    tmin : float
    tmax : float
    Returns
    -------
    None
    '''

    cluster_p_values = cluster_p_values.reshape(mean_over_channels.shape[1]
                                                ,mean_over_channels.shape[2])

    # Apply the mask to the image
    masked_img = T_obs.copy()
    masked_img[np.where(cluster_p_values > 0.05)] = 0

    vmax = np.max(0)
    vmin = np.min(T_obs)

    #cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
    
    # add time on the x axis 
    x = np.linspace(-0.5, 0, 126)
    y = np.logspace(*np.log10([6, 35]), num=12)

    # Plot the original image
    fig = plt.imshow(T_obs, origin='lower',
                     extent=[x.min(), x.max(), y.min(), y.max()],
                     aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar()

    # Plot the masked image on top with lower transparency
    fig = plt.imshow(masked_img, origin='lower', alpha=0.7,
                     extent=[x.min(), x.max(), y.min(), y.max()],
                     aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')

    # Add x and y labels
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')

    # save figure
    os.chdir(savedir_figure4)

    fig.figure.savefig(f"cluster_perm_{cond}_{str(tmin)}_{str(tmax)}.svg")

    # Show the plot
    plt.show()

# Works but now nice plotting implemented yet, potentially also overkill
# to run 3D cluster permutation test

def run_3D_cluster_perm_test(contrast_data=None):

    X = np.array([d.crop(-0.5, 0).data for d in contrast_data])

    # run 3D cluster permutation test
    # definde adjaceny matrix for cluster permutation test
   
    ch_adjacency = mne.channels.find_ch_adjacency(contrast_data[0].info,
                                                  ch_type='eeg')
    # frequency, time and channel adjacency
    com_adjacency = mne.stats.combine_adjacency(X.shape[2],
                                                X.shape[3], ch_adjacency[0])
    
    # change axes to match the format of the function
    X = np.transpose(X, [0, 3, 2, 1])

    # threshold free cluster enhancement
    threshold_tfce = dict(start=0, step=0.1)

    cluster_out = mne.stats.permutation_cluster_1samp_test(X,
                                                           n_permutations=1000,
                                                           adjacency=com_adjacency,
                                                           threshold=threshold_tfce,
                                                           n_jobs=-1)

def run_cluster_correlation_2D(cond='lowhigh', channel_names=['CP4', 'CP6', 'C4', 'C6']):

    # load behavioral data
    df = pd.read_csv(f"{behavpath}//prepro_behav_data.csv")

    out = figure1.prepare_behav_data(exclude_high_fa=False)

    # low - high exp. condition

    crit = out[0]

    dprime = out[1]

    # Load the contrast list from disk

    with open(f'{tfr_contrast_dir}\\'
              f'diff_all_subs_{cond}.pickle', 'rb') as file:
        contrast_data = pickle.load(file)

    spec_channel_list = []

    for i, channel in enumerate(channel_names):
        spec_channel_list.append(contrast_data[15].ch_names.index(channel))
    spec_channel_list

    X = np.array([d.crop(-0.5, 0).data for d in contrast_data])

    mean_over_channels = np.mean(X[:, spec_channel_list, :, :], axis=1)

    cluster_correlation.permutation_cluster_correlation_test(mean_over_channels, crit)

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