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
import pyvistaqt  # for proper 3D plotting
from behavioral_data_analysis import figure1

# for plots in new windows
# %matplotlib qt

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 
                     'font.sans-serif': 'Arial'})

# datapaths
tfr_single_trial_power_dir = "D:\\expecon_ms\\data\\eeg\\sensor\\induced_tfr\\single_trial_power"
tfr_contrast_dir = "D:\\expecon_ms\\data\\eeg\\sensor\\induced_tfr\\condition_contrasts"
savedir_figure4 = 'D:\\expecon_ms\\figs\\manuscript_figures\\Figure4'
dir_cleanepochs = "D:\\expecon_ms\\data\\eeg\\prepro_ica\\clean_epochs"
behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021','022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046',
          '047', '048', '049')


def calculate_power_per_trial(tmin=-1, tmax=0.5,
                              zero_pad=0, reject_criteria=dict(eeg=200e-6),
                              flat_criteria=dict(eeg=1e-6), 
                              induced_power=0):

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
    
    freq_list = np.arange(6, 35, 1)
    n_cycles = freq_list / 2.0  # different number of cycle per frequency

    df_all = []

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

        if induced_power == 1:
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

        metadata = epochs.metadata

        df_all.append(metadata)

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
        mne.time_frequency.write_tfrs(f"{tfr_single_trial_power_dir}//{subj}_power_per_trial_evoked_nozeropad-tfr.h5", power)

    df = pd.concat(df_all)
    df.to_csv(f"{behavpath}//prepro_behav_data_after_rejectepochs.csv")

def contrast_conditions(cond='highlow', cond_a='high',
                        cond_b='low', induced_power=0,
                        tmin=-1, tmax=0.5):
    
    '''calculate the difference between conditions for each subject
    average across trials and calculate grand average over participants
    for each frequency and time point. Data is saved to disk as a .h5 file.
    
    Parameters
    ----------
    cond : str
        condition to be contrasted
    cond_a : str
    cond_b : str
    induced_power : int
        1 if induced power should be calculated, 0 if evoked power should be calculated
    tmin : float
    tmax : float
    Returns
    -------
    None
    '''

    evokeds_high, evokeds_low = [], []

    # load behavioral data 
    # (make sure has the same amount of trials as epochs)

    data = pd.read_csv(f"{behavpath}//prepro_behav_data_after_rejectepochs.csv")

    for counter, subj in enumerate(IDlist):
       
        # save induced power
        power_induced_all = []

        # skip those participants
        if subj == '040' or subj == '045' or subj == '032' or subj == '016':
            continue

        power = mne.time_frequency.read_tfrs(f"{tfr_single_trial_power_dir}\\{subj}_power_per_trial_evoked_nozeropad-tfr.h5")[0]
        
        # add metadata
        subj_data = data[data.ID == counter+7]

        power.metadata = subj_data

        power = power.copy().crop(tmin, tmax)

        if induced_power == 1:

            # subtract evoked response
            power_evoked = power.average()

            for epoch in range(power.data.shape[0]):
                power_trial = power.data[epoch,:,:,:]
                power_induced = power_trial - power_evoked.data
                power_induced_all.append(power_induced)
            
            power_epochs = np.array(power_induced_all)

            # put back into epochs structure

            power = mne.time_frequency.EpochsTFR(data=power_epochs,
                                                 info=power.info,
                                                 times=power.times,
                                                 freqs=power.freqs,
                                                 metadata=power.metadata)

        if cond == 'highlow':
            # get high and low probability trials 
            power_a = power[((power.metadata.cue == 0.75))]
            power_b = power[((power.metadata.cue == 0.25))]
        elif cond == 'prevchoice':
            # get previous no and previous yes trials
            power_a = power[(power.metadata.prevsayyes == 1)]
            power_b = power[(power.metadata.prevsayyes == 0)]
        elif cond == 'hitmiss':
            # get hit and miss trials
            power_a = power[((power.metadata.isyes == 1) &
                            (power.metadata.sayyes == 1))]
            power_b = power[((power.metadata.isyes == 1) &
                            (power.metadata.sayyes == 0))]
        
        # average across trials
        evoked_power_a = power_a.average()
        evoked_power_b = power_b.average()

        evokeds_high.append(evoked_power_a)
        evokeds_low.append(evoked_power_b)

    # Save the list to a file
    with open(f'{tfr_contrast_dir}//{cond_a}_all_subs_evoked_nzp.pickle', 'wb') as file:
        pickle.dump(evokeds_high, file)

    with open(f'{tfr_contrast_dir}//{cond_b}_all_subs_evoked_nzp.pickle', 'wb') as file:
        pickle.dump(evokeds_low, file)

    return evokeds_high, evokeds_low

def plot_grand_average(cond_a='high', cond_b='low',
                       tmin=-0.8, tmax=0.2,
                       channel_names=['CP4', 'C4', 'C6', 'CP6']):

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
              f'{cond_a}_all_subs_evoked_nzp.pickle', 'rb') as file:
        evokeds_high = pickle.load(file)

    with open(f'{tfr_contrast_dir}\\'
              f'{cond_b}_all_subs_evoked_nzp.pickle', 'rb') as file:
        evokeds_low = pickle.load(file)

        
    out = figure1.prepare_behav_data()
        
    dprime = out[0]

    crit = out[1]   
    
    # baseline correction
    evokeds_high = [e.apply_baseline((-0.8,-0.6), mode='zscore') for e in evokeds_high]
    evokeds_low = [e.apply_baseline((-0.8,-0.6), mode='zscore') for e in evokeds_low]

    # contrast data
    contrast_data = np.array([h.crop(tmin,tmax).pick_channels(channel_names).data-l.crop(tmin,tmax).pick_channels(['CP4']).data for h,l in zip (evokeds_high, evokeds_low)])
    times = evokeds_high[0].times

    high_data = np.array([h.crop(tmin,tmax).pick_channels(channel_names).data for h in evokeds_high])
    low_data = np.array([l.crop(tmin,tmax).pick_channels(channel_names).data for l in evokeds_low])

    # average over alpha band and participants
    alpha_high = np.mean(high_data[:,:,2:8,:], axis=(0,1,2))
    alpha_low = np.mean(low_data[:,:,2:8,:], axis=(0,1,2))

    alpha_high_sub = np.mean(high_data[:,:,2:8,:], axis=(1,2))
    alpha_low_sub = np.mean(low_data[:,:,2:8,:], axis=(1,2))

    diff = alpha_high_sub-alpha_low_sub
    
    t,p,H = mne.stats.permutation_t_test(diff)

    print(f'{times[np.where(p<0.05)]} alpha')

    alpha_sd_high = np.std(high_data[:,:,2:8,:], axis=(0,1,2))
    alpha_sd_low = np.std(low_data[:,:,2:8,:], axis=(0,1,2))

    beta_high = np.mean(high_data[crit>0,:,14:20,:], axis=(0,1,2))
    beta_low = np.mean(low_data[crit>0,:,14:20,:], axis=(0,1,2))

    beta_high_sub = np.mean(high_data[:,:,14:20,:], axis=(1,2))
    beta_low_sub = np.mean(low_data[:,:,14:20,:], axis=(1,2))

    diff = beta_high_sub-beta_low_sub

    t,p,H = mne.stats.permutation_t_test(diff)

    print(f'{times[np.where(p<0.05)]} beta')

    beta_sd_high = np.std(high_data[:,:,2:8,:], axis=(0,1,2))
    beta_sd_low = np.std(low_data[:,:,2:8,:], axis=(0,1,2))

    plt.plot(np.linspace(tmin, tmax, alpha_high.shape[0]), alpha_high, label='alpha 0.75')
    plt.plot(np.linspace(tmin, tmax, alpha_high.shape[0]), alpha_low, label='alpha 0.25')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.savefig(f'{savedir_figure4}\\{cond_a}_{cond_b}_alpha.svg', dpi=300)

    plt.plot(np.linspace(tmin, tmax, alpha_high.shape[0]), beta_high, label='beta high')
    plt.plot(np.linspace(tmin, tmax, alpha_high.shape[0]), beta_low, label='beta low')

    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Power')
    plt.savefig(f'{savedir_figure4}\\{cond_a}_{cond_b}_beta_zoom.svg', dpi=300)

    contrast_data = np.array([h.crop(tmin,tmax).data-l.crop(tmin,tmax).data for h,l in zip (evokeds_high, evokeds_low)])
    
    X = contrast_data.mean(axis=0)

    X_tfr = mne.time_frequency.AverageTFR(data=X, 
                                          times=evokeds_high[0].times,
                                          freqs=evokeds_high[0].freqs, 
                                          nave=39, 
                                          info=evokeds_high[0].info)
    X_tfr.crop(-0.5,0).plot(picks=['CP4'])

    high_gra = mne.grand_average(evokeds_high)
    low_gra = mne.grand_average(evokeds_low)

    topo = high_gra.plot_topo()
    topo = low_gra.plot_topo()

    tfr = high_gra.plot(picks=channel_names, combine='mean')[0]
    tfr = low_gra.plot(picks=channel_names, combine='mean')[0]

    diff = high_gra - low_gra

    diff.crop(-0.8,0.3).plot(picks=channel_names,
                                  combine='mean')[0]

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

    contrast_data = np.array([h.crop(tmin,tmax).data-l.crop(tmin,tmax).data for h,l in zip (evokeds_high, evokeds_low)])

    spec_channel_list = []

    for i, channel in enumerate(channel_names):
        spec_channel_list.append(evokeds_high[15].ch_names.index(channel))
    spec_channel_list

    mean_over_channels = np.mean(contrast_data[:, spec_channel_list, :, :], axis=1)

    threshold_tfce = dict(start=0, step=0.01)

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

    vmax = np.max(T_obs)
    vmin = np.min(T_obs)

    #cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
    
    # add time on the x axis 
    x = np.linspace(tmin, tmax, mean_over_channels.shape[2])
    y = np.arange(6, 35, 1)

    # Plot the original image
    fig = plt.imshow(T_obs, origin='lower',
                     extent=[x[0], x[-1], y[0], y[-1]],
                     aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar()

    # Plot the masked image on top with lower transparency
    fig = plt.imshow(masked_img, origin='lower', alpha=0.7, 
                     extent=[x[0], x[-1], y[0], y[-1]],
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

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X,
                                                           n_permutations=100,
                                                           adjacency=com_adjacency,
                                                           #threshold=threshold_tfce,
                                                           n_jobs=-1)
    
    print(f'The minimum p-value is {min(cluster_p_values)}')
    
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

    mean_over_channels = np.mean(contrast_data[:, spec_channel_list, :, :], axis=1)

    out = cluster_correlation.permutation_cluster_correlation_test(mean_over_channels, 
                                                             crit, threshold=0.05, 
                                                             test='pearson')

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

        # randomly sample from low power trials to match number of trials
        # in high power condition (equalize epoch counts mne not supported
        # for tfrepochs object (yet))
        # run 100 times and average over TFR average object

def permutate_trials(n_permutations=500, power_a=None, power_b=None):

    """ Permutate trials between two conditions and equalize trial counts"""

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
