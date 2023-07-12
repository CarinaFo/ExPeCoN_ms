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
from behav import figure1
from permutation_tests import cluster_correlation
import pyvistaqt  # for proper 3D plotting


# for plots in new windows
# %matplotlib qt

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif', 
                     'font.sans-serif': 'Arial'})

# datapaths
tfr_single_trial_power_dir = r"D:\expecon_ms\data\\eeg\sensor\tfr"

tfr_contrast_dir = "D:\\expecon_ms\\data\\eeg\\sensor\\tfr\\contrasts_corr"
savedir_figure4 = 'D:\\expecon_ms\\figs\\manuscript_figures\\Figure4'
dir_cleanepochs = "D:\\expecon_ms\\data\\eeg\\prepro_ica\\clean_epochs_iclabel"
behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021', '022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046',
          '047', '048', '049')


def calculate_power_per_trial(tmin=-1.5, tmax=1.5, laplace=0, induced=True,
                              zero_pad=0, save=1, reject_criteria=dict(eeg=200e-6),
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
    
    df_all, spectra = [], []

    for idx, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)

        # load cleaned epochs
        epochs = mne.read_epochs(f"{dir_cleanepochs}"
                                 f"/P{subj}_epochs_after_ic-label-epo.fif")

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

        if laplace:
            epochs = mne.preprocessing.compute_current_source_density(epochs)

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

        metadata = epochs.metadata

        # save metadata after dropping bad epochs
        df_all.append(metadata)

        if zero_pad:

            # zero pad the data on both sides to avoid leakage and edge artifacts
            data = epochs.get_data()

            data = zero_pad_data(data)

            # put back into epochs structure

            epochs = mne.EpochsArray(data, epochs.info, tmin=tmin * 2)

        mne.epochs.equalize_epoch_counts([epochs])

        # subtract evoked response for each condition
        if induced:
            epochs = epochs.subtract_evoked()
        
        # calculate prestimulus power per trial
        spectrum = epochs.compute_psd(fmin=2, fmax=40, tmin=-0.4, tmax=0, n_jobs=-1)
        spec_data = spectrum.get_data()

        np.save(f"D:/expecon_ms/data/eeg/sensor/psd/{subj}_psd.npy", spec_data)

        spectra.append(spectrum)

    df = pd.concat(df_all)

    df.to_csv(f"{behavpath}//prepro_behav_data_after_dropbads.csv")

    return spectra

def extract_band_power():

    df = [s.metadata for s in spectra]
    df = pd.concat(df)

    # average over frequency bands and log 10 transform
    theta_pow = [np.log10(np.mean(s.get_data()[:,:,:2], axis=2)) for s in spectra]
    alpha_pow = [np.log10(np.mean(s.get_data()[:,:,2:5], axis=2)) for s in spectra]
    beta_pow = [np.log10(np.mean(s.get_data()[:,:,6:-2], axis=2)) for s in spectra]
    gamma_pow = [np.log10(np.mean(s.get_data()[:,:,-2:], axis=2)) for s in spectra]

    # extract band power from the power spectrum in channel C4

    spectra[0].ch_names.index('C4')
    # extract power for channel C4 only
    theta_pow = [p[:,22] for p in theta_pow]
    alpha_pow = [p[:,22] for p in alpha_pow]
    beta_pow = [p[:,22] for p in beta_pow]
    gamma_pow = [p[:,22] for p in gamma_pow]

    # Flatten the list of tuples into one long list
    theta_pow = [item for sublist in theta_pow for item in sublist]
    alpha_pow = [item for sublist in alpha_pow for item in sublist]
    beta_pow = [item for sublist in beta_pow for item in sublist]
    gamma_pow = [item for sublist in gamma_pow for item in sublist]

    # add band power to dataframe
    df['theta_pow'] = theta_pow
    df['alpha_pow'] = alpha_pow
    df['beta_pow'] = beta_pow
    df['gamma_pow'] = gamma_pow

    # save dataframe
    df.to_csv(f"{behavpath}//behav_power_df.csv")

    beta_gr = df.groupby(['ID', 'cue'])['beta_pow'].mean()
    alpha_gr = df.groupby(['ID', 'cue'])['alpha_pow'].mean()
    theta_gr = df.groupby(['ID', 'cue'])['theta_pow'].mean()
    gamma_gr = df.groupby(['ID', 'cue'])['gamma_pow'].mean()

    power_gr = {'theta': theta_gr, 'alpha': alpha_gr, 'beta': beta_gr, 'gamma': gamma_gr}

    for keys, values in power_gr.items():
        # Extract values for low and high expectation conditions
        low_expectation = values.loc[:, 0.25]
        high_expectation = values.loc[:, 0.75]

        # Perform paired t-test
        t_statistic, p_value = scipy.stats.wilcoxon(low_expectation, high_expectation)

        # Print the t-test results
        print("Paired t-test results:")
        print("t-statistic:", t_statistic)
        print("p-value:", p_value)

        plt.boxplot([low_expectation, high_expectation], labels=['Low Expectation', 'High Expectation'])

        # Set plot title and labels
        plt.title(f'{keys} Power Comparison')
        plt.xlabel('Condition')
        plt.ylabel(f'{keys} Power')

        # Display the plot
        plt.show()


def visualize_contrasts(cond_a='high', cond_b='low',
                       tmin=-0.6, tmax=0.6,
                       channel_names=['C4']):

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

    power_a_all, power_b_all = [], []

    # load single trial power
    for idx, subj in enumerate(IDlist):
        
        # load power
        power_a = mne.time_frequency.read_tfrs(f"{tfr_single_trial_power_dir}//{subj}_{cond_a}_power_per_trial_induced-tfr.h5")

        power_b = mne.time_frequency.read_tfrs(f"{tfr_single_trial_power_dir}//{subj}_{cond_b}_power_per_trial_induced-tfr.h5")

        # average over epochs per condition
        evoked_a = power_a[0].average()
        evoked_b = power_b[0].average()

        power_a_all.append(evoked_a)
        power_b_all.append(evoked_b)
    
    # average over participants
    gra_a = mne.grand_average(power_a_all)
    gra_b = mne.grand_average(power_b_all)

    diff = gra_a - gra_b

    # load signal detection dataframe
    out = figure1.prepare_for_plotting()
        
    sdt = out[0][0]
    
    # contrast data
    contrast_data = np.array([h.data-l.data for h, l in zip(power_a_all, power_b_all)])
    times = power_a_all[0].times

    high_data = np.array([h.pick_channels(['C4']).data for h in power_a_all])
    low_data = np.array([l.pick_channels(['C4']).data for l in power_b_all])

    # average over alpha band and participants
    alpha_high = np.mean(high_data[:,:,1:7,:], axis=(0,1,2))
    alpha_low = np.mean(low_data[:,:,1:7,:], axis=(0,1,2))

    alpha_high_sub = np.mean(high_data[:,:,1:7,:], axis=(1 ,2))
    alpha_low_sub = np.mean(low_data[:,:,1:7,:], axis=(1, 2))

    diff = alpha_high_sub-alpha_low_sub
    
    t,p,H = mne.stats.permutation_t_test(diff)

    print(f'{power_a_all[0].times[np.where(p<0.05)]} alpha')

    alpha_sd_high = np.std(high_data[:,:,1:7,:], axis=(0,1,2))
    alpha_sd_low = np.std(low_data[:,:,1:7,:], axis=(0,1,2))

    beta_high = np.mean(high_data[:,:,9:,:], axis=(0,1,2))
    beta_low = np.mean(low_data[:,:,9:,:], axis=(0,1,2))

    beta_high_sub = np.mean(high_data[:,:,9:,:], axis=(1,2))
    beta_low_sub = np.mean(low_data[:,:,9:,:], axis=(1,2))

    diff = beta_high_sub-beta_low_sub

    t,p,H = mne.stats.permutation_t_test(diff)

    print(f'{power_a_all[0].times[np.where(p<0.05)]} beta')

    beta_sd_high = np.std(high_data[:,:,12:16,:], axis=(0,1,2))
    beta_sd_low = np.std(low_data[:,:,12:16,:], axis=(0,1,2))

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

    topo = gra_a.plot_topo()
    topo = gra_b.plot_topo()

    tfr = gra_a.plot(picks=channel_names, combine='mean')[0]
    tfr = low_gra.plot(picks=channel_names, combine='mean')[0]

    topo = diff.plot_topo()
    tfr = diff.copy().crop(-1,0).plot(picks=channel_names,
                                  combine='mean')[0]

    topo.savefig(f'{savedir_figure4}//{cond}_topo.svg')
    
    tfr.savefig(f'{savedir_figure4}//{cond}_baseline_tfr.svg')


def run_2D_cluster_perm_test(channel_names=['C4'],
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

    contrast_data = np.array([h.crop(tmin,tmax).data-l.crop(tmin,tmax).data for h,l in zip (power_a_all, power_b_all)])

    spec_channel_list = []

    for i, channel in enumerate(channel_names):
        spec_channel_list.append(power_a_all[18].ch_names.index(channel))
    spec_channel_list

    mean_over_channels = np.mean(contrast_data[:, spec_channel_list, :, :], axis=1)

    threshold_tfce = dict(start=0, step=0.01)

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                            mean_over_channels,
                            n_jobs=-1,
                            n_permutations=n_perm,
                            tail=0)
    
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
    y = np.arange(6, 36, 1)

    # Plot the original image
    fig = plt.imshow(T_obs, origin='lower',
                     extent=[x[0], x[-1], y[0], y[-1]],
                     aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar()

    # Add x and y labels
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')

    # save figure
    os.chdir(savedir_figure4)

    fig.figure.savefig(f"cluster_perm_{cond}_{str(tmin)}_{str(tmax)}.svg")

    # Show the plot
    plt.show()

        # Plot the masked image on top with lower transparency
    fig = plt.imshow(masked_img, origin='lower', alpha=0.7, 
                     extent=[x[0], x[-1], y[0], y[-1]],
                     aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    

# Works but now nice plotting implemented yet, potentially also overkill
# to run 3D cluster permutation test

def run_3D_cluster_perm_test(contrast_data=None):

    X = np.array([d.data for d in contrast_data])

    # run 3D cluster permutation test
    # definde adjaceny matrix for cluster permutation test
   
    ch_adjacency = mne.channels.find_ch_adjacency(evokeds_high[0].info,
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

    # low - high exp. condition

    crit_diff = np.array(sdt.criterion[sdt.cue == 0.75]) - np.array(sdt.criterion[sdt.cue == 0.25])

    d_diff = np.array(sdt.dprime[sdt.cue == 0.75]) - np.array(sdt.dprime[sdt.cue == 0.25])

    # load random effects from glmmer model
    re = pd.read_csv("D:\expecon_ms\data\\behav\mixed_models\\brms\\brms_betaweights.csv")
    
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
                                                             d_diff, threshold=0.05, 
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



def contrast_conditions(cond='highlow', cond_a='high',
                        cond_b='low', induced_power=0,
                        tmin=-1, tmax=1):
    
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

    data = pd.read_csv(f"{behavpath}//prepro_behav_data_after_dropbads.csv")

    for counter, subj in enumerate(IDlist):
       
        # save induced power
        power_induced_all = []

        power = mne.time_frequency.read_tfrs(f"{tfr_single_trial_power_dir}//{subj}_power_per_trial_induced-tfr.h5")[0]
        
        # add metadata
        subj_data = data[data.ID == counter+7]

        power.metadata = subj_data

        power = power.copy().crop(tmin, tmax)

        if induced_power:

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
            power_a = power[(power.metadata.cue == 0.75)]
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
    with open(f'{tfr_contrast_dir}//{cond_a}_all_subs_induced.pickle', 'wb') as file:
        pickle.dump(evokeds_high, file)

    with open(f'{tfr_contrast_dir}//{cond_b}_all_subs_induced.pickle', 'wb') as file:
        pickle.dump(evokeds_low, file)

    return evokeds_high, evokeds_low