#################################################################################################
# investigate pre-stimulus power
##################################################################################################

# import packages
import os
import pickle
import random
import sys
from pathlib import Path

# add path to sys.path.append() if package isn't found
sys.path.append('D:\\expecon_ms\\analysis_code')

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyvistaqt  # for proper 3D plotting
import scipy
import seaborn as sns
from behav import figure1
from permutation_tests import cluster_correlation

# for plots in new windows
# %matplotlib qt

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif', 
                     'font.sans-serif': 'Arial'})

# colormaps
original_cmap = plt.cm.Blues
cmap_data = original_cmap(np.linspace(0, 1, 256))

# Reverse the colormap data
reversed_cmap_data = cmap_data[::-1]

# Create a new colormap from the reversed data
reversed_cmap = mcolors.ListedColormap(reversed_cmap_data)
    
# datapaths
savedir_figure4 = Path('D:/expecon_ms/figs/manuscript_figures/Figure4')
dir_cleanepochs = Path('D:/expecon_ms/data/eeg/prepro_ica/clean_epochs_corr')
behavpath = Path('D:/expecon_ms/data/behav/behav_df')

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021', '022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046',
          '047', '048', '049')


def compute_tfr(tmin=-1, tmax=0.5, fmax=35, fmin=7, laplace=0,
                induced=False, zero_pad=0, psd=0):

    '''calculate prestimulus time-frequency representations per trial
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
        zero_pad : boolean, info: zero pad the data on both sides to avoid leakage and edge artifacts
        psd : boolean, info: calculate power spectral density
        Returns
        -------
        None
        '''

    # Define frequencies and cycles for multitaper method
    freqs = np.arange(fmin, fmax, 1)
    cycles = freqs/4.0
    
    # store behavioral data
    df_all = []

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

        # apply CSD to the data (less point spread)
        if laplace:
            epochs = mne.preprocessing.compute_current_source_density(epochs)

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

        # drop epochs with abnormal strong signal (> 200 mikrovolts)
        epochs.drop_bad(reject=dict(eeg=200e-6))

        # avoid leakage and edge artifacts by zero padding the data
        if zero_pad:

            # zero pad the data on both sides to avoid leakage and edge artifacts
            data = epochs.get_data()

            data = zero_pad_data(data)

            # put back into epochs structure

            epochs = mne.EpochsArray(data, epochs.info, tmin=tmin * 2)

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
            spectrum_path = Path('D:/expecon_ms/data/eeg/sensor/psd/')

            # save the spectra to disk
            np.save(f'{spectrum_path}{Path("/")}{subj}_psd.npy', spec_data)
            
        else:
            
            # calculate prestimulus power per trial
            epochs.crop(tmin=tmin, tmax=tmax)

            # first create conditions and equalize trial numbers
            epochs_high = epochs[((epochs.metadata.cue == 0.75))]
            epochs_low = epochs[((epochs.metadata.cue == 0.25))]

            mne.epochs.equalize_epoch_counts([epochs_high, epochs_low])

            hit = epochs[(epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1)]
            miss = epochs[(epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 0)]

            mne.epochs.equalize_epoch_counts([hit, miss])

            # Assign a sequential count for each row within each 'blocks' and 'subblock' group
            epochs.metadata['trial_count'] = epochs.metadata.groupby(['block', 'subblock']).cumcount()

            df_all.append(epochs.metadata)

            # Filter the first 6 trials from each 'blocks' and 'subblock' group
            first_trials = epochs.metadata[epochs.metadata['trial_count'] < 6]

            epochs_high = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.trial_count < 6))]
            epochs_low = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.trial_count < 6))]

            mne.epochs.equalize_epoch_counts([epochs_high, epochs_low])

            tfr_path = Path('D:/expecon_ms/data/eeg/sensor/tfr')

            if os.path.exists(f'{tfr_path}{Path("/")}{subj}_high-tfr.h5'):
                print('TFR already exists')
            else:
                tfr_high, _ = mne.time_frequency.tfr_multitaper(epochs_high, 
                                                                freqs=freqs,
                                                                n_cycles=cycles,
                                                                n_jobs=-1)
                    
                tfr_low, _ = mne.time_frequency.tfr_multitaper(epochs_low, 
                                                               freqs=freqs,
                                                               n_cycles=cycles,
                                                               n_jobs=-1)

                tfr_high.save(f'{tfr_path}{Path("/")}{subj}_high-tfr.h5')
                tfr_low.save(f'{tfr_path}{Path("/")}{subj}_low-tfr.h5')
            
            if os.path.exists(f'{tfr_path}{Path("/")}{subj}_hit-tfr.h5'):
                print('TFR already exists')
            else:
                tfr_hit,_ = mne.time_frequency.tfr_multitaper(hit, freqs=freqs,
                                                              n_cycles=cycles,
                                                              n_jobs=-1)
                
                tfr_miss,_ = mne.time_frequency.tfr_multitaper(miss, 
                                                               freqs=freqs,
                                                               n_cycles=cycles,
                                                               n_jobs=-1)
                
                tfr_hit.save(f'{tfr_path}{Path("/")}{subj}_hit-tfr.h5')
                tfr_miss.save(f'{tfr_path}{Path("/")}{subj}_miss-tfr.h5')
            
            if os.path.exists(f'{tfr_path}{Path("/")}{subj}_high_6trialsaftercue-tfr.h5'):
                print('TFR already exists')
            else:
                tfr_high, _ = mne.time_frequency.tfr_multitaper(epochs_high, 
                                                                   freqs=freqs,
                                                                   n_cycles=cycles,
                                                                   n_jobs=-1)
                
                tfr_low, _ = mne.time_frequency.tfr_multitaper(epochs_low,
                                                                  freqs=freqs,
                                                                  n_cycles=cycles,
                                                                  n_jobs=-1)
                
                tfr_high.save(f'{tfr_path}{Path("/")}{subj}_high_6trialsaftercue-tfr.h5')
                tfr_low.save(f'{tfr_path}{Path("/")}{subj}_low_6trialsaftercue-tfr.h5')
                
    df = pd.concat(df_all)

    df.to_csv(f'{Path("D:/expecon_ms/data/behav/behav_df")}{Path("/")}df_after_tfr_analysis.csv')

    return 'Done with tfr computation', freqs, df


def visualize_contrasts(cond_a='high_6trialsaftercue', cond_b='low_6trialsaftercue':

    '''plot the grand average of the difference between conditions
    after loading the contrast data using pickle
    Parameters
    ----------
    cond : str
        which condition tfr to load 
    Returns
    -------
    diff: mne.time_frequency.AverageTFR
        difference between conditions
    gra_a: mne.time_frequency.AverageTFR
        grand average of condition a
    gra_b: mne.time_frequency.AverageTFR
        grand average of condition b
    tfr_a_all: list
        list of tfr objects for condition a
    tfr_b_all: list
        list of tfr objects for condition b
    '''

    tfr_a_all, tfr_b_all = [], []

    # load single trial power
    for idx, subj in enumerate(IDlist):
        
        # load tfr data
        tfr_path = Path('D:/expecon_ms/data/eeg/sensor/tfr')

        tfr_a = mne.time_frequency.read_tfrs(f'{tfr_path}{Path("/")}{subj}_{cond_a}-tfr.h5', condition=0)

        tfr_b = mne.time_frequency.read_tfrs(f'{tfr_path}{Path("/")}{subj}_{cond_b}-tfr.h5', condition=0)

        tfr_a_all.append(tfr_a)
        tfr_b_all.append(tfr_b)
    
    # average over participants
    gra_a = mne.grand_average(tfr_a_all)
    gra_b = mne.grand_average(tfr_b_all)

    # difference between conditions (2nd level)
    diff = gra_a - gra_b
    diff.data = diff.data*10**11

    return diff, gra_a, gra_b, tfr_a_all, tfr_b_all

def plot_figure4(channel_names=['C4']):

    '''plot figure 4
    Parameters
    ----------
    channel_names : list of char (channel names)
    Returns
    -------
    None
    '''

    # Create a 2x3 grid of plots (2 rows, 3 columns)
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 9))

    # which time windows to plot
    time_windows = [(-1, -0.5), (-0.4, 0.1), (-0.1, 0.4)]

    # which axes to plot the time windows
    axes_first_row = [(0,0), (0,1), (0,2)]
    axes_second_row = [(1,0), (1,1), (1,2)]

    # now populate first row with tfr contrasts
    for t, a in zip(time_windows, axes_first_row):

        tfr_fig = diff.copy().crop(tmin=t[0], tmax=t[1]).plot(picks=channel_names[0], combine='mean',
                                                    cmap=reversed_cmap, axes=axs[a[0], a[1]], show=False)[0]

    # now plot cluster permutation output in the second row
    # pick channel
    ch_index = tfr_a.ch_names.index(channel_names[0])

    for t, a in zip(times, axes_second_row):

        # contrast data
        X = np.array([h.copy().crop(tmin=t[0],tmax=t[1]).data - l.copy().crop(tmin=t[0],tmax=t[1]).data for h, l in zip(tfr_a_all, tfr_b_all)])

        # pick channel
        X = X[:, ch_index, :, :]

        print(X.shape) # should be participants x frequencies x timepoints

        # run cluster test over time and frequencies (no need to define adjacency)
        T_obs, clusters, cluster_p, H0 = mne.stats.permutation_cluster_1samp_test(
                                X,
                                n_jobs=-1,
                                n_permutations=10000,
                                tail=0)
        # run function to plot significant cluster in time and frequency space
        plot_cluster_test_output(tobs = T_obs, cluster_p_values = cluster_p, clusters=clusters, fmin=7, fmax=35,
                                data_cond=tfr_a_all, tmin=t[0], tmax=t[1], ax0=a[0], ax1=a[1])

    # finally save figure
    plt.savefig(f'{savedir_figure4}{Path("/")}fig4_tfr.svg', dpi=300, format='svg')

 def run_3D_cluster_test():

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
                                                           #threshold=threshold_tfce,
                                                           n_jobs=-1)
    
    print(f'The minimum p-value is {min(cluster_p_values)}')

    good_cluster_inds = np.where(cluster_p_values < 0.05)

    # Find the index of the overall minimum value
    min_index = np.unravel_index(np.argmin(T_obs), T_obs.shape)

    print("Index of the overall minimum value:", min_index)

    freqs = np.arange(fmin, fmax, 1)

    # significant frequencies
    freqs[np.unique(clusters[0][0])]

    # significant timepoints
    times = tfr_a_all[0].copy().crop(tmin, tmax).times

    times[np.unique(clusters[0][1])]

def plot_cluster_test_output(tobs=None, cluster_p_values=None, clusters=None, fmin=7, fmax=35,
                             data_cond=None, tmin=0, tmax=0.5, ax0=0, ax1=0):
    
    freqs = np.arange(fmin, fmax, 1)

    times = 1e3 * data_cond[0].copy().crop(tmin,tmax).times

    T_obs_plot = np.nan * np.ones_like(T_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= 0.05:
            T_obs_plot[c] = T_obs[c]

    vmax = np.max(np.abs(T_obs))
    vmin = -vmax

    fig1 = axs[ax0, ax1].imshow(
        T_obs,
        cmap=plt.cm.gray,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    fig2 = axs[ax0, ax1].imshow(
        T_obs_plot,
        cmap=plt.cm.RdBu_r,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    # Add colorbar for each subplot
    fig.colorbar(fig2, ax = axs[ax0, ax1])

    axs[ax0, ax1].set_xlabel("Time (ms)")
    axs[ax0, ax1].set_ylabel("Frequency (Hz)")

def correlate_cluster_with_behav():

    # correlate cluster with criterion change

    fmin=min(np.unique(clusters[1][0]))
    fmax=max(np.unique(clusters[1][0]))
    tmin=min(np.unique(clusters[1][1]))
    tmax=max(np.unique(clusters[1][1]))

    #cluster before cue:
    X1 = X[:,fmin:fmax, tmin:tmax]
    X1 = X[:, min_index[0], min_index[1]]
    
    X1 = np.mean(X1, axis=(1,2))

    X1 = X1*10**11

    # load signal detection dataframe
    out = figure1.prepare_for_plotting()
        
    sdt = out[0][0]

    crit_diff = np.array(sdt.criterion[sdt.cue == 0.75]) - np.array(sdt.criterion[sdt.cue == 0.25])

    d_diff = np.array(sdt.dprime[sdt.cue == 0.75]) - np.array(sdt.dprime[sdt.cue == 0.25])

    # load random effects from glmmer model
    re = pd.read_csv("D:\expecon_ms\data\\behav\mixed_models\\brms\\brms_betaweights.csv")

    np.corrcoef(re.cue, X1)

    scipy.stats.pearsonr(re.cue, X1)

    sns.regplot(re.cue, X1)
    
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


def run_2D_cluster_perm_test(channel_names=['C3', 'CP1','Pz','CP2','C4','C1','CP3','P1','P2','CPz','CP4','C2','FC4'],
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

    prevyes_cluster = ['Fz', 'F3', 'FC1', 'CP1', 'Pz', 'P3', 'C4', 'FC2', 'F4', 'AF3', 'AFz', 'F1', 'FC3', 'C1','P1', 'P2', 'FC4', 'AF4', 'F2']
    highlow_cluster1 = ['Fz','FC1','C3','Pz','P7','O1','Oz','P4','P8', 'CP2','Cz','C4','FC2','AF3','AFz','F1','C1','CP3','P1','PO7','PO3','POz','PO4','PO8','P6','P2','CP4','C2','AF4']

    contrast_data = np.array([h.copy().crop(tmin,tmax).data-l.copy().crop(tmin,tmax).data
                               for h,l in zip(high, low)])

    spec_channel_list = []

    for i, channel in enumerate(channel_names):
        spec_channel_list.append(tfr_high_all[15].ch_names.index(channel))
    spec_channel_list

    mean_over_channels = np.mean(contrast_data[:, spec_channel_list, :, :], axis=(1))

    threshold_tfce = dict(start=0, step=0.01)

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                            mean_over_channels,
                            n_jobs=-1,
                            #threshold=threshold_tfce,
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
    x = np.linspace(tmin, tmax, X.shape[2])
    y = np.arange(7, 35, 1)

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

    X = np.array([d.data for d in X])

    # run 3D cluster permutation test
    # definde adjaceny matrix for cluster permutation test
   
    ch_adjacency = mne.channels.find_ch_adjacency(tfr_high_all[0].info,
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
                                                             re.prev_choice, threshold=0.1, 
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


def extract_band_power():

    df = [s.metadata for s in spectra]
    df = pd.concat(df)

    # save freqs
    freqs = spectra[0].freqs

    # contrast conditions
    spectra_high = [spectrum[spectrum.metadata.cue == 0.75] for spectrum in spectra]
    spectra_low = [spectrum[spectrum.metadata.cue == 0.25] for spectrum in spectra]

    # prepare for fooof (channel = C4 = index 24)
    ch_index = spectra[0].ch_names.index('C4')

    spectra_high_c4 = np.array([np.mean(np.log10(psd.get_data()[:, ch_index, :]), axis=0) for psd in spectra_high])
    spectra_low_c4 = np.array([np.mean(np.log10(psd.get_data()[:, ch_index, :]), axis=0) for psd in spectra_low])
    
    spectra_high_allchannel = np.array([np.mean(np.log10(psd.get_data()), axis=0) for psd in spectra_high])
    spectra_low_allchannel = np.array([np.mean(np.log10(psd.get_data()), axis=0) for psd in spectra_low])
    
    # t-test over difference
    diff_channels = spectra_high_allchannel - spectra_low_allchannel

    diff = spectra_high_c4 - spectra_low_c4

    # permutation test
    T,p,H = mne.stats.permutation_t_test(diff, n_permutations=10000, tail=0, n_jobs=-1)
    print(np.where(p < 0.05))
    freqs[np.where(p < 0.05)]

    # 1D cluster test
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(diff, 
                                                                                     n_permutations=10000, 
                                                                                     tail=0, 
                                                                                     n_jobs=-1)
    
    # 2D cluster test over frequencies and channels
    ch_adjacency,_ = mne.channels.find_ch_adjacency(spectra[0].info,
                                                  ch_type='eeg')
    
    # channels should be the last dimension for cluster test
    diff_channels = np.transpose(diff_channels, [0, 2, 1])

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(diff_channels, 
                                                                                     n_permutations=10000, 
                                                                                     tail=0,
                                                                                     adjacency=ch_adjacency,
                                                                                     n_jobs=-1)

    min(cluster_p_values)

    # grand average over participants
    gra_high = np.mean(spectra_high_c4, axis=0)
    gra_low = np.mean(spectra_low_c4, axis=0)

    # plot an example PSD
    plt.plot(freqs, gra_high, label='0.75')
    plt.plot(freqs, gra_low, label='0.25')
    plt.legend()
    plt.savefig('D:\\expecon_ms\\figs\\manuscript_figures\\Figure6_PSD\\high_low_psd.svg',
                dpi=300)
    
    # average over frequency bands and log 10 transform
    alpha_pow = [np.log10(np.mean(s.get_data()[:,:,1:4], axis=2)) for s in spectra]
    beta_pow = [np.log10(s.get_data()[:,:,14]) for s in spectra]
  
    # extract band power from the power spectrum in channel C4
    alpha_pow = [p[:,ch_index] for p in alpha_pow]
    beta_pow = [p[:,ch_index] for p in beta_pow]

    # Flatten the list of tuples into one long list
    alpha_pow = [item for sublist in alpha_pow for item in sublist]
    beta_pow = [item for sublist in beta_pow for item in sublist]

    # add band power to dataframe
    df['alpha_pow'] = alpha_pow
    df['beta_pow'] = beta_pow

    # save dataframe
    df.to_csv(f"{behavpath}//behav_power_df.csv")

    beta_gr = df.groupby(['ID', 'cue'])['beta_pow'].mean()
    alpha_gr = df.groupby(['ID', 'cue'])['alpha_pow'].mean()

    power_gr = {'beta': beta_gr}

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

        plt.savefig('D:\\expecon_ms\\figs\\manuscript_figures\\Figure6_PSD\\boxplot_beta_diff.svg', dpi=300, format='svg')
        # Display the plot
        plt.show()

        # low - high exp. condition

    # load signal detection dataframe
    out = figure1.prepare_for_plotting()
        
    sdt = out[0][0]

    crit_diff = np.array(sdt.criterion[sdt.cue == 0.75]) - np.array(sdt.criterion[sdt.cue == 0.25])

    d_diff = np.array(sdt.dprime[sdt.cue == 0.75]) - np.array(sdt.dprime[sdt.cue == 0.25])

    # load random effects from glmmer model
    re = pd.read_csv("D:\expecon_ms\data\\behav\mixed_models\\brms\\brms_betaweights.csv")
    
    diff = high_expectation - low_expectation

    np.corrcoef(diff, re['cue_prev'])

    scipy.stats.pearsonr(diff, re['cue'])
