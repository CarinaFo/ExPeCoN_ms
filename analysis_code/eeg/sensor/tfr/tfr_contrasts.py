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

IDlist = ['007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021', '022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046',
          '047', '048', '049']


def compute_tfr(tmin=-0.5, tmax=0, fmax=35, fmin=7, laplace=0,
                induced=False, mirror_data=1, psd=0):

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
        mirror_data : boolean, info: mirror the data on both sides to avoid edge artifacts
        psd : boolean, info: calculate power spectral density
        Returns
        -------
        None
        '''

    # Define frequencies and cycles for multitaper method
    freqs = np.arange(fmin, fmax, 1)
    cycles = freqs/4.0
    
    # store behavioral data
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

        # crop epopchs in desired time window
        epochs.crop(tmin=tmin, tmax=tmax)

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
 
            tfr_path = Path('D:/expecon_ms/data/eeg/sensor/tfr')

            epochs_a = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.prevsayyes == 1))]
            epochs_b = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.prevsayyes == 0))]

            mne.epochs.equalize_epoch_counts([epochs_a, epochs_b]) # minimize timing differences

            if os.path.exists(f'{tfr_path}{Path("/")}{subj}_high_no_mirror-tfr.h5'):
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
                
                tfr_a.save(f'{tfr_path}{Path("/")}{subj}_high_yes_mirror-tfr.h5', overwrite=True)
                tfr_b.save(f'{tfr_path}{Path("/")}{subj}_high_no_mirror-tfr.h5', overwrite=True)

    return 'Done with tfr computation', freqs


def visualize_contrasts(cond_a='high_yes', cond_b='high_no'):

    '''plot the grand average of the difference between conditions
    after loading the contrast data using pickle
    Parameters
    ----------
    cond_a : str
        which condition tfr to load 
    cond_b : str
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

    # now contrast conditions: high vs low
    #tfr_a_all = np.array([a[a.metadata.cue == 0.75].get_data() for a in tfr_all])
    #tfr_b_all = np.array([b[b.metadata.cue == 0.25].get_data() for b in tfr_all])

    #tfr_a, tfr_b = equalize_epochs(tfr_a_all, tfr_b_all)

    return diff, gra_a, gra_b, tfr_a_all, tfr_b_all

def plot_figure4(channel_names=['C4'], fmin=7, fmax=35):

    '''plot figure 4: tfr contrasts (first row for different time windows)
    and significant clusters (second row)
    Parameters
    channel_names : list of char 
        channels to analyze
    fmin : int
        minimum frequency to plot
    fmax : int
        maximum frequency to plot
    ----------
    channel_names : list of char (channel names)
    Returns
    -------
    None
    '''

    # Create a 2x3 grid of plots (2 rows, 3 columns)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(16, 10))

    # which time windows to plot
    time_windows = [(-1, -0.4), (-0.4, 0.5)]
    # for mirrored data
    #time_windows = [(-1, 0)]

    # which axes to plot the time windows
    axes_first_row = [(0,0), (0,1)]
    axes_second_row = [(1,0), (1,1)]

    # now populate first row with tfr contrasts
    for t, a in zip(time_windows, axes_first_row):

        tfr_fig = diff.copy().crop(tmin=t[0], tmax=t[1]).plot(picks=channel_names[0], combine='mean', vmin=-1, 
                                                    vmax =1, cmap = 'viridis', axes=axs[a[0], a[1]], show=False)[0]

    # now plot cluster permutation output in the second row
    # pick channel
    ch_index = tfr_a.ch_names.index(channel_names[0])

    for t, a in zip(time_windows, axes_second_row):

        # contrast data
        X = np.array([h.copy().crop(tmin=t[0], tmax=t[1]).data - l.copy().crop(tmin=t[0], tmax=t[1]).data for h, l in zip(tfr_a_all, tfr_b_all)])

        # pick channel
        X = X[:, ch_index, :, :]

        print(X.shape) # should be participants x frequencies x timepoints

        # run cluster test over time and frequencies (no need to define adjacency)
        T_obs, clusters, cluster_p, H0 = mne.stats.permutation_cluster_1samp_test(
                                X,
                                n_jobs=-1,
                                n_permutations=10000,
                                tail=0)

        if len(cluster_p) > 0:

            print(f'The minimum p-value is {min(cluster_p)}')

            good_cluster_inds = np.where(cluster_p < 0.05)
            
            if len(good_cluster_inds[0]) > 0:

                # Find the index of the overall minimum value
                min_index = np.unravel_index(np.argmin(T_obs), T_obs.shape)

                freqs = np.arange(fmin, fmax, 1)
                times = tfr_a_all[0].copy().crop(t[0], t[1]).times

                #print(f'Frequencies: {str(freqs[np.unique(clusters[good_cluster_inds[0]][0])])}')
                #print(f'Timepoints: {str(times[np.unique(clusters[good_cluster_inds[0]][1])])}')

        # run function to plot significant cluster in time and frequency space
        plot_cluster_test_output(tobs = T_obs, cluster_p_values = cluster_p, clusters=clusters, fmin=7, fmax=35,
                                data_cond=tfr_a_all, tmin=t[0], tmax=t[1], ax0=a[0], ax1=a[1])

    # finally save figure
    plt.savefig(f'{savedir_figure4}{Path("/")}fig4_{cond_a}_{cond_b}_tfr_C4.svg', dpi=300, format='svg')
    plt.savefig(f'{savedir_figure4}{Path("/")}fig4_{cond_a}_{cond_b}_tfr_C4.png', dpi=300, format='png')

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
    freqs[np.unique(clusters[162][1])]

    # significant timepoints
    times = tfr_a_all[0].copy().crop(tmin, tmax).times

    times[np.unique(clusters[162][0])]

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
        cmap=plt.cm.bwr,
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

    fmin=min(np.unique(clusters[0][0]))
    fmax=max(np.unique(clusters[0][0]))
    tmin=min(np.unique(clusters[0][1]))
    tmax=max(np.unique(clusters[0][1]))

    #cluster before cue:
    X1 = X[:,fmin:fmax, tmin:tmax]
    X1 = X[:, min_index[0], min_index[1]]

    X1 = X[:,10:12,50:95]
   
    X1 = np.mean(X1, axis=(1,2))

    X1 = X1*10**10

    # load signal detection dataframe
    out = figure1.prepare_for_plotting()
        
    sdt = out[0][0]

    crit_diff = np.array(sdt.criterion[sdt.cue == 0.75]) - np.array(sdt.criterion[sdt.cue == 0.25])

    d_diff = np.array(sdt.dprime[sdt.cue == 0.75]) - np.array(sdt.dprime[sdt.cue == 0.25])

    # load random effects from glmmer model
    #re = pd.read_csv(f'{Path("D:/expecon_ms/data/behav/mixed_models/brms/brms_betaweights.csv")}')
    re = pd.read_csv(f'{Path("D:/expecon_ms/data/behav/mixed_models/brms/regweights_6trials.csv")}')

    np.corrcoef(re.criterion, X1)

    scipy.stats.pearsonr(re.criterion, X1)

    sns.regplot(re.criterion, X1)
    
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

    df_new = []

    # Assign a sequential count for each row within each 'blocks' and 'subblock' group
    for idx, id in enumerate(IDlist):
        df_sub = df[df.ID == idx+7]
        df_sub['trial_count'] = df_sub.groupby(['block', 'subblock']).cumcount()
        df_new.append(df_sub)

    df_new = pd.concat(df_new)

    # add trial count to metadata
    spectra_new = []

    for i, spectrum in enumerate(spectra):
        df_sub = df_new[df_new['ID'] == i+7]
        spectrum.metadata = df_sub
        spectra_new.append(spectrum)

    # save freqs
    freqs = spectra[0].freqs

    # contrast conditions
    spectra_high = [spectrum[((spectrum.metadata.cue == 0.75) & (spectrum.metadata.trial_count < 6))] for spectrum in spectra_new]
    spectra_low = [spectrum[((spectrum.metadata.cue == 0.25) & (spectrum.metadata.trial_count < 6))] for spectrum in spectra_new]

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
    alpha_pow = [np.log10(np.mean(s.get_data()[:,:,0:4], axis=2)) for s in spectra]
    beta_pow = [np.log10(s.get_data()[:,:,4:]) for s in spectra]
  
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

    power_gr = {'beta': beta_gr, 'alpha': alpha_gr}

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

        # plt.savefig('D:\\expecon_ms\\figs\\manuscript_figures\\Figure6_PSD\\boxplot_beta_diff.svg', dpi=300, format='svg')
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
    
    low_expectation = beta_gr.loc[:, 0.25]
    high_expectation = beta_gr.loc[:, 0.75]
    diff = high_expectation - low_expectation

    np.corrcoef(diff, re['cue_prev'])

    scipy.stats.pearsonr(diff, re['cue'])


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

