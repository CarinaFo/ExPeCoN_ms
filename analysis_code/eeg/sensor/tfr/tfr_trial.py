#################################################################################################
# investigate pre-stimulus power per trial and frequency band
##################################################################################################

# import packages
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import seaborn as sns
# add path to sys.path.append() if package isn't found
sys.path.append(f'{Path("D:/expecon_ms/analysis_code")}')
from behav import figure1

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# set directory paths
behav_dir = Path('D:/expecon_ms/data/behav/behav_df/')
tfr_dir = Path('D:/expecon_ms/data/eeg/sensor/tfr')
savedir_fig5 = Path('D:/expecon_ms/figs/manuscript_figures/figure5')

IDlist = ['007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021', '022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046',
          '047', '048', '049']

# frequencies from tfr calculation
freq_list = np.arange(7, 35, 1)

# define frequency bands we are interested in
freq_bands = {'alpha': (7, 13), 'beta': (15, 25)}


def save_band_power_per_trial(time_intervals={'900to700': (-0.9, -0.7), '300to100': (-0.3, -0.1),
                                              '150to0': (-0.15, 0)}, channel_name=['C4']):

    """This function saves the power per trial per frequency band in a csv file. 
    The power is calculated for the time interval specified and averaged over 
    the specified channel.
    The power is calculated for the following frequency bands:
        - alpha (7-13 Hz)
        - beta (15-25 Hz)
    Return:
        - csv file with power per trial per frequency band
    """

    # save single subject dataframes in a list
    brain_behav = []

    for subj in IDlist:

        # load single trial power
        power = mne.time_frequency.read_tfrs(f'{tfr_dir}{Path("/")}{subj}_epochs-tfr.h5')[0]
        
        # get behavioral data
        behav_data = power.metadata

        # loop over the time intervals
        for keys, time in time_intervals.items():

            power_crop = power.copy().crop(tmin=time[0], tmax=time[1]).pick_channels(channel_name)

            # now we average over time and channels
            power_crop.data = np.mean(power_crop.data, axis=(1, 3))

            # now we average over the frequency band and time interval 
            # and add the column to the 
            # behavioral data frame
            behav_data[f'alpha_{keys}'] = np.mean(power_crop.data[:, 0:7], 
                                                  axis=1)  # 7-13 Hz
            behav_data[f'beta_{keys}'] = np.mean(power_crop.data[:, 8:19], 
                                                 axis=1) # 15-25 Hz

        # save the data in a list
        brain_behav.append(behav_data)

    # concatenate the list of dataframes and save as csv
    pd.concat(brain_behav).to_csv(f'{behav_dir}{Path("/")}brain_behav.csv')

    return brain_behav


def power_criterion_corr():

    """This function correlates the power difference between low and high
    expectations trials with the difference in dprime and criterion for
    different frequency bands.
    """

    # load brain behav dataframe
    df = pd.read_csv(f'{behav_dir}{Path("/")}brain_behav.csv')

    freqs = ['alpha_150to0', 'beta_150to0']

    diff_p_list = []

    # now get power per participant and per condition
    for f in freqs:

        power = df.groupby(['ID', 'cue']).mean()[f]

        low = power.unstack()[0.25].reset_index()
        high = power.unstack()[0.75].reset_index()

        # log transform and calculate the difference
        diff_p = np.log(np.array(low[0.25]))-np.log(np.array(high[0.75]))

        diff_p_list.append(diff_p)

    # add the power difference to a dictionary
    power_dict = {'alpha': diff_p_list[0], 'beta': diff_p_list[1]}

    # load random effects
    #re = pd.read_csv(f'{Path("D:/expecon_ms/data/behav/mixed_models/brms/random_effects.csv")}')
    
    # load behavioral SDT data
    out = figure1.prepare_for_plotting()
        
    sdt = out[0][0]
    # calculate criterion difference
    c_diff = np.array(sdt.criterion[sdt.cue == 0.25]) - np.array(sdt.criterion[sdt.cue == 0.75])
    # dprime difference between conditions
    dprime_diff = np.array(sdt.dprime[sdt.cue == 0.25]) - np.array(sdt.dprime[sdt.cue == 0.75])

    # add the sdt data to a dictionary
    sdt_params = {'dprime': dprime_diff, 'criterion': c_diff}

    for p_key, p_value in power_dict.items(): # loop over the power difference
        for keys, values in sdt_params.items(): # loop over the criterion and dprime difference

            # plot regression line
            fig = sns.regplot(p_value, values)

            plt.xlabel(f'{p_key} power difference')
            plt.ylabel(f'{keys} difference')
            plt.show()
            # calculate correlation
            print(scipy.stats.spearmanr(p_value, values))
            # save figure
            fig.figure.savefig(f'{savedir_fig5}{Path("/")}{p_key}_{keys}_.svg')
            plt.show()