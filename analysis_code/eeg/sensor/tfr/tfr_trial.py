#################################################################################################
# investigate pre-stimulus power per trial and frequency band
##################################################################################################

# import packages
import os
import random
import sys

# add path to sys.path.append() if package isn't found
sys.path.append('D:\\expecon_ms\\analysis_code')

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import seaborn as sns
from behav import figure1

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# datapaths

behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'
tfr_single_trial_power_dir = "D:\\expecon_ms\\data\\eeg\\sensor\\tfr"
brain_behav_path = "D:\\expecon_ms\\figs\\brain_behavior"

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')

freq_list = np.arange(6, 35, 1)

freq_bands = {'alpha': (7, 13), 'beta': (15, 25)}

def save_band_power_per_trial(time_intervals={'800to500': (-0.8,-0.5), '400to100': (-0.4,-0.1)}):

    """This function saves the power per trial per frequency band in a csv file. 
    The power is calculated for the prestimulus period and averaged over the channel C4.
    The power is calculated for the following frequency bands:
        - alpha (8-14 Hz)
        - beta (14-21 Hz)
    """

    # load behavioral data
    data = pd.read_csv(f"{behavpath}//prepro_behav_data_after_dropbads.csv")

    # save single subject dataframes in a list
    brain_behav = []

    for counter, subj in enumerate(IDlist):

        # load single trial power
        power = mne.time_frequency.read_tfrs(f"{tfr_single_trial_power_dir}\\{subj}_power_per_trial_induced-tfr.h5")[0]

        # add metadata
        subj_data = data[data.ID == counter+7]
        power.metadata = subj_data

        for keys, values in time_intervals.items():

            power_crop = power.copy().crop(tmin=values[0], tmax=values[1]).pick_channels(['C4'])

            # now we average over time and channels
            power_crop.data = np.mean(power_crop.data, axis=(1, 3))

            # now we average over the frequency band and time interval 
            # and add the column to the 
            # behavioral data frame
            subj_data[f'alpha_{keys}'] = np.mean(power_crop.data[:, 1:8], axis=1)  # 7-14 Hz
            subj_data[f'beta_{keys}'] = np.mean(power_crop.data[:, 9:20], axis=1) # 15-25 Hz

        # save the data in a list
        brain_behav.append(subj_data)

    # concatenate the list of dataframes and save as csv
    pd.concat(brain_behav).to_csv(f"{brain_behav_path}\\brain_behav.csv")


def power_criterion_corr():

    """This function correlates the power difference between low and high
    expectations trials with the difference in dprime and criterion for
    different frequency bands. 
    """

    df = pd.read_csv(f"{brain_behav_path}\\brain_behav.csv")

    freqs = ['alpha_400to100', 'beta_400to100']

    diff_p_list = []

    # now get beta power per participant and per condition
    for f in freqs:

        power = df.groupby(['ID', 'cue']).mean()[f]

        low = power.unstack()[0.25].reset_index()
        high = power.unstack()[0.75].reset_index()

        diff_p = np.log(np.array(low[0.25]))-np.log(np.array(high[0.75]))

        diff_p_list.append(diff_p)

    power_dict = {'alpha': diff_p_list[0], 'beta': diff_p_list[1]}

    #load random effects
    re = pd.read_csv("D:\expecon_ms\data\\behav\mixed_models\\brms\\random_effects.csv")
    
    out = figure1.prepare_for_plotting()
        
    sdt = out[0][0]

    c_diff = np.array(sdt.criterion[sdt.cue == 0.25]) - np.array(sdt.criterion[sdt.cue == 0.75])
    dprime_diff = np.array(sdt.dprime[sdt.cue == 0.25]) - np.array(sdt.dprime[sdt.cue == 0.75])

    sdt_params = {'dprime': dprime_diff, 'criterion': c_diff}

    for p_key, p_value in power_dict.items(): # loop over the power difference
        for keys, values in sdt_params.items(): # loop over the criterion and dprime difference

            fig = sns.regplot(p_value, values)

            plt.xlabel(f'{p_key} power difference')
            plt.ylabel(f'{keys} difference')
            plt.show()

            print(scipy.stats.spearmanr(p_value, values))

            os.chdir(r"D:\expecon_ms\\figs\brain_behavior")

            fig.figure.savefig(f'{p_key}_{keys}_.svg')

            plt.show()