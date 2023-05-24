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
from behavioral_data_analysis import figure1

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# datapaths

behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'
tfr_single_trial_power_dir = "D:\\expecon_ms\\data\\eeg\\sensor\\induced_tfr\\single_trial_power"
brain_behav_path = "D:\\expecon_ms\\figs\\brain_behavior"

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')

freq_list = np.logspace(*np.log10([6, 35]), num=12)

freq_bands = {'theta': (6, 9), 'alpha': (8, 14), 'low_beta': (14, 21), 'beta_gamma': (20, 36)}


def save_band_power_per_trial():

    """This function saves the power per trial per frequency band in a csv file. 
    The power is calculated for the prestimulus period (-400 to -200ms) and
    averaged over the channels CP4, C4, C6, CP6.
    The power is calculated for the following frequency bands:
        - theta (6-9 Hz)
        - alpha (8-14 Hz)
        - low beta (14-21 Hz)
        - beta gamma (20-36 Hz)
    """

    # load behavioral data
    os.chdir(behavpath)

    data = pd.read_csv("prepro_behav_data_after_rejectepochs.csv")

    # save single subject dataframes in a list
    brain_behav = []

    for counter, subj in enumerate(IDlist):

        # skip those participants
        if subj == '040' or subj == '045' or subj == '032' or subj == '016':
            continue

        power = mne.time_frequency.read_tfrs(f"{tfr_single_trial_power_dir}\\{subj}_power_per_trial-tfr.h5")[0]
        
        # add metadata
        subj_data = data[data.ID == counter+7]
        power.metadata = subj_data

        # save power per trial per frequency band
        # strongest effects between -400 and -200ms prestimulus
        # ROI channels selected

        power.crop(-0.4, -0.2).pick_channels(['CP4', 'C4', 'C6', 'CP6'])

        # now we average over time and channels

        power.data = np.mean(power.data, axis=(1,3))

        # now we average over the frequency bands and add the column to the 
        # behavioral data frame

        subj_data['theta_pw'] = np.mean(power.data[:,0:2], axis=1) # 6-9 Hz
        subj_data['alpha_pw'] = np.mean(power.data[:,2:6], axis=1)  # 8-14 Hz
        subj_data['low_beta_pw'] = np.mean(power.data[:,5:8], axis=1) # 14-21 Hz
        subj_data['beta_gamma_pw'] = np.mean(power.data[:,9:], axis=1) # 20-36 Hz

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

    freqs = ['theta_pw', 'alpha_pw', 'low_beta_pw', 'beta_gamma_pw']

    diff_p_list = []

    # now get beta power per participant and per condition
    for f in freqs:

        power = df.groupby(['ID', 'cue']).mean()[f]

        low = power.unstack()[0.25].reset_index()
        high = power.unstack()[0.75].reset_index()

        diff_p = np.log(np.array(low[0.25]))-np.log(np.array(high[0.75]))

        diff_p_list.append(diff_p)

    power_dict = {'theta': diff_p_list[0], 'alpha': diff_p_list[1], 'low_beta': diff_p_list[2], 'beta_gamma': diff_p_list[3]}
    
    out = figure1.prepare_behav_data()
        
    dprime = out[0]

    crit = out[1]   

    sdt_params = {'dprime': dprime, 'criterion': crit}

    for p_key, p_value in power_dict.items(): # loop over the power difference
        for keys, values in sdt_params.items(): # loop over the criterion and dprime difference

            print(scipy.stats.pearsonr(p_value, values))

            fig = sns.regplot(p_value, values)
            
            os.chdir(r"D:\expecon_ms\figs\brain_behavior")

            fig.figure.savefig(f'{p_key}_{keys}_.svg')

            plt.show()