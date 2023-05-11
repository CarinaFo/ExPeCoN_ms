#################################################################################################
# investigate pre-stimulus power per trial and frequency band
##################################################################################################

# import packages
import os
import mne
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import matplotlib.pyplot as plt
import random

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# datapaths
tfr_single_trial_power_dir = "D:\\expecon_ms\\data\\eeg\\sensor\\induced_tfr\\single_trial_power"
savedir_figure4 = 'D:\\expecon_ms\\figs\\manuscript_figures\\Figure4'
dir_cleanepochs = "D:\\expecon_ms\\data\\eeg\\prepro_ica\\clean_epochs"
behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')


def load_power_per_trial():
    
    diff_all_subs, diff_all_subs_hitmiss = [], []

    # load behavioral data

    os.chdir(behavpath)

    data = pd.read_csv("prepro_behav_data_after_rejectepochs.csv")

    for counter, subj in enumerate(IDlist):

        # skip those participants
        if subj == '040' or subj == '045' or subj == '032' or subj == '016':
            continue

        power = mne.time_frequency.read_tfrs(f"{tfr_single_trial_power_dir}\\{subj}_power_per_trial-tfr.h5")[0]
        # add metadata
        subj_data = data[data.ID == counter+7]

        power.metadata = subj_data


# average over specific frequency bands

    for f in frequency_list:

        if f == 'alpha':

            power = mne.time_frequency.tfr_multitaper(epochs, n_cycles=ncycles, freqs=freqs, return_itc=False,
                                                        n_jobs=15, average=False)
            if baseline == 1:
                power.apply_baseline(base_interval, mode='zscore')

            range = (7, 14)
            power.crop(-0.5, 0, range[0], range[1]).pick_channels(channels)
            alpha_power.append(power.data[:, :, :, :])  # epochsxchannelsxfrequenciesxtimepoints


        elif f == 'low_beta':
            power = mne.time_frequency.tfr_multitaper(epochs, n_cycles=ncycles, freqs=freqs, return_itc=False,
                                                        n_jobs=15, average=False)
            if baseline == 1:
                power.apply_baseline(base_interval, mode='zscore')

            range = (15, 21)
            power.crop(-0.5, 0, range[0], range[1]).pick_channels(channels)
            low_beta_power.append(power.data[:,:,:,:])

        elif f == 'high_beta':
            power = mne.time_frequency.tfr_multitaper(epochs, n_cycles=ncycles, freqs=freqs, return_itc=False,
                                                        n_jobs=15, average=False)
            if baseline == 1:
                power.apply_baseline(base_interval, mode='zscore')

            range = (22, 35)
            power.crop(-0.5, 0, range[0], range[1]).pick_channels(channels)
            high_beta_power.append(power.data[:,:,:,:])
                

    #calculate mean over time and frequencies

    mean_alpha = [np.mean(a, axis=(1, 2, 3)) for a in alpha_power]

    mean_alpha = np.concatenate(mean_alpha)

    mean_low_beta = [np.mean(a, axis=(1, 2, 3)) for a in low_beta_power]

    mean_beta_low = np.concatenate(mean_low_beta)
    
    mean_high_beta = [np.mean(a, axis=(1, 2, 3)) for a in high_beta_power]

    mean_beta_high = np.concatenate(mean_high_beta)

    return metadata_list, mean_alpha, mean_beta_low, mean_beta_high

def power_criterion_corr():

    """function that saves alpha and beta power per trial with the behavioral data in a csv file
    correlates and extracts criterion change and prestimulus power change and plots regression plot
    """
    df = pd.concat(metadata_list)

    df['alpha'] = mean_alpha
    df['low_beta'] = mean_beta_low
    df['high_beta'] = mean_beta_high
    #save as .csv dataframe for statistical analysis

    os.chdir(savepath_power_trial)

    df.to_csv('single_trial_power_' + str(abs(tmin)) + 'to' + str(abs(tmax)) + '.csv')

    freqs = ['alpha', 'low_beta', 'high_beta']

    diff_p_list, diff_s_list = [], []

    # now get beta power per participant and per condition
    for f in freqs:

        power = df.groupby(['ID', 'cue']).mean()[f]

        low = power.unstack()[0.25].reset_index()
        high = power.unstack()[0.75].reset_index()

        diff_p = np.log(np.array(high[0.75]))-np.log(np.array(low[0.25]))

        diff_p_list.append(diff_p)

        # get SDT parameters per participant

    out = prepare_data(df)
        
    crit = out[0][2]

    dprime = out[0][3]

    sdt_params = {'criterion': crit, 'd_prime': dprime}

    for keys,values in sdt_params.items():

        low = values[keys][:41]

        high = values[keys][41:]

        diff = np.array(high) - np.array(low)

        diff_s_list.append(diff)

    for p in diff_p_list: # loop over the power difference
        for s in diff_s_list: # loop over the criterion and dprime difference

            print(scipy.stats.pearsonr(s, p))

            fig = sns.regplot(s, p)
            
            os.chdir(r"D:\expecon_ms\figs\brain_behavior")

            fig.figure.savefig(f'{keys}_.svg')

            plt.show()

    return df


def prepare_data(data=None):

    """this function clean the data for each variable. The data is saved as two list (condition and congruency) with the dataframes that will be passed into the function"""
    # Prepare the data for the hitrate
    signal = data[data.isyes == 1]
    signal_grouped = signal.groupby(['ID', 'cue']).mean()['sayyes']
    signal_grouped.head()

    low_condition = signal_grouped.unstack()[0.25].reset_index()
    high_condition = signal_grouped.unstack()[0.75].reset_index()

    # Prepare the data to input for the function
    hitrate_merge = pd.merge(low_condition, high_condition)
    hitrate_melt = hitrate_merge.melt(id_vars='ID', value_name='Value', var_name='Variable')

    hitrate_condition = hitrate_melt.rename(columns={'Value': 'hitrate', 'Variable': 'condition'})

    # Prepare the data for the false alarm rate
    noise = data[data.isyes == 0]
    noise_grouped = noise.groupby(['ID', 'cue']).mean()['sayyes']
    noise_grouped.head()

    low_condition = noise_grouped.unstack()[0.25].reset_index()
    high_condition = noise_grouped.unstack()[0.75].reset_index()

    # Prepare the data to input for the function
    farate_merge = pd.merge(low_condition, high_condition)
    farate_melt = farate_merge.melt(id_vars='ID', value_name='Value', var_name='Variable')

    farate_condition = farate_melt.rename(columns={'Value': 'farate', 'Variable': 'condition'})

    # Calculate SDT from hitrate and false alarm rate
    hitrate_low = signal_grouped.unstack()[0.25]
    farate_low = noise_grouped.unstack()[0.25]

    d_prime_low = pd.Series([stats.norm.ppf(h) - stats.norm.ppf(f) for h, f in zip(hitrate_low, farate_low)])
    criterion_low = pd.Series([-0.5 * (stats.norm.ppf(h) + stats.norm.ppf(f)) for h, f in zip(hitrate_low, farate_low)])

    hitrate_high = signal_grouped.unstack()[0.75]
    farate_high = noise_grouped.unstack()[0.75]
    # Avoid infinity
    farate_high[16] = 0.00001

    d_prime_high = pd.Series([stats.norm.ppf(h) - stats.norm.ppf(f) for h, f in zip(hitrate_high, farate_high)])
    criterion_high = pd.Series([-0.5 * (stats.norm.ppf(h) + stats.norm.ppf(f)) for h, f in zip(hitrate_high, farate_high)])

    # Prepare the data for the criterion
    criterion_high = pd.DataFrame(criterion_high, columns=['values'])
    criterion = pd.DataFrame(criterion_low).join(criterion_high)
    criterion.rename(columns={0: '0.25', 'values': '0.75'}, inplace=True)

    criterion['ID'] = criterion.index

    criterion_melt = criterion.melt(id_vars='ID', value_name='Value', var_name='Variable')

    criterion_condition = criterion_melt.rename(columns={'Value': 'criterion', 'Variable': 'condition'})

    # Prepare the data for the sensitivity
    d_prime_high = pd.DataFrame(d_prime_high, columns=['values'])
    d_prime = pd.DataFrame(d_prime_low).join(d_prime_high)
    d_prime.rename(columns={0: '0.25', 'values': '0.75'}, inplace=True)

    d_prime['ID'] = d_prime.index

    d_prime_melt = d_prime.melt(id_vars='ID', value_name='Value', var_name='Variable')

    d_prime_condition = d_prime_melt.rename(columns={'Value': 'd_prime', 'Variable': 'condition'})

    # Congruency effects on confidence

    # Create new column that defines congruency
    data['congruency'] = ((data.cue == 0.25) & (data.sayyes == 0) | (data.cue == 0.75) & (data.sayyes == 1))

    data_grouped = data.groupby(['ID', 'congruency']).mean()['conf']

    con_confidence = data_grouped.unstack()[True].reset_index()
    incon_confidence = data_grouped.unstack()[False].reset_index()

    # Prepare the data to input for the function
    confidence_merge = pd.merge(con_confidence, incon_confidence).melt(id_vars='ID', value_name='Value', var_name='Variable')

    confidence_congruency = confidence_merge.rename(columns={'Value': 'confidence','Variable': 'congruency'})

    # Prepare the data for accuracy

    # Create a new column to define accuracy
    data['correct'] = ((data.isyes == 1) & (data.sayyes == 1) | (data.isyes == 0) & (data.sayyes == 0))

    data_grouped = data.groupby(['ID', 'congruency']).mean()['correct']

    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()

    # Prepare the data to use for the function
    congruency_merge = pd.merge(con_condition, incon_condition)
    accuracy_congruency = congruency_merge.melt(id_vars='ID', value_name='Value', var_name='Variable')

    accuracy_congruency = accuracy_congruency.rename(columns={'Value': 'accuracy','Variable': 'congruency'})

    data_grouped_cue = data.groupby(['ID','cue']).mean()['correct']

    acc_low = data_grouped_cue.unstack()[0.25].reset_index()
    acc_high = data_grouped_cue.unstack()[0.75].reset_index()

    accuracy_merge = pd.merge(acc_low,acc_high)
    accuracy_condition = accuracy_merge.melt(id_vars='ID', value_name='Value', var_name='Variable')

    accuracy_condition = accuracy_condition.rename(columns={'Value': 'accuracy','Variable': 'condition'})

    # Prepare the data for reaction times
    # By cue condition

    RT_grouped_cue = data.groupby(['ID', 'cue']).mean()['respt1']

    low_condition = RT_grouped_cue.unstack()[0.25].reset_index()
    high_condition = RT_grouped_cue.unstack()[0.75].reset_index()

    RT_cue_merge = pd.merge(low_condition,high_condition)
    RT_condition = RT_cue_merge.melt(id_vars='ID', value_name='Value', var_name='Variable')

    RT_condition = RT_condition.rename(columns={'Value': 'RT','Variable': 'condition'})

    # By congruency
    RT_grouped_congruency = data.groupby(['ID', 'congruency']).mean()['respt1']

    low_condition = RT_grouped_congruency.unstack()[True].reset_index()
    high_condition = RT_grouped_congruency.unstack()[False].reset_index()

    RT_congruency_merge = pd.merge(low_condition,high_condition)
    RT_congruency = RT_congruency_merge.melt(id_vars='ID', value_name='Value', var_name='Variable')

    RT_congruency = RT_congruency.rename(columns={'Value': 'RT','Variable': 'congruency'})

    list_condition = [hitrate_condition, farate_condition, criterion_condition, d_prime_condition, accuracy_condition, RT_condition]
    list_congruency = [confidence_congruency, accuracy_congruency, RT_congruency]

    return list_condition, list_congruency




