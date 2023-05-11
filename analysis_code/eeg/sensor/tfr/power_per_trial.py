#################################################################################################
# investigate pre-stimulus power
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

def calculate_power_per_trial(tmin=-0.5, tmax=0,
                              zero_pad=1, reject_criteria=dict(eeg=200e-6),
                              flat_criteria=dict(eeg=1e-6)):

    '''calculate prestimulus alpha power per trial (should be positive and rather small)
    baseline correction can be applied. takes the mean over prestimulus period and alpha range in electrode CP4.
    Stats analysis in power_per_trial.R
    return: .csv file with behavioral columns and prestimulus alpha power per trial for further analysis in R'''

    freqs = np.logspace(*np.log10([6, 35]), num=12)
    n_cycles = freqs / 2.0  # different number of cycle per frequency

    metadata_list = []

    for counter, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)
        # skip those participants
        if subj == '040' or subj == '045' or subj == '032' or subj == '016':
            continue

        # load cleaned epochs
        epochs = mne.read_epochs(f"{dir_cleanepochs}/P{subj}_epochs_after_ica-epo.fif")

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
        
        # remove trials with rts >= 2.5 (no response trials) and trials with rts < 0.1
    
        epochs = epochs[epochs.metadata.respt1 > 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # remove first trial of each block (trigger delays)
        epochs = epochs[epochs.metadata.trial != 1]

        # subtract evoked response
        epochs = epochs.subtract_evoked()

        # load behavioral data

        os.chdir(behavpath)

        data = pd.read_csv("prepro_behav_data.csv")

        subj_data = data[data.ID == counter+7]

        if ((counter == 5) or (counter == 13) or (counter == 21) or (counter == 28)):  # first epoch has no data
            epochs.metadata = subj_data.iloc[1:, :]
        elif counter == 17:
            epochs.metadata = subj_data.iloc[3:, :]
        else:
            epochs.metadata = subj_data

        # drop bad epochs
        epochs.drop_bad(reject=reject_criteria, flat=flat_criteria)

        # add metadata per subject

        metadata_list.append(epochs.metadata)

        # crop the data in the pre-stimulus window

        epochs.crop(tmin, tmax)

        # zero pad the data on both sides to avoid leakage and edge artifacts

        data = epochs.get_data()

        if zero_pad == 1:

            data = zero_pad_data(data)

            # put back into epochs structure

            epochs = mne.EpochsArray(data, epochs.info, tmin=tmin * 2)

        # get prestimulus power

        power = mne.time_frequency.tfr_multitaper(epochs, n_cycles=n_cycles, freqs=freqs, return_itc=False,
                                                  n_jobs=-1, average=False)
        
        os.chdir(tfr_single_trial_power_dir)

        mne.time_frequency.write_tfrs(f"{subj}_power_per_trial-tfr.h5", power)


def contrast_conditions():
    
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

        power_high = power[((power.metadata.cue == 0.75) &
                           (power.metadata.isyes == 0) &
                           (power.metadata.sayyes == 0))]
        power_low = power[((power.metadata.cue == 0.25) &
                          (power.metadata.isyes == 0) &
                          (power.metadata.sayyes == 0))]

        # randomly sample from low power trials to match number of trials in high power condition

        random_sample = power_high.data.shape[0]
        idx_list = list(range(power_low.data.shape[0]))

        power_low_idx = random.sample(idx_list, random_sample)

        power_low.data = power_low.data[power_low_idx, :, :, :]

        power_hit = power[((power.metadata.isyes == 1) &
                          (power.metadata.sayyes == 1))]
        power_miss = power[((power.metadata.isyes == 1) &
                           (power.metadata.sayyes == 0))]

        evoked_power_hit = power_hit.average()
        evoked_power_miss = power_miss.average()

        evoked_power_high = power_high.average()
        evoked_power_low = power_low.average()

        diff = evoked_power_high - evoked_power_low

        diff_hitmiss = evoked_power_hit - evoked_power_miss

        diff_all_subs.append(diff)
        diff_all_subs_hitmiss.append(diff_hitmiss)

def plot_grand_average(contrast_data=None, cond='hitmiss'):

    diff_gra = mne.grand_average(contrast_data).apply_baseline(baseline=(-1, -0.8))

    diff_gra.copy().crop(-0.5, 0).plot_joint()

    topo = diff_gra.copy().crop(-0.5, 0).plot_topo()

    tfr = diff_gra.copy().crop(-0.5, 0).plot(picks=['CP4', 'CP6', 'C4', 'C6', 'P4', 'CP6'], combine='mean')[0]

    topo.savefig(f'{savedir_figure4}//{cond}_topo.svg')
    
    tfr.savefig(f'{savedir_figure4}//{cond}_tfr.svg')

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

def run_2D_cluster_perm_test(channel_names=['CP4', 'CP6', 'C4', 'C6', 'P4', 'CP6'],
                                             contrast_data=None):
    spec_channel_list = []

    for i, channel in enumerate(channel_names):
        spec_channel_list.append(contrast_data[15].ch_names.index(channel))
    spec_channel_list

    X = np.array([d.crop(-0.5, 0).data for d in contrast_data])

    mean_over_channels = np.mean(X[:, spec_channel_list, :, :], axis=1)

    threshold_tfce = dict(start=0, step=0.1)

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                            mean_over_channels,
                            n_jobs=-1, 
                            n_permutations=n_perm,
                            threshold=threshold_tfce,
                            tail=0, 
                            seed=1) # threshold=threshold_tfce)
    
def plot2D_cluster_output(cond='hitmiss'):

    min(cluster_p_values)
    
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

    # Plot the original image with lower transparency
    fig = plt.imshow(T_obs, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto',
    vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar()
    # Plot the masked image on top
    fig = plt.imshow(masked_img, alpha=0.3, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
    aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')

    # Add x and y labels
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')

    # save figure
    os.chdir(savedir_figure4)

    fig.figure.savefig(f"cluster_perm_{cond}_{str(tmin)}_{str(tmax)}.svg")

    # Show the plot
    plt.show()

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



