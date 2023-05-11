########################################induced time frequency representation analysis######################################


# Author: Carina Forster
# Date: 2023-04-03

# This script runs time-frequency analysis on clean epochs using morlet wavelets or multitaper
# it allows to extract epochs based on a attribute if metadata is attached to the dataframe
# subtraction of the evoked potential is included
# UPDATE: included zero-padding (symmetric) to exclude edge and post-stimulus artifacts

# Functions:
# prepare_tfcontrasts: calculate time-frequency estimates using multitaper or morlet wavelets and save to 
#                      h5 file to disk (can be used for single trial analysis)
# run_ttest: runs a paired ttest between 2 conditions and plots t values in sensor space over
#               specified time and frequency range
# cluster_test: runs a cluster based permutation test over channels,time and frequencies
#               and plots the results in sensor space

# load packages

import os
import mne
import numpy as np
import scipy.stats
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import scipy.ndimage as ndimage
import seaborn as sns

# set matplotlib backend to qt
%matplotlib qt

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# set paths
dir_cleanepochs = r"D:\expecon_ms\data\eeg\prepro_ica\clean_epochs"
dir_droplog =  r"D:\expecon_ms\data\eeg\prepro_ica\droplog"
savepath_tfr_multitaper = r"D:\expecon_ms\data\eeg\sensor\induced_tfr\tfr_multitaper"

# savepath figures
savepath_figs = r"D:\expecon_ms\figs\eeg\sensor\tfr"
behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'

# participant index
IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')

# Functions

def prepare_tfcontrasts(tmin=-1, tmax=1, ncycles=3.0, fmin=7, fmax=30,
                        baseline=0, baseline_interval=(-0.8, -0.5), induced=1,
                        mode='mean', sfreq=250,
                        save=1, laplace=1, method='multitaper', cond="highlow_prevno", zero_pad=0,
                        reject_criteria=dict(eeg=200e-6),  # 200 µV
                        flat_criteria=dict(eeg=1e-6), savedropfig=0):

    """this function runs morlet wavelets or multitaper on clean,epoched data
    it allows to extract epochs based on a attribute if metadata is attached to the dataframe
    subtraction of the evoked potential is included
    UPDATE: included zero-padding (symmetric) to exclude edge and post-stimulus artifacts
    see zero_pad_data() function docstring for more infos on zero padding

    output: list of TF output per subject (saved as h5 file)
    PSD shape: n_subjects,n_epochs, n_channels, n_freqs for power spectral density
    (can be used for single trial analysis)"""

    freqs = np.arange(fmin, fmax+1, 1) # define frequencies of interest
    n_cycles = freqs / ncycles  # different number of cycles per frequency

    all_tfr_a, all_tfr_b = [], []

    # loop over participants
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
        # some weird trigger stuff going on?
        epochs = epochs[epochs.metadata.trial != 1]

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

        if savedropfig == 1:

            fig = epochs.plot_drop_log(show=False)

            fig.savefig(f"{dir_droplog}/P{subj}_droplog.png", dpi=300, bbox_inches='tight')

        # apply laplace filter
        if laplace == 1:

            epochs = mne.preprocessing.compute_current_source_density(epochs)
        

        # create experimental conditions
        if cond == "hitmiss":

            epochs_a = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1))]
            epochs_b = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 0))]

            cond_a = "hit"
            cond_b = "miss"

        elif cond == "prev_choice":

            epochs_a = epochs[epochs.metadata.prevsayyes == 1]
            epochs_b = epochs[epochs.metadata.prevsayyes == 0]

            cond_a = "prev_yes"
            cond_b = "prev_no"

        elif cond == "highlow_prevyes":

            epochs_a = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.prevsayyes == 1))]
            epochs_b = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.prevsayyes == 1))]

            cond_a = "high_yes"
            cond_b = "low_yes"

        elif cond == "highlow_prevno":

            epochs_a = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.prevsayyes== 0))]
            epochs_b = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.prevsayyes == 0))]

            cond_a = "high_no"
            cond_b = "low_no"

        elif cond == "highlow":
        
            epochs_a = epochs[(epochs.metadata.cue == 0.75)]
            epochs_b = epochs[(epochs.metadata.cue == 0.25)]

            cond_a = "high"
            cond_b = "low"

        elif cond == "signalnoise":

            epochs_a = epochs[epochs.metadata.isyes == 1]
            epochs_b = epochs[epochs.metadata.isyes == 0]

            cond_a = "signal"
            cond_b = "noise"

        # make sure there is the same amount of trials in both conditions
        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        if induced:

            # subtract evoked activity in the time window of interest for each condition separately
            epochs_a = epochs_a.subtract_evoked()
            epochs_b = epochs_b.subtract_evoked()

        # crop the data in the desired time window
        epochs_a.crop(tmin, tmax)
        epochs_b.crop(tmin, tmax)

        # extract the data from the epochs structure
        dataa = epochs_a.get_data()
        datab = epochs_b.get_data()  # epochs*channels*times

        # zero pad the data on both ends with the signal length to avoid leakage and edge artifacts
        if zero_pad == 1:

            dataa = zero_pad_data(dataa)
            datab = zero_pad_data(datab)

            #put back into epochs structure

            epochs_a = mne.EpochsArray(dataa, epochs_a.info, tmin=tmin*2)

            epochs_b = mne.EpochsArray(datab, epochs_b.info, tmin=tmin*2)

        # tfr using morlet wavelets or multitiaper

        if method == 'morlet':

            tfr_a = mne.time_frequency.tfr_morlet(epochs_a, freqs=freqs, n_cycles=n_cycles, return_itc=False,
                                                         n_jobs=-1, output='power', average=True, use_fft=True)

            tfr_b = mne.time_frequency.tfr_morlet(epochs_b, freqs=freqs, n_cycles=n_cycles, return_itc=False,
                                                    n_jobs=-1, output='power', average=True, use_fft=True)


            # add tfr per particiapnt to a list
            all_tfr_a.append(tfr_a)
            all_tfr_b.append(tfr_b)

        else:

            tfr_a = mne.time_frequency.tfr_multitaper(epochs_a, n_cycles=ncycles, freqs=freqs, return_itc=False,
                                                      n_jobs=-1)
            tfr_b = mne.time_frequency.tfr_multitaper(epochs_b, n_cycles=ncycles, freqs=freqs, return_itc=False,
                                                      n_jobs=-1)
            
            # add tfr per particiapnt to a list
            all_tfr_a.append(tfr_a)
            all_tfr_b.append(tfr_b)

    # save tfrs to disk

    if method == 'morlet':
        os.chdir(savepath_tfr_morlet)
    else:
        os.chdir(savepath_tfr_multitaper)

    mne.time_frequency.write_tfrs(f"{method}_{cond_a}-tfr.h5", all_tfr_a)


    mne.time_frequency.write_tfrs(f"{method}_{cond_b}-tfr.h5", all_tfr_b)


    return all_tfr_a, all_tfr_b, fmin, fmax, freqs, tmin, tmax, cond_a, cond_b, epochs_a, epochs_b, method


def run_ttest(channel_names = ['CP4', 'CP6', 'C4', 'C6']):

    """run a simple t test between conditions without correcting for multiple comparisions and plot the values"""

    a, b, fmin, fmax, freqs, tmin, tmax, cond_a, cond_b, epochs_a_padded, epochs_b_padded, tf_method = prepare_tfcontrasts()
    a, b, fmin, fmax, freqs, tmin, tmax, cond_a, cond_b, epochs_a_padded, epochs_b_padded, tf_method = out

    os.chdir("D:\\expecon_ms\\data\\eeg\\sensor\\induced_tfr\\tfr_multitaper")
  
    a = mne.time_frequency.read_tfrs(f"{method}_{cond_a}-tfr.h5")
    b = mne.time_frequency.read_tfrs(f"{method}_{cond_b}-tfr.h5")

    acrop = [ax.copy().crop(tmin, tmax) for ax in a]
    bcrop = [bx.copy().crop(tmin, tmax) for bx in b]

    channel_names = [epochs_a_padded[0].pick_types(eeg=True).ch_names[ch] for ch in channel_names]

    # baseline correction of TF data

    if zscore:

        acrop = [a.apply_baseline((-1,-0.5), mode="percent") for a in acrop]
        bcrop = [a.apply_baseline((-1,-0.5), mode="percent") for a in bcrop]

    # plot grand average per condition and the difference between conditions

    a_gra = mne.grand_average([ax.copy().pick_channels(channel_names) for ax in acrop])
    b_gra = mne.grand_average([bx.copy().pick_channels(channel_names) for bx in bcrop])

    diff_gra = a_gra - b_gra

    a_gra.plot()
    b_gra.plot()

    diff = diff_gra.plot(show=False)

    os.chdir(savepath_figs)

    plt.savefig(f"evoked_grand_average_{cond_a}_{cond_b}.svg")

    info = acrop[0].info
    times = acrop[0].times

    power_a_array = np.array([a.data for a in acrop])
    power_b_array = np.array([b.data for b in bcrop])

    X = np.array(power_a_array)-np.array(power_b_array)

    n_subs, n_chan, n_freqs, n_timepoints = power_a_array.shape

    stat_cond, pval_cond = scipy.stats.ttest_rel(power_a_array[:,:,:,:], power_b_array[:,:,:,:], axis=0)

    fmin=7
    fmax=30

    freqs = np.arange(fmin, fmax+1,1) 
    Conab = mne.time_frequency.AverageTFR(info, stat_cond, times, freqs[:], nave=len(IDlist))

    fig = mne.viz.plot_tfr_topomap(Conab, colorbar=True, size=11, tmin=times[0], tmax=times[-1], show_names=True,
                             cbar_fmt='%1.1f')
    
    os.chdir(savepath_figs)
    
    fig.figure.savefig(f"{cond_a}_{cond_b}_{str(tmin)}_{str(tmax)}.svg")

    return power_a_array, power_b_array, n_subs, n_chan, n_freqs, n_timepoints, stat_cond, freqs,tmin,tmax, cond_a, cond_b,\
           times, info, tf_method, acrop


def cluster_time_frequency(a=None, b=None, jobs=15, n_perm=10000):

    """runs a cluster permutation test over time and frequency space (2D), select channel 
    or average over specified channels (ROI), uses threshold free cluster enhancement to determine cluster threshold,
    channels are picked based on topo plot (see above)
    Plots cluster output on top of T-map, saves figure as svg file
    input 3D numpy array: 3D: participants, channels, timepoints """

    a, b, n_subs, n_chan, n_freqs, n_timepoints, stat_cond, freqs,tmin,tmax, cond_a, cond_b,\
           times, info, tf_method, acrop = run_ttest()

    spec_channel_list = []

    for i, channel in enumerate(channel_names):
        spec_channel_list.append(a[15].ch_names.index(channel))
    spec_channel_list

    mean_over_channels = np.mean(power_a_array[:, spec_channel_list, :, :], axis=1)
    mean_over_channels_b = np.mean(power_b_array[:, spec_channel_list, :, :], axis=1)

    X = mean_over_channels - mean_over_channels_b

    threshold_tfce = dict(start=0, step=0.1)

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        X[:,:,:],
        n_jobs=-1, n_permutations=n_perm,  tail=0, seed=1) # threshold=threshold_tfce)
    
    # 3D cluster test

    # load cleaned epochs
    subj='007'
    epochs = mne.read_epochs(f"{dir_cleanepochs}/P{subj}_epochs_after_ica-epo.fif")

    ch_adjacency,_ = mne.channels.find_ch_adjacency(epochs.info, ch_type='eeg')
    n_times = len(epochs.crop(tmin, tmax).times)
    n_freqs = len(freqs)

    adjacency = mne.stats.combine_adjacency(
        n_times,  # regular lattice adjacency for times
        n_freqs,  # regular lattice adjaency for freqs
        ch_adjacency,  # custom matrix, or use mne.channels.find_ch_adjacency
        ) 

    X = np.transpose(X, [0, 3, 2, 1])

    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
        X[:,:,:,:], adjacency=adjacency, threshold=3,
        n_jobs=-1, n_permutations=n_perm,  tail=0)

    cluster_p = cluster_p_values.reshape(X.shape[1],X[:,:,:].shape[2])

    # Apply the mask to the image
    masked_img = T_obs.copy()
    masked_img[np.where(cluster_p_values > 0.05)] = 0

    vmax = np.max(T_obs)
    vmin = np.min(T_obs)

    #cnorm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)  # min, center & max ERDS
    
    # add time on the x axis 
    x = np.linspace(-0.5,0,126)
    y = np.arange(7,31,1)

    # Plot the original image with lower transparency
    fig = plt.imshow(T_obs, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto',
    vmin=vmin, vmax=vmax, cmap='viridis')
    plt.colorbar()
    # Plot the masked image on top
    fig = plt.imshow(masked_img, alpha=0.5, origin='lower', extent=[x.min(), x.max(), y.min(), y.max()],
    aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')

    # Add x and y labels
    plt.xlabel('Time (s)')
    plt.ylabel('Freq (Hz)')

    # save figure
    os.chdir(savepath_figs)

    fig.figure.savefig(f"cluster_perm_{cond_a}_{cond_b}_{str(tmin)}_{str(tmax)}.svg")

    # Show the plot
    plt.show()


df = pd.read_csv("D:\\expecon_ms\\data\\behav\\sdt_para.csv")
c = pd.read_csv("D:\\expecon_ms\\data\\behav\\criterion_per_cond.csv")
df = df.drop(9, axis=0)

behavior = df.iloc[:,1].values  # create np array of behavior

#  plot 2D correlation matrix of observed data
fig = plt.figure()
ax = sns.heatmap(corr_matrix)
ax.invert_yaxis()  # to have lowest freq at the bottom


    ########################################################## Helper functions ################################################

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



def permutation_cluster_correlation_test(X, behavior, test='pearson', threshold=0.01, n_permutations=100):
    '''performs cluster based permutation test on time freq correlation data

    each time freq voxel is correlated with behavioral data (across participants),
    based on threshold only sig correlations are kept and clustered,
    cluster mass (= T-value) is calculated as the sum of t-values of neighboring significant
    correlations, observed cluster mass of the biggest T-value and second biggest T-value is
    returned,
    in addition T-value distribution is returned when behavior array is permuted and
    biggest cluster mass is computed each time

    Parameters
    ----------
    X : ndarray
        eeg data, expected shape is (subjects, frequencies, timepoints)
    behavior : ndarray
        behavioural variable, expected shape is (subjects)
    test : str, optional
        'pearson' or 'spearman', by default 'pearson'
    threshold : float, optional
        initial clustering threshold - vertices with data values more 
        extreme than threshold will be used to form clusters, by default 0.05
    n_permutations : int, optional
        how often to permute behavioral data, by default 100

    Returns
    -------
    corr_matrix : ndarray
        correlation values for each time frequency voxel with behavioral data,
        expected shape is (frequencies, timepoints)
    cluster_matrix : ndarray
        each time frequency voxel which does not belong to a cluster is 0,
        cluster voxels are numbered, adjacent voxels have the same number,
        expected shape is (frequencies, timepoints)
    n_cluster : int
        number of clusters found
    observed_T : float
        cluster mass of the biggest observed cluster
    observed_T_2 : float
        cluster mass of the second biggest observed cluster
    T_distribution : ndarray
        all T values from permuted correlations tests,
        expected shape is (n_permutations,)
    '''

    # create correlation matrix
    corr_matrix = np.zeros([X.shape[1], X.shape[2]])
    t_matrix = np.zeros([X.shape[1], X.shape[2]])
    p_matrix = np.zeros([X.shape[1], X.shape[2]])
    n_VP = X.shape[0]
    n_freq = X.shape[1]
    n_time = X.shape[2]


    # calculate observed r, p and t values for each voxel
    for f in range(n_freq):
        for time in range(n_time):
            ERD = X[:, f, time]
            if test == 'pearson':
                r = np.corrcoef(ERD, behavior)[0, 1]  # pearson correlation
            else:
                r = stats.spearmanr(ERD, behavior)[0] # spearman correlation
            corr_matrix[f, time] = r

            # calculates t-values for each correlation value
            # is this also correct for spearman??
            t_value = (r * np.sqrt(n_VP - 2)) / (np.sqrt(1 - np.square(r)))
            t_matrix[f, time] = t_value

            # calculate p value based on t value
            # is this also correct for spearman??
            p = (1 - stats.t.cdf(x=abs(t_value), df=n_VP - 2)) * 2
            p_matrix[f, time] = p

    # keep only sig cluster
    # set all p-values to 0 where p-value is > .05
    p_matrix = np.where(p_matrix < threshold, p_matrix, 0)

    # label features: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
    cluster = ndimage.label(p_matrix)
    cluster_matrix = cluster[0]
    n_cluster = cluster[1]

    # sum t values of individual clusters
    input_ndi = t_matrix
    labels = cluster_matrix
    index = np.arange(1, n_cluster, 1)
    t_sum_all_observed = ndimage.sum_labels(input_ndi, labels, index)

    def sort_abs(arr):
        return sorted([abs(x) for x in arr])

    if t_sum_all_observed is not None:
        # sort t values of individual clusters
        t_sum_all_observed_abs = sort_abs(t_sum_all_observed)

        observed_T = t_sum_all_observed_abs[-1]

        if len(t_sum_all_observed_abs) > 1:
            observed_T_2 = t_sum_all_observed_abs[-2]
        else:    
            observed_T_2 = 0
    else:
        print("no clusters found")

    # now calculate big T for permuted behavioral data (correlation matrix)
    # https://benediktehinger.de/blog/science/statistics-cluster-permutation-test/
    # preallocate array
    T_distribution = np.zeros(n_permutations)
    
    for i, shuffle in enumerate(T_distribution):
        np.random.shuffle(behavior)  # shuffle behavior values randomly

        for f in range(n_freq):
            for time in range(n_time):
                ERD = X[:, f, time]
                r = np.corrcoef(ERD, behavior)[0, 1]  # pearson correlation

                t_value = (r * np.sqrt(n_VP - 2)) / (np.sqrt(1 - np.square(r)))
                t_matrix[f, time] = t_value

                p = (1 - stats.t.cdf(x=abs(t_value), df=n_VP - 2)) * 2
                p_matrix[f, time] = p

        # keep only sig cluster
        # set all t-values to 0 where p-value is > .05
        t_matrix = np.where(p_matrix <= threshold, t_matrix, 0)

        # label features: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.label.html
        cluster = ndimage.label(t_matrix)
        cluster_matrix_perm = cluster[0]
        n_cluster = cluster[1]

        # sum t values of individual clusters
        input_ndi = t_matrix
        labels = cluster_matrix_perm
        index = np.arange(1, n_cluster, 1)
        t_sum_all = ndimage.sum_labels(input_ndi, labels, index)

        t_sum_all_abs = abs(t_sum_all)

        cluster_mass = t_sum_all_abs.max()

        T_distribution[i] = cluster_mass

    return corr_matrix, cluster_matrix, observed_T, T_distribution