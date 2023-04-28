import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mne
import scipy.stats as stats


# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# set paths
dir_cleanepochs = r"D:\expecon_ms\data\eeg\prepro_ica\clean_epochs"
savepath_tfr_multitaper_single_trial = r"D:\expecon_ms\data\eeg\sensor\induced_tfr\tfr_multitaper\single_trial_power"
savepath_tfr_morlet_single_trial = r"D:\expecon_ms\data\eeg\sensor\induced_tfr\tfr_morlet\single_trial_power"

# savepath figures
savepath_figs = r"D:\expecon_ms\figs\eeg\sensor\tfr"

# participant index
IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')

# load c_diff, d_diff and prev_c

df = pd.read_csv("D:\\expecon_ms\\data\\behav\\sdt_para.csv")

method='multitaper'
cond_a = 'high'
cond_b = 'low'
channel_names = ['CP4', 'CP6', 'C4', 'C6']
channel_names = ['CP4', 'CP6', 'C4', 'C6', 'C2', 'CP2', 'P2', 'P4', 'P6']
fmin=7
fmax=35
freqs = np.arange(fmin, fmax+1,1) # define frequencies of interest

def correlate_brain_behav():

    # load EEG data
    os.chdir(r"D:\\expecon_ms\\data\\eeg\\sensor\\evoked_tfr\\tfr_morlet\\laplace")

    a = mne.time_frequency.read_tfrs(f"{method}_{cond_a}-tfr.h5")
    b = mne.time_frequency.read_tfrs(f"{method}_{cond_b}-tfr.h5")

    X = np.array([ax.crop(-0.4,0).pick_channels(channel_names).data - bx.crop(-0.4,0).pick_channels(channel_names).data for ax,bx in zip(a,b)])

    #alpha = [:6]
    #low_beta = [6:13]
    #high_beta = [13:23]
    #gamma = [23:]

    X_avg = np.mean(X[:,:,6:13,:], axis=(1,2,3))

    np.corrcoef(df.iloc[:,1], X_avg)

    stats.pearsonr(df.iloc[:,1], X_avg)

    sns.regplot(x=df.iloc[:,1], y=X_avg)

# Functions

def prepare_tfcontrasts_singletrial(tmin=-0.4, tmax=0, ncycles=3.0, fmin=7, fmax=35,
                        baseline=0, baseline_interval=(-0.8, -0.5), induced=True,
                        mode='mean', sfreq=250,
                        save=1, laplace=1, method='morlet', zero_pad=1):

    """this function runs morlet wavelets or multitaper on clean,epoched data
    subtraction of the evoked potential is included
    UPDATE: included zero-padding (symmetric) to exclude edge and post-stimulus artifacts
    see zero_pad_data() function docstring for more infos on zero padding

    output: list of TF output per subject (saved as h5 file)
    PSD shape: n_subjects,n_epochs, n_channels, n_freqs for power spectral density
    (can be used for single trial analysis)"""

    freqs = np.arange(fmin, fmax+1,1) # define frequencies of interest
    n_cycles = freqs / ncycles  # different number of cycles per frequency

    all_tfr_a, all_tfr_b = [], []

    # loop over participants
    for counter, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)

        # skip those participants
        if subj == '040' or subj == '045' or subj == '032':
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

        if induced:

            # subtract evoked activity in the time window of interest for each condition separately
            epochs = epochs.subtract_evoked()
  
        # crop the data in the desired time window
        epochs.crop(tmin, tmax)

        # apply laplace filter
        if laplace == 1:

            epochs= mne.preprocessing.compute_current_source_density(epochs)

        # extract the data from the epochs structure
        data = epochs.get_data()

        # zero pad the data on both ends with the signal length to avoid leakage and edge artifacts
        if zero_pad == 1:

            data = zero_pad_data(data)
  
            #put back into epochs structure

            epochs = mne.EpochsArray(data, epochs.info, tmin=tmin*2)

        # tfr using morlet wavelets or multitiaper

        tfr = mne.time_frequency.tfr_morlet(epochs, n_cycles=ncycles, freqs=freqs, return_itc=False,
                                                    n_jobs=-1, average=False, output='power')

        os.chdir(savepath_tfr_morlet_single_trial)

        mne.time_frequency.write_tfrs(f"{subj}_{method}-tfr.h5", tfr, overwrite=True)


    
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



