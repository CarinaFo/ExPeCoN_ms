# Extract beta power peaks

import random
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mne
from scipy import stats
from scipy.signal import find_peaks
from scipy import signal


# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# datapath

dir_cleanepochs = r"D:\expecon_ms\data\eeg\prepro_ica\clean_epochs"

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')


random_subs= random.sample(IDlist, 5)

for idx, subj in enumerate(random_subs):

    # print participant ID
    print('Analyzing ' + subj)
    # skip those participants
    if subj == '040' or subj == '045':
        continue

    # load cleaned epochs
    os.chdir(dir_cleanepochs)

    epochs = mne.read_epochs('P' + subj + '_epochs_after_ica-epo.fif')

    # Remove 6 blocks with hitrates < 0.2 or > 0.8

    if subj == '010':
        epochs = epochs[epochs.metadata.block != 6]
    if subj == '012':
        epochs = epochs[epochs.metadata.block != 6]
    if subj == '026':
        epochs = epochs[epochs.metadata.block != 4]
    if subj == '030':
        epochs = epochs[epochs.metadata.block != 3]
    if subj == '032':
        epochs = epochs[epochs.metadata.block != 2]
        epochs = epochs[epochs.metadata.block != 3]
    if subj == '039':
        epochs = epochs[epochs.metadata.block != 3]

    # remove trials with rts >= 2.5 (no response trials) and trials with rts < 0.1
    before_rt_removal = len(epochs.metadata)
    epochs = epochs[epochs.metadata.respt1 > 0.1]
    epochs = epochs[epochs.metadata.respt1 != 2.5]
    # some weird trigger stuff going on?
    epochs = epochs[epochs.metadata.trial != 1]

    # subtract evoked data
    epochs = epochs.subtract_evoked()
    
    epochs.filter(15, 25, fir_design='firwin')

    epochs.data = epochs.crop(-0.5,0.5).get_data()**2

    high = epochs[epochs.metadata.cue == 0.75]
    low = epochs[epochs.metadata.cue == 0.25]

    high.plot_image(picks=['CP4'])
    low.plot_image(picks=['CP4'])

def psd():

    evokeds = [evoked_high, evoked_low]

    for e in evokeds:

        # Compute PSD using Welch's method
        f, Pxx = signal.welch(np.mean(e.data, axis=0), 250, scaling='spectrum')

        # Define frequency range of interest
        fmin = 15
        fmax = 30

        # Find peaks in the PSD within the frequency range of interest
        peaks, _ = find_peaks(Pxx[(f >= fmin) & (f <= fmax)], prominence=0.1)

        # Plot PSD with peaks marked
        plt.figure()
        plt.plot(f, Pxx)
        plt.plot(f[peaks], Pxx[peaks], "x")
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power Spectral Density (V^2/Hz)')
        plt.title('PSD with peaks in frequency range {}-{} Hz'.format(fmin, fmax))
        plt.show()
