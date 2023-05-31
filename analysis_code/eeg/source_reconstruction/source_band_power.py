# Calculate source induced power

import matplotlib.pyplot as plt
import pandas as pd
import os.path as op
import numpy as np
import mne
from mne import io
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator, source_band_induced_power

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 22, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# set paths
dir_cleanepochs = 'D:\\expecon_ms\\data\\eeg\prepro_ica\\clean_epochs'
behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'

# figure path
savedir_figs = 'D:\\expecon_ms\\figs\\manuscript_figures\\Figure3'

# participant index
IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026','027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')

# source stuff
inv_op = read_inverse_operator('D:\\expecon_ms\\data\\eeg\\source\\fsaverage-6oct-inv.fif')

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'

# this extracts a certain brain area only
label_S1 = 'rh.BA3a'
fname_labelS1 = subjects_dir + '\\fsaverage\\label\\%s.label' % label_S1
labelS1 = mne.read_label(fname_labelS1)
label_S2 = 'rh.BA3b'
fname_labelS2 = subjects_dir + '\\fsaverage\\label\\%s.label' % label_S2
labelS2 = mne.read_label(fname_labelS2)

label_aparc = 'rh.aparc'
fname_labelaparc = subjects_dir + '\\fsaverage\\label\\%s.label' % label_aparc
labelap = mne.read_label(fname_labelaparc)


def extract_source_power(source_band=0):

    """Extract source power in a specific frequency band

    Returns
    -------
    stcs_high : list of SourceEstimate
        Source power in the high frequency band
    stcs_low : list of SourceEstimate
        Source power in the low frequency band
    """

    reject_criteria=dict(eeg=200e-6)
    flat_criteria=dict(eeg=1e-6)

    freqs = np.arange(7, 30, 2)  # define frequencies of interest
    n_cycles = freqs / 3.0  # different number of cycle per frequency

    stcs_a_list, stcs_b_list = [], []

    for counter, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)
        # skip those participants
        if subj == '040' or subj == '045' or subj == '032' or subj == '016':
            continue

        # load cleaned epochs
        epochs = mne.read_epochs(f"{dir_cleanepochs}"
                                    f"/P{subj}_epochs_after_ica-epo.fif")

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

        # remove trials with rts >= 2.5 (no response trials) 
        # and trials with rts < 0.1
        epochs = epochs[epochs.metadata.respt1 > 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # remove first trial of each block (trigger delays)
        epochs = epochs[epochs.metadata.trial != 1]

        # subtract evoked response
        epochs = epochs.subtract_evoked()

        # load behavioral data
        data = pd.read_csv(f"{behavpath}//prepro_behav_data.csv")

        subj_data = data[data.ID == counter+7]

        if ((counter == 5) or (counter == 13) or
            (counter == 21) or (counter == 28)):  # first epoch has no data
            epochs.metadata = subj_data.iloc[1:, :]
        elif counter == 17:
            epochs.metadata = subj_data.iloc[3:, :]
        else:
            epochs.metadata = subj_data

        # drop bad epochs
        epochs.drop_bad(reject=reject_criteria, flat=flat_criteria)

        epochs_a = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1))]
        epochs_b = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 0))]

        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        #epochs_a = epochs[((epochs.metadata.cue == 0.75))]
        #epochs_b = epochs[((epochs.metadata.cue == 0.25))]

        if source_band:

            # Compute a source estimate per frequency band
            bands = dict(alpha=[9, 11], beta=[18, 22])

            # 5 cycles is default
            stcs_a = source_band_induced_power(
            epochs_a, inverse_operator=inv_op, label=labelS1, bands=bands, 
            use_fft=True, n_jobs=None,
            method='eLORETA'
            )

            stcs_a_list.append(stcs_a)

            stcs_b = source_band_induced_power(
            epochs_b, inverse_operator=inv_op, label=labelS1, bands=bands, use_fft=True, n_jobs=None,
            method='eLORETA'
            )

            stcs_b_list.append(stcs_b)

            
            for b, stc in stcs_a.items():
                stc.save(f"D:\expecon_ms\data\eeg\source\\induced_power_bands_s1\\{subj}_induced_power_{b}_hit")
            for b, stc in stcs_b.items():
                stc.save(f"D:\expecon_ms\data\eeg\source\\induced_power_bands_s1\\{subj}_induced_power_{b}_miss")

        else:
            
            # compute the source space power and the inter-trial coherence
            power_a, itc = mne.minimum_norm.source_induced_power(
            epochs_a,
            inverse_operator=inv_op,
            freqs=freqs,
            label=labelS1,
            n_cycles=n_cycles,
            n_jobs=None,
            method='eLORETA')

            stcs_a_list.append(power_a)

            # compute the source space power and the inter-trial coherence
            power_b, itc = mne.minimum_norm.source_induced_power(
            epochs_b,
            inverse_operator=inv_op,
            freqs=freqs,
            label=labelS1,
            n_cycles=n_cycles,
            n_jobs=None,
            method='eLORETA')

            stcs_b_list.append(power_b)

            # save the source estimates
            np.save(f"D:\expecon_ms\data\eeg\source\\induced_power_s1\\{subj}_induced_power_hit", power_a)
            np.save(f"D:\expecon_ms\data\eeg\source\\induced_power_s1\\{subj}_induced_power_miss", power_b)

    return stcs_a_list, stcs_b_list

def load_and_plot_source_power_bands():

    """This function loads the source estimates per participant and power band
    and plots the average across participants and frequency bands.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    None
    """

    # define the bands
    bands=['alpha', 'beta']

    # first we load the source estimates per participant and power band
    hits_all, miss_all = [], []

    for counter, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)
        # skip those participants
        if subj == '040' or subj == '045' or subj == '032' or subj == '016':
            continue
        
        for b in bands:
            hits_all.append(mne.read_source_estimate(f"D:\expecon_ms\data\eeg\source\\induced_power_bands_s1\\{subj}_induced_power_{b}_hit"))
            miss_all.append(mne.read_source_estimate(f"D:\expecon_ms\data\eeg\source\\induced_power_bands_s1\\{subj}_induced_power_{b}_miss"))

    # now we extract the bands and average across participants and source space
    alpha_hits_all, beta_hits_all = [],[]
    alpha_miss_all, beta_miss_all = [],[]


    alpha_hits_all = [h.crop(-0.5,0.3).data for h in hits_all[:39]]
    beta_hits_all = [h.crop(-0.5,0.3).data for h in hits_all[39:]]

    alpha_miss_all = [h.crop(-0.5,0.3).data for h in miss_all[:39]]
    beta_miss_all = [h.crop(-0.5,0.3).data for h in miss_all[39:]]


    alpha_hits = np.mean(np.array(alpha_hits_all), axis=(0,1))
    beta_hits = np.mean(np.array(beta_hits_all), axis=(0,1))

    alpha_miss = np.mean(np.array(alpha_miss_all), axis=(0,1))
    beta_miss = np.mean(np.array(beta_miss_all), axis=(0,1))

    # finally we plot the results
    plt.plot(np.linspace(-0.5,0.3,201), alpha_hits, label="alpha_hit")
    plt.plot(np.linspace(-0.5,0.3,201), alpha_miss, label="alpha_miss")
    plt.xlabel("Time (ms)")
    plt.ylabel("Power")
    plt.legend()
    plt.title("Mean source induced power")
    plt.show()

    plt.plot(np.linspace(-0.5,0.3,201), beta_hits, label="beta_hit")
    plt.plot(np.linspace(-0.5,0.3,201), beta_miss, label="beta_miss")
    plt.xlabel("Time (ms)")
    plt.ylabel("Power")
    plt.legend()
    plt.title("Mean source induced power")
    plt.show()

