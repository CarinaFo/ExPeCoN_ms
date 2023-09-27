# script contains function to calculate and plot source induced power


# Author: Carina Forster
# email: forster@cbs.mpg.de

# load packages
import os.path as op
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import source_band_induced_power

# Specify the file path for which you want the last commit date
file_path = "D:\expecon_ms\\analysis_code\\eeg\\source\\source_reco.py"

last_commit_date = subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", file_path]).decode("utf-8").strip()

print("Last Commit Date for", file_path, ":", last_commit_date)
# load source space files
fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'

_oct = '6'

fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')
src_fname = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')

# Read the source space and the forward solution
src = mne.read_source_spaces(src_fname)
fwd = mne.read_forward_solution(fwd_dir)

# datapaths
dir_cleanepochs = Path('D:/expecon_ms/data/eeg/prepro_ica/clean_epochs_corr')
behavpath = Path('D:/expecon_ms/data/behav/behav_df/')

# save paths for beamforming
beamformer_path = Path('D:/expecon_ms/data/eeg/source/high_low_pre_beamformer')
figs_dics_path = Path('D:/expecon_ms/figs/manuscript_figures/figure6_tfr_contrasts/source')
# save paths for mne
mne_path = Path('D:/expecon_ms/data/eeg/source/high_low_pre_eLORETA')
save_dir_cluster_output = Path('D:/expecon_ms/figs/eeg/sensor/cluster_permutation_output')

IDlist = ['007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034','035', '036',
          '037', '038', '039','040', '041', '042', '043', '044','045', '046', '047', '048', '049']
# Calculate source induced power

# set font to Arial and font size to 22
plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})


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


def extract_source_power(source_band=1):

    """Extract source power in a specific frequency band

    Returns
    -------
    stcs_a : list of SourceEstimate
        Source power in the high frequency band
    stcs_b: list of SourceEstimate
        Source power in the low frequency band
    """

    stc_a_list, stc_b_list = [], []

    for idx, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)

        # load cleaned epochs (after ica component rejection)
        epochs = mne.read_epochs(f'{dir_cleanepochs}'
                                    f'/P{subj}_epochs_after_ica_0.1Hzfilter-epo.fif')

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
            
        # remove trials with rts >= 2.5 (no response trials) and trials with rts < 0.1
        epochs = epochs[epochs.metadata.respt1 >= 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # load behavioral data
        data = pd.read_csv(f'{behavpath}//prepro_behav_data.csv')

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
        
        # drop bad epochs
        epochs.drop_bad(reject=dict(eeg=200e-6))

        # epochs for high and low probability condition
        epochs_a = epochs[epochs.metadata.cue == 0.75]
        epochs_b = epochs[epochs.metadata.cue == 0.25]

        if source_band:

            # Compute a source estimate per frequency band

            # create noise covariance with a bias of data length

            noise_cov = create_noise_cov(epochs_a.get_data()[0,:,:].shape, epochs_a.info)

            inv_op = mne.minimum_norm.make_inverse_operator(epochs_a.info, fwd, noise_cov)

            bands = dict(alpha=[9, 11], beta=[18, 22])

            # 5 cycles is default
            stcs_a = source_band_induced_power(
            epochs_a, inverse_operator=inv_op, label=labelS1, bands=bands, 
            use_fft=True, n_jobs=None,
            method='eLORETA'
            )

            stc_a_list.append(stcs_a)

            stcs_b = source_band_induced_power(
            epochs_b, inverse_operator=inv_op, label=labelS1, bands=bands, use_fft=True, n_jobs=None,
            method='eLORETA'
            )

            stc_b_list.append(stcs_b)

            for b, stc in stcs_a.items():
                stc.save(f"D:\expecon_ms\data\eeg\source\\{subj}_induced_power_{b}_high")
            for b, stc in stcs_b.items():
                stc.save(f"D:\expecon_ms\data\eeg\source\\{subj}_induced_power_{b}_low")

    return stc_a_list, stc_b_list

def load_and_plot_source_power_bands(tmin=-1, tmax=0):

    """This function loads the source estimates per participant and power band
    and plots the average across participants and frequency bands.
    
    Parameters
    ----------
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    
    Returns
    -------
    None
    """

    # define the bands
    bands=['alpha', 'beta']

    # first we load the source estimates per participant and power band
    high_all, low_all = [], []

    for subj in IDlist:

        for b in bands:

            high_all.append(mne.read_source_estimate(f"D:\expecon_ms\data\eeg\source\\{subj}_induced_power_{b}_high"))
            low_all.append(mne.read_source_estimate(f"D:\expecon_ms\data\eeg\source\\{subj}_induced_power_{b}_low"))

    # now we extract the bands and average across participants and source space
    alpha_hits_all, beta_hits_all = [],[]
    alpha_miss_all, beta_miss_all = [],[]


    alpha_hits_all = [h.crop(tmin, tmax).data for h in high_all[:43]]
    beta_hits_all = [h.crop(tmin, tmax).data for h in high_all[43:]]

    alpha_miss_all = [h.crop(tmin, tmax).data for h in low_all[:43]]
    beta_miss_all = [h.crop(tmin, tmax).data for h in low_all[43:]]


    alpha_hits = np.mean(np.array(alpha_hits_all), axis=(0,1))
    beta_hits = np.mean(np.array(beta_hits_all), axis=(0,1))

    alpha_miss = np.mean(np.array(alpha_miss_all), axis=(0,1))
    beta_miss = np.mean(np.array(beta_miss_all), axis=(0,1))

    # finally we plot the results
    plt.plot(np.linspace(tmin, tmax, alpha_hits_all[0].shape[1]), alpha_hits, label="alpha_high")
    plt.plot(np.linspace(tmin, tmax, alpha_hits_all[0].shape[1]), alpha_miss, label="alpha_low")
    plt.xlabel("Time (ms)")
    plt.ylabel("Power")
    plt.legend()
    plt.title("Mean source induced power")
    plt.show()

    plt.plot(np.linspace(tmin, tmax, alpha_hits_all[0].shape[1]), beta_hits, label="beta_high")
    plt.plot(np.linspace(tmin, tmax, alpha_hits_all[0].shape[1]), beta_miss, label="beta_low")
    plt.xlabel("Time (ms)")
    plt.ylabel("Power")
    plt.legend()
    plt.title("Mean source induced power")
    plt.show()



################################ Helper functions ################################


def create_noise_cov(data_size, data_info):
    """
    Computes identity noise covariance with a bias of data length
    Method is by Mina Jamshidi Idaji (minajamshidi91@gmail.com)
    :param tuple data_size: size of original data (dimensions - 1D)
    :param mne.Info data_info: info that corresponds to the original data
    :returns: (mne.Covariance) - noise covariance for further source reconstruction
    """

    data1 = np.random.normal(loc=0.0, scale=1.0, size=data_size)
    raw1 = mne.io.RawArray(data1, data_info)

    return mne.compute_raw_covariance(raw1, tmin=0, tmax=None)