"""
==========================================
Compute source power using DICS beamformer
==========================================

Compute a Dynamic Imaging of Coherent Sources (DICS) :footcite:`GrossEtAl2001`
filter from single-trial activity to estimate source power across a frequency
band. This example demonstrates how to source localize the event-related
synchronization (ERS) of beta band activity in the
:ref:`somato dataset <somato-dataset>`.
"""

# Author: Marijn van Vliet <w.m.vanvliet@gmail.com>
#         Roman Goj <roman.goj@gmail.com>
#         Denis Engemann <denis.engemann@gmail.com>
#         Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# License: BSD-3-Clause

import os.path as op

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mne.beamformer import apply_dics_tfr_epochs, make_dics
from mne.time_frequency import csd_tfr, tfr_morlet

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


reject_criteria = dict(eeg=200e-6)
flat_criteria = dict(eeg=1e-6)

fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'

_oct = '6'
fwd_dir = op.join(subjects_dir, subject, 'bem', f'{subject}-oct{_oct}-fwd.fif')
fwd = mne.read_forward_solution(fwd_dir)

stc_all = []

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

    # induced power
    epochs = epochs.subtract_evoked().crop(tmin=-1, tmax=1)

    # Use Morlet wavelets to compute sensor-level time-frequency (TFR)
    # decomposition for each epoch. We must pass ``output='complex'`` if we wish to
    # use this TFR later with a DICS beamformer. We also pass ``average=False`` to
    # compute the TFR for each individual epoch.

        
    freqs = np.logspace(np.log10(12), np.log10(30), 9)
    n_cycles = 3.0  # different number of cycle per frequency

    epochs_tfr = mne.time_frequency.tfr_morlet(
                epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False,
                output="complex", average=False)

    # Compute the Cross-Spectral Density (CSD) matrix for the sensor-level TFRs.
    # We are interested in increases in power relative to the baseline period, so
    # we will make a separate CSD for just that period as well.
    csd = csd_tfr(epochs_tfr, tmin=-0.5, tmax=0.5)

    # Computing DICS spatial filters using the CSD that was computed on the baseline
    # interval.
    filters = make_dics(
        epochs.info,
        fwd,
        csd,
        pick_ori="max-power",
        reduce_rank=True,
        real_filter=True, label=pcc_label[0]
    )

    del csd
# project the TFR for each epoch to source space
    epochs_stcs = apply_dics_tfr_epochs(epochs_tfr, filters, return_generator=True)

    del epochs_tfr

    stc_all.append(epochs_stcs)

# average over all participants and plot the grand source power

beta_source_rh = mne.read_source_estimate("D:\expecon_ms\data\eeg\source\\beta_source-rh.stc")

# get center of mass of beta source
com = beta_source_rh.center_of_mass(subject, hemi=1)

# define label based on center of mass
pcc_label = mne.grow_labels(subject, extents=50, seeds=111449, hemis='rh')


data = np.zeros((501, epochs.times.size))

# loop over generator object (loop over epochs)
for epoch_stcs in epochs_stcs:
    for stc in epoch_stcs:
        data += (stc.data * np.conj(stc.data)).real

stc.data = data / len(epochs) / len(freqs)

alpha_source = np.array([s.data for s in stc_all])

alpha_gra = np.mean(alpha_source, axis=0)

alpha_stc = mne.SourceEstimate(alpha_gra, 
                               tmin=-0.5, 
                               tstep=.004, 
                               vertices=stc_all[0].vertices)

# extract for each participant the center of mass
com_all = []
for s in stc_all:
    com = s.center_of_mass(hemi=1) # 1 = right hemisphere
    com_all.append(com)

# save label for each participant
all_labels = []
for c in com_all:
    label = mne.label.select_sources('fsaverage', label='rh', 
                                     location=c[0])
    all_labels.append(label)

# now extract the power value in this center of mass for each participant
# and each epoch and save it
    
message = "DICS source power in the 12-30 Hz frequency band"
brain = alpha_stc.plot(
    hemi="rh",
    views="medial",
    subjects_dir=subjects_dir,
    subject=subject,
    time_label=message,
    background='white'
)
