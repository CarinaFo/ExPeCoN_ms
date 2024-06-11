#!/usr/bin/python3
"""
The script contains functions that source reconstruct 62 channel EEG data.

It is based on MNE methods (e.g., eLORETA) or beamforming for time-frequency.

Moreover, the script includes functions for statistical analysis in source space:
    permutation t-test or cluster permutation test in source space

Also, it includes a function to plot contrasts in source space.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2024
"""

# %% Import
from __future__ import annotations

import random
import subprocess
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets import fetch_fsaverage
import time

from expecon_ms.configs import PROJECT_ROOT, config, params, paths
from expecon_ms.utils import zero_pad_or_mirror_data

# turn off warnings for a cleaner output
import warnings

warnings.filterwarnings("ignore")
# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/source/source_reco.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)

print("Last Commit Date for", __file__path, ":", last_commit_date)

# where to store source files for the forward solution
Path(paths.data.templates).mkdir(parents=True, exist_ok=True)

# save paths for beamforming
Path(paths.data.eeg.source.beamformer).mkdir(parents=True, exist_ok=True)

# save paths for mne
Path(paths.data.eeg.source.mne).mkdir(parents=True, exist_ok=True)

# participant IDs
participants = config.participants

# fetch fsaverage files and the save path
subjects_dir = fetch_fsaverage()

# Load labels for S2 and posterior parietal cortex

label_parietal = mne.read_labels_from_annot("fsaverage", 'aparc', hemi='rh', subjects_dir=subjects_dir, regexp='superiorparietal')
label_insula = mne.read_labels_from_annot("fsaverage", 'aparc', hemi='rh', subjects_dir=subjects_dir, regexp='insula')
label_postcentral = mne.read_labels_from_annot("fsaverage", 'aparc', hemi='rh', subjects_dir=subjects_dir, regexp='postcentral')
label_suptemp = mne.read_labels_from_annot("fsaverage", 'aparc', hemi='rh', subjects_dir=subjects_dir, regexp='superiortemporal')
label_ofc = mne.read_labels_from_annot("fsaverage", 'aparc', hemi='rh', subjects_dir=subjects_dir, regexp='orbitofrontal')
label_supm = mne.read_labels_from_annot("fsaverage", 'aparc', hemi='rh', subjects_dir=subjects_dir, regexp='supramarginal')

label_ofc = label_ofc[0] + label_ofc[1]

label_S1 = 'rh.BA3a'
fname_labelS1 = subjects_dir + '\\fsaverage\\label\\%s.label' % label_S1
label_S1 = mne.read_label(fname_labelS1)


label_S2 = 'rh.BA3b'
fname_labelS2 = subjects_dir + '\\fsaverage\\label\\%s.label' % label_S2
label_S2 = mne.read_label(fname_labelS2)

# load functional labels for volatile env.
func_labels_prob2 = mne.read_label(Path(paths.data.templates, f"func_label_probability_2_-700-rh.label"), color='magenta')
func_labels_prevresp2 = mne.read_label(Path(paths.data.templates, f"func_label_prev_resp_2_-700-rh.label"), color='cyan')

# load functional labels for stable env.
func_labels_prob1= mne.read_label(Path(paths.data.templates, f"func_label_probability_1_-700-rh.label"), color='magenta')
func_labels_prevresp1 = mne.read_label(Path(paths.data.templates, f"func_label_prev_resp_1_-700-rh.label"), color='cyan')

# which areas are part of the label?
mne.vertex_to_mni(func_labels_prob1.get_vertices_used(), hemis=1, subject='fsaverage')
func_labels_prob2.compute_area('fsaverage', subjects_dir)
func_labels_prevresp2.compute_area('fsaverage', subjects_dir)
# data_cleaning parameters defined in config.toml
# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def make_new_forward_solution(setup_source_space: bool):
    """
    Create a new forward solution for source reconstruction.

    Args:
    ----
    setup_source_space: bool, info: if True, create new source space.

    Returns:
    -------
    .fif file containing the forward solution

    """
    # fetch fsaverage files and the save path
    subjects_dir = fetch_fsaverage()

    # set the root path to fsaverage files
    fs_average_root_path = Path(subjects_dir, "bem")

    # you can create your own source space with the desired spacing
    if setup_source_space:
        src = mne.setup_source_space("fsaverage", spacing="oct6", add_dist="patch", subjects_dir=subjects_dir)
        mne.write_source_spaces(fs_average_root_path / "fsaverage-oct6-src.fif", src)

    # load bem solution, source space and transformation matrix
    bem = fs_average_root_path / "fsaverage-5120-5120-5120-bem-sol.fif"
    src_fname = fs_average_root_path / "fsaverage-oct6-src.fif"  # 5mm between sources
    trans_dir = fs_average_root_path / "fsaverage-trans.fif"

    # Read the source space
    src = mne.read_source_spaces(src_fname)

    # load example epoch
    epochs = mne.read_epochs(Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon1, "P015_icacorr_0.1Hz-epo.fif"))

    # set up the forward solution
    fwd = mne.make_forward_solution(epochs.info, trans=trans_dir, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=None)

    mne.write_forward_solution(Path(paths.data.templates, "5120-fwd.fif"), fwd)


def run_source_reco(
    study: int,
    cond: str,
    mirror: bool,
    dics: int,
    fmin: int,
    fmax: int,
    tmin: int,
    tmax: int,
    drop_bads: bool,
    subtract_evoked: bool,
    plot_alignment: bool,
) -> None:
    """
    Run source reconstruction on epoched EEG data.

    Methods supported: eLoreta or DICS beamforming for oscillatory source analysis.

    Args:
    ----
    study : int, info: which study to analyze: 1 (block, stable environment) or 2 (trial,
    volatile environment)
    cond : str, info: which condition to analyze: "probability" or "prev_resp"
    mirror: bool, info: if True, avoid leakage and edge artifacts by zero padding or mirroring the data
    fmin: int, info: lower frequency bound for DICS beamforming
    fmax: int, info: upper frequency bound for DICS beamforming
    tmin: int, info: lower time bound for DICS beamforming
    tmax: int, info: upper time bound for DICS beamforming
    dics: 1 for DICS beamforming, 0 for eLoreta
    drop_bads: if True, bad epochs are dropped
    subtract_evoked: if True, subtract evoked response from each epoch
    plot_alignment: if True, plot alignment of electrodes with source space

    Returns:
    -------
    .stc files for each hemisphere containing source reconstructions for each participant, shape: vertices-x-timepoints

    """
    # set the save path for source estimates
    save_path = Path(paths.data.eeg.source.beamformer) if dics == 1 else Path(paths.data.eeg.source.mne)

    # read the forward solution
    fwd = mne.read_forward_solution(Path(paths.data.templates, "5120-fwd.fif"))

    if study == 1:
        id_list = participants.ID_list_expecon1
        # load behavioral data
        data = pd.read_csv(Path(paths.data.behavior, "prepro_behav_data_1.csv"))

    elif study == 2:  # noqa: PLR2004
        id_list = participants.ID_list_expecon2
        # load behavioral data
        data = pd.read_csv(Path(paths.data.behavior, "prepro_behav_data_2.csv"))
    else:
        raise ValueError("input should be 1 or 2 for the respective study")

    # now loop over participants
    for idx, subj in enumerate(id_list):
        # print participant ID
        print("Analyzing " + subj)

        # plot alignment of electrodes with source space for one participant
        if (idx == 0) & (plot_alignment):
            plot_source_space_electrodes_alignment()

        if cond == "probability":
            # set condition names
            cond_a_name = f"high_prevhit_{tmin}_{tmax}"
            cond_b_name = f"low_prevhit_{tmin}_{tmax}"
            # add mirror to filename if data is mirrored
            if mirror:
                cond_a_name = f"{cond_a_name}_mirror"
                cond_b_name = f"{cond_b_name}_mirror"
        elif cond == "prev_resp":
            # set condition names
            cond_a_name = f"prevyesresp_stimprevcurrent_{tmin}_{tmax}"
            cond_b_name = f"prevnoresp_stimprevcurrent_{tmin}_{tmax}"
            # add mirror to filename if data is mirrored
            if mirror:
                cond_a_name = f"{cond_a_name}_mirror"
                cond_b_name = f"{cond_b_name}_mirror"
        elif cond == "control":
            # set condition names
            cond_a_name = "stimulus"
            cond_b_name = "noise"
            
        source_files = Path(save_path, f"contrast_{cond}_{subj}_{study}-lh.stc")

        if source_files.exists():
            print(f"Skipping {subj} because it already exists.")
            continue

        print(f"Processing {subj}.")
        # load clean epochs (after ica component rejection)
        if study == 1:
            epochs = mne.read_epochs(
                Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon1, f"P{subj}_icacorr_0.1Hz-epo.fif")
            )
        elif study == 2:  # noqa: PLR2004
            # skip ID 13
            if subj == "013":
                continue
            epochs = mne.read_epochs(
                Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon2, f"P{subj}_icacorr_cnv_0.1Hz-epo.fif")
            )
            # rename columns
            epochs.metadata = epochs.metadata.rename(
                columns={"resp1_t": "respt1", "stim_type": "isyes", "resp1": "sayyes"}
            )
        else:
            raise ValueError("input should be 1 or 2 for the respective study")

        # clean epochs (remove blocks based on hit and false alarm rates, reaction times, etc.)
        epochs = drop_trials(data=epochs)

        # get behavioral data for current participant
        if study == 1:
            subj_data = data[idx + config.participants.pilot_counter == data.ID]
        elif study == 2:  # noqa: PLR2004
            subj_data = data[idx + 1 == data.ID]
        else:
            raise ValueError("input should be 1 or 2 for the respective study")

        # get drop log from epochs
        drop_log = epochs.drop_log

        # Ignored bad epochs are those defined by the user as bad epochs
        search_string = "IGNORED"
        # remove trials where epochs are labeled as too short
        indices = [index for index, tpl in enumerate(drop_log) if tpl and search_string not in tpl]

        # drop trials without a corresponding epoch
        if indices:
            epochs.metadata = subj_data.reset_index().drop(indices)
        else:
            epochs.metadata = subj_data

        if drop_bads:
            # drop epochs with abnormal strong signal (> 200 micro-volts)
            epochs.drop_bad(reject=dict(eeg=200e-6))

        if subtract_evoked:
            # subtract evoked response from each epoch
            epochs = epochs.subtract_evoked()

        # avoid leakage and edge artifacts by zero-padding or mirroring the data
        # on both ends
        if mirror:
            metadata = epochs.metadata

            epoch_data = epochs.crop(tmin, tmax).get_data()

            # zero pad = False = mirror the data on both ends
            data_mirror = zero_pad_or_mirror_data(epoch_data, zero_pad=False)

            # put back into the epoch structure
            epochs = mne.EpochsArray(data_mirror, epochs.info, tmin=tmin * 2)

            # add metadata back
            epochs.metadata = metadata

        if cond == "probability":
            epochs_a = epochs[((epochs.metadata.cue == params.high_p) & (epochs.metadata.previsyes == 1) & (epochs.metadata.prevresp == 1))]
            epochs_b = epochs[((epochs.metadata.cue == params.low_p) & (epochs.metadata.previsyes == 1) & (epochs.metadata.prevresp == 1))]
            if mirror:
                cond_a_name = f"{cond_a_name}_mirror"
                cond_b_name = f"{cond_b_name}_mirror"

        elif cond == "prev_resp":
            if study == 1:
                epochs_a = epochs[
                    ((epochs.metadata.prevresp == 1) & (epochs.metadata.previsyes == 1) & 
                      (epochs.metadata.cue == params.high_p))
                ]
                epochs_b = epochs[
                    ((epochs.metadata.prevresp == 0) & (epochs.metadata.previsyes == 1) &
                     (epochs.metadata.cue == params.high_p))
                ]
            else:
                epochs_a = epochs[
                    (((epochs.metadata.prevresp == 1) & (epochs.metadata.prevcue == epochs.metadata.cue) &
                     (epochs.metadata.cue == params.high_p)))
                ]

                epochs_b = epochs[
                    (((epochs.metadata.prevresp == 0) & (epochs.metadata.prevcue == epochs.metadata.cue)  & 
                     (epochs.metadata.cue == params.high_p)))
                ]
            if mirror:
                cond_a_name = f"{cond_a_name}_mirror"
                cond_b_name = f"{cond_b_name}_mirror"
        elif cond == "control":
            epochs_a = epochs[(epochs.metadata.isyes == 1)]
            epochs_b = epochs[(epochs.metadata.isyes == 0)]
        else:
            raise ValueError("input should be 'probability' or 'prev_resp'")

        # make sure we have an equal number of trials in both conditions
        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        if dics == 1:
            # define the frequency band of interest
            freqs = np.arange(fmin, fmax, 1)

            # set the cycles for the Morlet wavelet
            n_cycles = freqs / 4.0

            # Computing the cross-spectral density matrix for the beta frequency band, for
            # different time intervals.
            # csd for all epochs
            csd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=tmin, tmax=tmax, n_cycles=n_cycles)
            # csd for high-prob trials only
            csd_a = mne.time_frequency.csd_morlet(epochs_a, freqs, tmin=tmin, tmax=tmax, n_cycles=n_cycles)
            # csd for low-prob trials only
            csd_b = mne.time_frequency.csd_morlet(epochs_b, freqs, tmin=tmin, tmax=tmax, n_cycles=n_cycles)

            info = epochs.info

            # To compute the source power for a frequency band, rather than each frequency
            # separately, we average the CSD objects across frequencies.
            csd = csd.mean()
            csd_a = csd_a.mean()
            csd_b = csd_b.mean()

            # Computing DICS spatial filters using the CSD that was computed for all epochs
            filters = mne.beamformer.make_dics(
                info, fwd, csd, noise_csd=None, pick_ori="max-power", reduce_rank=True, real_filter=True
            )

            # Applying DICS spatial filters separately to each condition
            source_power_a, freqs = mne.beamformer.apply_dics_csd(csd_a, filters)
            source_power_b, freqs = mne.beamformer.apply_dics_csd(csd_b, filters)

            source_power_a.save(Path(save_path, f"{cond_a_name}_{subj}_{study}"))
            source_power_b.save(Path(save_path, f"{cond_b_name}_{subj}_{study}"))

        else:
            # average epochs for MNE
            evokeds_a = epochs_a.average()
            evokeds_b = epochs_b.average()

            evoked_contrast = mne.combine_evoked(all_evoked=[evokeds_a, evokeds_b], weights=[0.5, -0.5])
            evoked_contrast.crop(tmin, tmax)
            # filter in the beta band
            evoked_contrast.filter(l_freq=fmin, h_freq=fmax)

            # create noise covariance with a bias of data length
            # noise_cov = create_noise_cov(evokeds_high.data.shape, evokeds_high.info) # noqa: ERA001

            # all epochs for noise covariance computation
            noise_cov = mne.compute_covariance(
                epochs, tmin=tmin, tmax=tmax, method=["shrunk", "empirical"], rank="info"
            )

            # save covariance matrix
            mne.write_cov(fname="covariance_prestim.cov", cov=noise_cov, overwrite=True)

            # fixed forward solution for MNE methods
            fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True)

            info = evoked_contrast.info

            inv_op = mne.minimum_norm.make_inverse_operator(info, fwd_fixed, noise_cov, loose=0.2, depth=0.8)

            evoked_contrast.set_eeg_reference(projection=True)  # needed for inverse modeling

            method = "dSPM"
            snr = 3.0
            lambda2 = 1.0 / snr**2  # regularization

            stc = mne.minimum_norm.apply_inverse(evoked_contrast, inv_op, lambda2, method=method, pick_ori=None)

            stc.save(Path(save_path, f"contrast_{cond}_{subj}_{study}"))


def run_source_reco_per_trial(
    study: int,
    fmin: int,
    fmax: int,
    tmin: int,
    tmax: int,
    drop_bads: bool
) -> None:
    """
    Run source reconstruction on epoched EEG data.

    Methods supported: DICS beamforming for oscillatory source analysis.

    Args:
    ----
    study : int, info: which study to analyze: 1 (block, stable environment) or 2 (trial,
    volatile environment)
    fmin: int, info: lower frequency bound for DICS beamforming
    fmax: int, info: upper frequency bound for DICS beamforming
    tmin: int, info: lower time bound for DICS beamforming
    tmax: int, info: upper time bound for DICS beamforming
    drop_bads: if True, bad epochs are dropped

    Returns:
    -------
    .stc files for each hemidfsphere containing source reconstructions for each participant, shape: vertices-x-timepoints

    """

    # read the forward solution
    fwd = mne.read_forward_solution(Path(paths.data.templates, "5120-fwd.fif"))

    if study == 1:
        id_list = participants.ID_list_expecon1
        # load behavioral data
        data = pd.read_csv(Path(paths.data.behavior, "prepro_behav_data_1.csv"))
        label_prob = func_labels_prob1
        label_prev = func_labels_prevresp1
    elif study == 2:  # noqa: PLR2004
        id_list = participants.ID_list_expecon2
        # load behavioral data
        data = pd.read_csv(Path(paths.data.behavior, "prepro_behav_data_2.csv"))
        label_prob = func_labels_prob2
        label_prev = func_labels_prevresp2
    else:
        raise ValueError("input should be 1 or 2 for the respective study")

    source_power_allsubs_prob, source_power_allsubs_prevresp = [], []
    metadata_all = []

    # now loop over participants
    for idx, subj in enumerate(id_list):

        print(f"Processing {subj}.")

        # Start time
        start_time = time.time()

        # load clean epochs (after ica component rejection)
        if study == 1:
            epochs = mne.read_epochs(
                Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon1, f"P{subj}_icacorr_0.1Hz-epo.fif")
            )
        elif study == 2:  # noqa: PLR2004
            # skip ID 13
            if subj == "013":
                continue
            epochs = mne.read_epochs(
                Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon2, f"P{subj}_icacorr_0.1Hz-epo.fif")
            )
            # rename columns
            epochs.metadata = epochs.metadata.rename(
                columns={"resp1_t": "respt1", "stim_type": "isyes", "resp1": "sayyes"}
            )
        else:
            raise ValueError("input should be 1 or 2 for the respective study")

        # clean epochs (remove blocks based on hit and false alarm rates, reaction times, etc.)
        epochs = drop_trials(data=epochs)

        # get behavioral data for current participant
        if study == 1:
            subj_data = data[idx + config.participants.pilot_counter == data.ID]
        elif study == 2:  # noqa: PLR2004
            subj_data = data[idx + 1 == data.ID]
        else:
            raise ValueError("input should be 1 or 2 for the respective study")

        # get drop log from epochs
        drop_log = epochs.drop_log

        # Ignored bad epochs are those defined by the user as bad epochs
        search_string = "IGNORED"
        # remove trials where epochs are labeled as too short
        indices = [index for index, tpl in enumerate(drop_log) if tpl and search_string not in tpl]

        # drop trials without a corresponding epoch
        if indices:
            epochs.metadata = subj_data.reset_index().drop(indices)
        else:
            epochs.metadata = subj_data

        if drop_bads:
            # drop epochs with abnormal strong signal (> 200 micro-volts)
            epochs.drop_bad(reject=dict(eeg=200e-6))

        metadata_all.append(epochs.metadata)

        # define the frequency band of interest
        freqs = np.arange(fmin, fmax, 1)

        # set the cycles for the Morlet wavelet
        ncycles = freqs / 4.0

        # Compute CSD for all epochs
        csd_allepochs = mne.time_frequency.csd_morlet(epochs, freqs, tmin=tmin, tmax=tmax, n_cycles=ncycles,
                                            verbose=None)

        info = epochs.info

        # average broadband CSD across frequencies
        csd_allepochs = csd_allepochs.mean()

        # Computing DICS spatial filters using the CSD that was computed for all epochs
        filter_prevresp = mne.beamformer.make_dics(
            info, fwd, csd_allepochs, noise_csd=None, pick_ori="max-power", reduce_rank=True, real_filter=True,
            label=label_prev, verbose=None
        )

        # Computing DICS spatial filters using the CSD that was computed for all epochs
        filter_prob = mne.beamformer.make_dics(
            info, fwd, csd_allepochs, noise_csd=None, pick_ori="max-power", reduce_rank=True, real_filter=True,
            label=label_prob, verbose=None
        )

        source_power_alltrials_prob, source_power_alltrials_prevresp = [], []

        for e_idx in range(len(epochs)):

            # Compute CSD for each epoch
            csd = mne.time_frequency.csd_morlet(epochs[e_idx][0], freqs, tmin=tmin, 
                                                tmax=tmax, n_cycles=ncycles,
                                                verbose=None)
            csd = csd.mean()

            # Applying DICS spatial filters separately to each ROI
            source_power_prob, _ = mne.beamformer.apply_dics_csd(csd, filter_prob, verbose=None)
            source_power_prevresp, _ = mne.beamformer.apply_dics_csd(csd, filter_prevresp, verbose=None)

            # average over voxels
            source_data_prob = np.mean(np.array(source_power_prob.data))
            source_data_prevresp = np.mean(np.array(source_power_prevresp.data))

            source_power_alltrials_prob.append(source_data_prob)
            source_power_alltrials_prevresp.append(source_data_prevresp)

        source_power_allsubs_prob.append(source_power_alltrials_prob)
        source_power_allsubs_prevresp.append(source_power_alltrials_prevresp)

    data = pd.concat(metadata_all)

    # Flatten the nested list using list comprehension
    flat_list_prob = [item for sublist in source_power_allsubs_prob for item in sublist]
    flat_list_prev = [item for sublist in source_power_allsubs_prevresp for item in sublist]

    data['beta_source_prob'] = flat_list_prob
    data['beta_source_prev'] = flat_list_prev

    data.to_csv(Path(paths.data.behavior, f"brain_behav_source_-700-100_{study}.csv"))

    return source_power_allsubs_prob, source_power_allsubs_prevresp


def create_source_contrast_array(study: int, cond_a: str, cond_b: str, method: str):
    """
    Load source estimates per participant.

    Contrasts are stored in numpy arrays.

    Args:
    ----
    study: int, info: which study to analyze: 1 (block, stable environment) or 2 (trial)
    cond: str, info: which condition to analyze: "probability" or "prev_resp"
    cond_a: str, info: name of condition a
    cond_b: str, info: name of condition b
    method: str, info: which method to analyze: "beamformer" or "mne"

    Returns:
    -------
    shape of the numpy array: participants-x-vertices-x-timepoints

    """
    # load id list for the respective study
    if study == 1:
        id_list = participants.ID_list_expecon1
    elif study == 2:  # noqa: PLR2004
        id_list = participants.ID_list_expecon2
    else:
        raise ValueError("input should be 1 or 2 for the respective study")

    # set the path to the source files
    path_to_source = (
        Path(paths.data.eeg.source.beamformer) if method == "beamformer" else Path(paths.data.eeg.source.mne)
    )

    stc_all = []
    # loop over participants
    for subj in id_list:
        if (study == 2) and (subj == "013"):  # noqa: PLR2004
            continue
        # load source estimates
        stc_high = mne.read_source_estimate(path_to_source / f"{cond_a}_{subj}_{study}")
        stc_low = mne.read_source_estimate(path_to_source / f"{cond_b}_{subj}_{study}")
        # compute difference between conditions
        stc_diff = stc_high.data - stc_low.data
        # append to list
        stc_all.append(stc_diff)

    # convert the list to a numpy array
    return np.array(stc_all)


def plot_grand_average_source_contrast(study: int, cond: str, method: str, save_plots: bool):
    """
    Run a cluster-based permutation test over space and time.

    Args:
    ----
    study: int, info: which study to analyze: 1 (block, stable environment) or 2 (trial)
    cond: str, info: which condition to analyze: "probability" or "prev_resp"
    method: str, info: which method to analyze: "beamformer" or "mne"
    save_plots: bool, info: if True, save plots

    Returns:
    -------
    cluster output

    """
    if (study == 1) & (cond == "probability"):
        stc_array_hit = create_source_contrast_array(
            study=study, cond_a="high_prevhit_-0.7_-0.1", cond_b="low_prevhit_-0.7_-0.1", method=method
        )
        stc_array_miss = create_source_contrast_array(
            study=study, cond_a="high_prevmiss_-0.7_-0.1", cond_b="low_prevmiss_-0.7_-0.1", method=method
        )
        stc_array_cr = create_source_contrast_array(
            study=study, cond_a="high_prevcr_-0.7_-0.1", cond_b="low_prevcr_-0.7_-0.1", method=method
        )
        stc_array_conds = np.array([stc_array_hit, stc_array_miss, stc_array_cr])
        # mean over conditions (previous hit, previous miss, previous cr)
        stc_array = np.mean(stc_array_conds, axis=0)
    elif (study == 2) & (cond == "probability"):  # noqa: PLR2004
        stc_array = create_source_contrast_array(study=study, cond_a="high", cond_b="low", method=method)

    elif cond == "prev_resp":
        if study == 1:
            stc_array = create_source_contrast_array(
                study=study,
                cond_a="prevyesresp_stimprevcurrent_-0.7_-0.1",
                cond_b="prevnoresp_stimprevcurrent_-0.7_-0.1",
                method=method,
            )
        elif study == 2:  # noqa: PLR2004
                      stc_array = create_source_contrast_array(
                study=study,
                cond_a="prevyesresp",
                cond_b="prevnoresp",
                method=method,
            )
    elif cond == "control":
        stc_array = create_source_contrast_array(
            study=study,
            cond_a="stimulus",
            cond_b="noise",
            method=method,
        )

    # get rid of zero-dimension
    stc_array = np.squeeze(stc_array)

    # permutation test to get sign. vertices
    t_obs, p, _ = mne.stats.permutation_t_test(stc_array)  # _, p, _ = t, p, h
    print(f"% of significant vertices: {np.sum(p < params.alpha) / len(p)}")
    print(f"Number of significant vertices: {np.sum(p < params.alpha)}")

    # mean over participants
    x_avg = np.mean(stc_array, axis=0)

    # Read the source space for plotting
    src_fname = Path(paths.data.templates, "fsaverage-6oct-src.fif")
    src = mne.read_source_spaces(src_fname)
    # fetch fsaverage files and the save path
    subjects_dir = fetch_fsaverage()

    # put contrast (average) or p values in source space
    fsave_vertices = [s["vertno"] for s in src]
    stc = mne.SourceEstimate(t_obs, tmin=-0.7, tstep=0.0001, vertices=fsave_vertices, subject="fsaverage")

    right_hemi_data = stc.data[-len(stc.vertices[1]):]

    # Sort the array in ascending order
    sorted_stc = np.sort(right_hemi_data)

    # Determine the threshold index for the 10% most negative values
    threshold_index = int(0.1 * len(sorted_stc))

    # Find the threshold value
    threshold_value = sorted_stc[threshold_index]

    # Create a mask for values below the threshold
    mask = stc.data > threshold_value

    # Set values below the threshold to 0
    stc.data[mask] = 0

    # create functional labels
    func_labels = mne.stc_to_label(
        stc,
        src=src,
        smooth=True,
        subjects_dir=subjects_dir,
        connected=False,
        verbose="error",
    )
    
    mne.write_label(Path(paths.data.templates, f"func_label_{cond}_{study}_-700"), func_labels[1])

    print(min(t_obs), max(t_obs))

    views = ["lat"]	
    colorbar_conds = [True]

    # save source plot with colorbar and without for both hemispheres and views
    for view in views:
        for colbar in colorbar_conds:
            # plot average source or t values
            brain = stc.plot(
                surface="inflated",
                hemi='rh',
                views=view,
                colormap='coolwarm',
                subjects_dir=subjects_dir,
                time_viewer=True,
                backend="pyvistaqt",
                background="white",
                colorbar=colbar
                #clim=dict(kind="value", pos_lims = (-5, -4, -3))
            )
            #brain.add_annotation("aparc", borders=True, alpha=0.9)
            # If the label lives in the normal place in the subjects directory,
            # you can plot it by just using the name
            #brain.add_label("BA3a", borders=True, color="green", alpha=0.7)
            #brain.add_label("BA3b", borders=True, color="blue", alpha=0.7)
            brain.add_label(label_postcentral[0], borders=True, color="red", alpha=0.7)
            if save_plots:
                brain.savefig(
                    Path(
                        paths.figures.manuscript.figure4_source,
                        f"grand_average_{cond}_{method}_{study}_{view}_{hemi}_{colbar}.png",
                    )
                )

    return t_obs

def run_and_plot_cluster_test(study: int = 1, jobs: int = -1, n_perm: int = 10000):
    """
    Plot significant clusters of a spatio-temporal cluster permutation test.

    Args:
    ----
    study: int, info: which study to analyze
    jobs: int, info: how many jobs are to run in parallel
    n_perm: int, info: how many permutations to run

    Returns:
    -------
    plot of cluster output

    """
    stc_array = create_source_contrast_array(
        study=study, cond_a="high", cond_b="low", method="beamformer"
    )

    print("Computing adjacency.")

    # Read the source space for plotting
    src_fname = Path(paths.data.templates, "fsaverage-6oct-src.fif")
    src = mne.read_source_spaces(src_fname)

    # get adjacency matrix for source space
    adjacency = mne.spatial_src_adjacency(src)

    # Note that X needs to be a multidimensional array of shape
    # observations (subjects) × time × space, so we permute dimensions
    x = np.transpose(stc_array, [0, 2, 1])

    # spatio-temporal cluster permutation test
    clu = mne.stats.spatio_temporal_cluster_1samp_test(x, adjacency=adjacency, n_jobs=jobs, n_permutations=n_perm)

    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_idx = np.where(clu[2] < params.alpha)[0]

    # check if there are significant clusters to plot, otherwise break function
    if len(good_clusters_idx) == 0:
        return "No significant clusters."

    fsave_vertices = [s["vertno"] for s in src]

    # summarize cluster perm test output and prepare for visualization
    stc_all_cluster_vis = mne.stats.summarize_clusters_stc(
        clu, vertices=fsave_vertices, subject="fsaverage", p_thresh=params.alpha
    )

    # fetch fsaverage files and the save path
    subjects_dir = fetch_fsaverage()

 # Let's actually plot the first "time point" in the SourceEstimate, which
    # shows all the clusters, weighted by duration.
    # blue blobs are for condition A < condition B, red for A > B
    brain = stc_all_cluster_vis.plot(
        hemi="rh",
        views="lateral",
        subjects_dir=subjects_dir,
        subject = 'fsaverage',
        time_label="temporal extent (ms)",
        size=(800, 800),
        smoothing_steps=5,
        time_viewer=True,
        background="white",
        transparent=True,
        colorbar=True,
    )

    brain.save_image(Path(paths.figures.manuscript.figure4_source, "cluster_{cond}_{freq_band}_{method}.png"))
    return None


def create_noise_cov(data_size: tuple[int, int], data_info: mne.Info) -> mne.Covariance:
    """
    Compute identity noise covariance with a bias of data length.

    This method has been developed by Mina Jamshidi Idaji (minajamshidi91@gmail.com)

    Args:
    ----
    data_size: size of original data (dimensions - 1D)
    data_info: info that corresponds to the original data

    Returns:
    -------
    noise covariance for further source reconstruction

    """
    data1 = np.random.normal(loc=0.0, scale=1.0, size=data_size)
    raw1 = mne.io.RawArray(data1, data_info)
    return mne.compute_raw_covariance(raw1, tmin=0, tmax=None)


# Helper functions


def plot_source_space_electrodes_alignment():
    """Plot the alignment of the electrodes with the source space."""
    # select a random subject
    random_subj = random.choice(config.participants.ID_list_expecon2)  # noqa: S311

    # load cleaned epochs to extract info for plotting alignment
    epochs = mne.read_epochs(
        Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon2, f"P{random_subj}_icacorr_0.1Hz-epo.fif")
    )

    # fetch fsaverage files and the save path
    subjects_dir = fetch_fsaverage()

    # Read the source space for plotting
    src_fname = Path(paths.data.templates, "fsaverage-6oct-src.fif")
    src = mne.read_source_spaces(src_fname)

    # set the root path to fsaverage files
    fs_average_root_path = Path(subjects_dir, "bem")

    # set the root path to fsaverage files
    trans_dir = fs_average_root_path / "fsaverage-trans.fif"

    mne.viz.plot_alignment(
        epochs.info,
        trans_dir,
        subject="fsaverage",
        dig=False,
        src=src,
        subjects_dir=subjects_dir,
        verbose=True,
        meg=False,
        eeg=True,
    )


def drop_trials(data=None):
    """
    Drop trials based on behavioral data.

    Args:
    ----
    data: mne.Epochs, epoched data

    Returns:
    -------
    data: mne.Epochs, epoched data

    """
    # store number of trials before rt cleaning
    before_rt_cleaning = len(data.metadata)

    # remove no response trials or super fast responses
    data = data[data.metadata.respt1 != params.behavioral_cleaning.rt_max]
    data = data[data.metadata.respt1 > params.behavioral_cleaning.rt_min]

    # print rt trials dropped
    rt_trials_removed = before_rt_cleaning - len(data.metadata)

    print("Removed trials based on reaction time: ", rt_trials_removed)
    # Calculate hit rates per participant
    signal = data[data.metadata.isyes == 1]
    hit_rate_per_subject = signal.metadata.groupby(["ID"])["sayyes"].mean()

    print(f"Mean hit rate: {np.mean(hit_rate_per_subject):.2f}")

    # Calculate hit rates by participant and block
    hit_rate_per_block = signal.metadata.groupby(["ID", "block"])["sayyes"].mean()

    # remove blocks with hit-rates > 90 % or < 20 %
    filtered_groups = hit_rate_per_block[
        (hit_rate_per_block > params.behavioral_cleaning.hitrate_max)
        | (hit_rate_per_block < params.behavioral_cleaning.hitrate_min)
    ]
    print("Blocks with hit rates > 0.9 or < 0.2: ", len(filtered_groups))

    # Extract the ID and block information from the filtered groups
    remove_hit_rates = filtered_groups.reset_index()

    # Calculate false alarm rates by participant and block
    noise = data[data.metadata.isyes == 0]
    fa_rate_per_block = noise.metadata.groupby(["ID", "block"])["sayyes"].mean()

    # remove blocks with false alarm rates > 0.4
    filtered_groups = fa_rate_per_block[fa_rate_per_block > params.behavioral_cleaning.farate_max]
    print("Blocks with false alarm rates > 0.4: ", len(filtered_groups))

    # Extract the ID and block information from the filtered groups
    remove_fa_rates = filtered_groups.reset_index()

    # Hit-rate should be > the false alarm rate
    filtered_groups = hit_rate_per_block[
        hit_rate_per_block - fa_rate_per_block < params.behavioral_cleaning.hit_fa_diff
    ]
    print("Blocks with hit rates < false alarm rates: ", len(filtered_groups))

    # Extract the ID and block information from the filtered groups
    hit_vs_fa_rate = filtered_groups.reset_index()

    # Concatenate the dataframes
    combined_df = pd.concat([remove_hit_rates, remove_fa_rates, hit_vs_fa_rate])

    # Remove duplicate rows based on 'ID' and 'block' columns
    unique_df = combined_df.drop_duplicates(subset=["ID", "block"])

    # Merge the big dataframe with unique_df to retain only the non-matching rows
    data.metadata = data.metadata.merge(unique_df, on=["ID", "block"], how="left", indicator=True, suffixes=("", "_y"))

    data = data[data.metadata["_merge"] == "left_only"]

    # Remove the '_merge' column
    data.metadata = data.metadata.drop("_merge", axis=1)

    return data


# Unused functions


def extract_time_course_from_label(stc: np.ndarray, src: mne.SourceSpaces):
    """
    Extract the time course from a label in source space.

    Args:
    ----
    stc: np.ndarray, source estimates
    src: mne.SourceSpaces, source space

    Returns:
    -------
    time course for each label

    """
    # fetch fsaverage files and the save path
    subjects_dir = fetch_fsaverage()

    # this extracts a certain brain area
    label_s1 = "rh.BA3a"
    fname_labels1 = subjects_dir / f"fsaverage/label/{label_s1}.label"
    labels1 = mne.read_label(str(fname_labels1))
    label_s2 = "rh.BA3b"
    fname_labels2 = subjects_dir / f"fsaverage/label/{label_s2}.label"
    labels2 = mne.read_label(str(fname_labels2))
    label_aparc = "rh.aparc"
    fname_label_aparc = subjects_dir / f"fsaverage/label/{label_aparc}.label"
    label_ap = mne.read_label(str(fname_label_aparc))

    # Get labels for FreeSurfer 'aparc' cortical parcellation with 75 labels/hemi
    labels_parc = mne.read_labels_from_annot(
        "fsaverage",
        parc="aparc.a2009s",
        subjects_dir=subjects_dir,
    )

    # S1 label
    s1label = mne.extract_label_time_course(stc, labels1, src, allow_empty=True)
    # S2
    s2label = mne.extract_label_time_course(stc, labels2, src, allow_empty=True)
    # label ap
    ap_label = mne.extract_label_time_course(stc, label_ap, src, allow_empty=True)

    # extract activity in from source label
    # S1
    post_central_gyrus = mne.extract_label_time_course(stc, labels_parc[55], src, allow_empty=True)
    # S2
    g_front_inf_opercular_rh = mne.extract_label_time_course(stc, labels_parc[25], src, allow_empty=True)
    # ACC
    g_cingul_post_dorsal_rh = mne.extract_label_time_course(stc, labels_parc[19], src, allow_empty=True)

    return post_central_gyrus, g_front_inf_opercular_rh, g_cingul_post_dorsal_rh


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
