#!/usr/bin/python3
"""
Provide functions that compute time-frequency representations (TFRs).

Moreover, run a cluster test in 2D space (time and frequency)

This script produces figure 3 in Forster et al., 2025

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2024/2025
"""

# %% Import
from __future__ import annotations

import random
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import scipy

from expecon_ms.configs import PROJECT_ROOT, config, params, paths
from expecon_ms.utils import zero_pad_or_mirror_epochs, drop_trials

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# %matplotlib qt  # for plots in new windows

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/sensor/tfr_contrasts.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# Set font to Arial and font size to 14
plt.rcParams.update({
    "font.size": params.plot.font.size,
    "font.family": params.plot.font.family,
    "font.sans-serif": params.plot.font.sans_serif,
})

# Save tfr solutions
Path(paths.data.eeg.sensor.tfr.tfr_contrasts).mkdir(parents=True, exist_ok=True)

Path(paths.figures.manuscript.figure3).mkdir(parents=True, exist_ok=True)

# Participant IDs
participants = config.participants

# Data_cleaning parameters defined in config.toml

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def compute_tfr(
    study: int,
    cond: str,
    tmin: float,
    tmax: float,
    fmin: float,
    fmax: float,
    laplace: bool = False,
    induced: bool = True,
    mirror: bool = False,
    drop_bads: bool = True,
):
    """
    Calculate the time-frequency representations per trial (induced power).

    Power is calculated using multitaper method.
    Power is averaged over conditions after calculating power per trial.
    Data is saved in a tfr object per subject and stored to disk
    as a .h5 file.

    Args:
    ----
    study : int, info: which study to analyze: 1 (block, stable environment) or 2 (trial,
    volatile environment)
    cond : str, info: which condition to analyze: "probability" or "prev_resp"
    tmin : float: start time of the time window
    tmax : float: end time of the time window
    fmin: float:  start frequency of the frequency window
    fmax: float:  end frequency of the frequency window
    laplace: boolean, info: apply current source density transform
    induced : boolean, info: subtract evoked response from each epoch
    mirror : boolean, info: mirror the data on both sides to avoid edge artifacts
    drop_bads : boolean: drop epochs with abnormal strong signal (> 200 micro-volts)

    Returns:
    -------
    None

    """
    # Define frequencies and cycles for multitaper method
    freqs = np.arange(fmin, fmax + 1, 1)
    cycles = freqs/4.0

    # store behavioral data
    df_all = []
    # store epoch count per participant
    n_ave = []

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
            subj_data = data[idx + participants.pilot_counter == data.ID]
        elif study == 2:  # noqa: PLR2004
            subj_data = data[idx + 1 == data.ID]
        else:
            raise ValueError("input should be 1 or 2 for the respective study")

        # Ignored bad epochs are those defined by the user as bad epochs
        search_string = "IGNORED"
        # Remove trials where epochs are labeled as too short
        indices = [index for index, tpl in enumerate(epochs.drop_log) if tpl and search_string not in tpl]

        # Drop trials without a corresponding epoch
        if indices:
            epochs.metadata = subj_data.reset_index().drop(indices)
        else:
            epochs.metadata = subj_data

        if drop_bads:
            # drop epochs with abnormal strong signal (> 200 micro-volts)
            epochs.drop_bad(reject=dict(eeg=200e-6))

        # crop epochs in the desired time window
        epochs.crop(tmin=tmin, tmax=tmax)

        # apply CSD to the data (less point spread)
        if laplace:
            epochs = mne.preprocessing.compute_current_source_density(epochs)

        # avoid leakage and edge artifacts by mirroring the data on both ends (for simulations see)
        if mirror:
     
            # if zero pad = False = mirror the data on both ends
            padded_epochs = zero_pad_or_mirror_epochs(epochs, zero_pad=False, pad_length=100)
            
            # store the mirrored epochs
            epochs = padded_epochs

        # subtract evoked response from each epoch
        if induced:
            epochs = epochs.subtract_evoked()

        # store behavioral data
        df_all.append(epochs.metadata)

        if cond == "probability":
            if study == 1:

                epochs_a = epochs[
                    (
                        (epochs.metadata.cue == params.high_p)
                        & (epochs.metadata.previsyes == 0)
                        & (epochs.metadata.prevresp == 1)
                    )
                ]
                epochs_b = epochs[
                    (
                        (epochs.metadata.cue == params.low_p)
                        & (epochs.metadata.previsyes == 0)
                        & (epochs.metadata.prevresp == 1)
                    )
                ]
                cond_a_name = f"high_prevfa"
                cond_b_name = f"low_prevfa"

            elif study == 2:  # noqa: PLR2004
                epochs_a = epochs[(epochs.metadata.cue == params.high_p)]
                epochs_b = epochs[(epochs.metadata.cue == params.low_p)]
                cond_a_name = f"high"
                cond_b_name = f"low"

        elif cond == "prev_resp":
            if study == 1:
                epochs_a = epochs[
                    (
                        (epochs.metadata.prevresp == 1)
                        & (epochs.metadata.previsyes == 1)
                        & (epochs.metadata.cue == params.high_p)
                    )
                ]
                epochs_b = epochs[
                    (
                        (epochs.metadata.prevresp == 0)
                        & (epochs.metadata.previsyes == 1)
                        & (epochs.metadata.cue == params.high_p)
                    )
                ]
                cond_a_name = f"prevyesresp_highprob_prevstim"
                cond_b_name = f"prevnoresp_highprob_prevstim"
            elif study == 2:
                epochs_a = epochs[
                    ((epochs.metadata.prevresp == 1) & (epochs.metadata.prevcue == epochs.metadata.cue) &
                     (epochs.metadata.cue == params.low_p))
                ]

                epochs_b = epochs[
                    ((epochs.metadata.prevresp == 0) & (epochs.metadata.prevcue == epochs.metadata.cue) &
                     (epochs.metadata.cue == params.low_p))

                ]

                cond_a_name = f"prevyesresp_samecue_lowprob"
                cond_b_name = f"prevnoresp_samecue_lowprob"
        elif cond == "hitmiss":
            epochs_a = epochs[
                (
                    (epochs.metadata.sayyes == 1)
                    & (epochs.metadata.isyes == 1)
                    & (epochs.metadata.cue == params.high_p)
                )
            ]
            epochs_b = epochs[
                (
                    (epochs.metadata.sayyes == 0)
                    & (epochs.metadata.isyes == 1)
                    & (epochs.metadata.cue == params.high_p)
                )
            ]
            cond_a_name = f"hit_high_prob"
            cond_b_name = f"miss_high_prob"
        elif cond == 'response':
            epochs_a = epochs[
                (
                    (epochs.metadata.sayyes == 1)
                    & (epochs.metadata.isyes == 1)
                    & (epochs.metadata.cue == params.high_p)
                )
            ]
            epochs_b = epochs[
                (
                    (epochs.metadata.sayyes == 1)
                    & (epochs.metadata.isyes == 1)
                    & (epochs.metadata.cue == params.low_p)
                )
            ]
            cond_a_name = f"hit_high_prob"
            cond_b_name = f"hit_low_prob"

        else:
            raise ValueError("input should be 'probability' or 'prev_resp' or 'hitmiss'")

        # make sure we have an equal number of trials in both conditions
        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        n_epochs = len(epochs_a.events)
        n_ave.append(n_epochs)

        # Define the suffix based on whether mirroring was applied
        suffix_mirror = "_mirror" if mirror else ""
        suffix_induced = "_induced" if induced else ""

        # set tfr path
        if (Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{tmin}_{tmax}_{study!s}{suffix_mirror}{suffix_induced}-tfr.h5")).exists():
            print("TFR already exists")
        else:
            tfr_a = mne.time_frequency.tfr_multitaper(
                epochs_a, freqs=freqs, n_cycles=cycles, return_itc=False, n_jobs=-1, average=True
            )

            tfr_b = mne.time_frequency.tfr_multitaper(
                epochs_b, freqs=freqs, n_cycles=cycles, return_itc=False, n_jobs=-1, average=True
            )

            # Save TFR data with the appropriate filename (with or without "mirror" in the name)
            tfr_a.save(fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, 
                                f"{subj}_{cond_a_name}_{tmin}_{tmax}_{study!s}{suffix_mirror}{suffix_induced}-tfr.h5"))
            tfr_b.save(fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, 
                                f"{subj}_{cond_b_name}_{tmin}_{tmax}_{study!s}{suffix_mirror}{suffix_induced}-tfr.h5"))

    # save number of epochs per participant as csv file
    pd.DataFrame(n_ave).to_csv(Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"n_ave_{cond_a_name}_{study!s}.csv"))

    return "Done with tfr/erp computation", cond_a_name, cond_b_name


def load_tfr_conds(
    studies=[1, 2],
    cond="probability",
    cond_a_name="high",
    cond_b_name="low",
    cond_a_names=["high_prevhit", "high_prevmiss", "high_prevcr"],
    cond_b_names=["low_prevhit", "low_prevmiss", "low_prevcr"],
    tmin: int = -0.7, 
    tmax: int = 0,
    suffix_mirror: str = '_mirror', 
    suffix_induced: str ="_induced"
):
    """
    Load tfr data for the two conditions.

    Args:
    ----
    studies : list, info: load data from one study or both
    cond : str, info: which condition to analyze: "probability" or "prev_resp"
    cond_a_name : str
        which condition tfr to load: high or low or prevyesresp or prevnoresp
    cond_b_name : str
        which condition tfr to load: high or low or prevyesresp or prevnoresp
    cond_a_names : list of str
        list of condition names
    cond_b_names : list of str
        list of condition names

    Returns:
    -------
    tfr_a_all: list: list of tfr objects for condition a
    tfr_b_all: list: list of tfr objects for condition b

    """
    tfr_a_cond, tfr_b_cond = [], []  # store condition data

    for study in studies:
        tfr_a_all, tfr_b_all = [], []  # store participant data

        # adjust id list
        if study == 1:
            id_list = participants.ID_list_expecon1
        elif study == 2:  # noqa: PLR2004
            id_list = participants.ID_list_expecon2
        else:
            raise ValueError("input should be 1 or 2 for the respective study")
            # now load data for each participant and each condition
        if cond == "probability":
            if study == 1:
                for subj in id_list:
                    tfr_a_all_conds, tfr_b_all_conds = [], []  # store the conditions
                    for c_a, c_b in zip(cond_a_names, cond_b_names):
                        # load tfr data
                        tfr_a = mne.time_frequency.read_tfrs(
                            fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{c_a}_{tmin}_{tmax}_{study!s}{suffix_mirror}{suffix_induced}-tfr.h5"),
                            condition=0,
                        )
                        tfr_b = mne.time_frequency.read_tfrs(
                            fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{c_b}_{tmin}_{tmax}_{study!s}{suffix_mirror}{suffix_induced}-tfr.h5"),
                            condition=0,
                        )
                        tfr_a_all_conds.append(tfr_a)
                        tfr_b_all_conds.append(tfr_b)
                    tfr_a_all.append(tfr_a_all_conds)
                    tfr_b_all.append(tfr_b_all_conds)
            elif study == 2:  # noqa: PLR2004
                for subj in id_list:
                    if subj == "013":
                        continue
                    # load tfr data
                    tfr_a = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{tmin}_{tmax}_{study!s}{suffix_mirror}{suffix_induced}-tfr.h5"),
                        condition=0,
                    )
                    tfr_b = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_b_name}_{tmin}_{tmax}_{study!s}{suffix_mirror}{suffix_induced}-tfr.h5"),
                        condition=0,
                    )
                    tfr_a_all.append(tfr_a)
                    tfr_b_all.append(tfr_b)

        elif cond == "prev_resp":
            if study == 2:
                for subj in id_list:
                    if subj == "013":
                        continue
                    tfr_a_all_conds, tfr_b_all_conds = [], []  # store the conditions
                    for c_a, c_b in zip(cond_a_names, cond_b_names):
                        # load tfr data
                        tfr_a = mne.time_frequency.read_tfrs(
                            fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{c_a}_{tmin}_{tmax}_{study!s}{suffix_mirror}{suffix_induced}-tfr.h5"),
                            condition=0,
                        )
                        tfr_b = mne.time_frequency.read_tfrs(
                            fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{c_b}_{tmin}_{tmax}_{study!s}{suffix_mirror}{suffix_induced}-tfr.h5"),
                            condition=0,
                        )
                        tfr_a_all_conds.append(tfr_a)
                        tfr_b_all_conds.append(tfr_b)
                    tfr_a_all.append(tfr_a_all_conds)
                    tfr_b_all.append(tfr_b_all_conds)
            elif study == 1:
                for subj in id_list:
                    # load tfr data
                    tfr_a = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{tmin}_{tmax}_{study!s}{suffix_mirror}{suffix_induced}-tfr.h5"),
                        condition=0,
                    )
                    tfr_b = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_b_name}_{tmin}_{tmax}_{study!s}{suffix_mirror}{suffix_induced}-tfr.h5"),
                        condition=0,
                    )
                    tfr_a_all.append(tfr_a)
                    tfr_b_all.append(tfr_b)
        elif cond == 'hitmiss' or 'response':
            if study == 2:
                for subj in id_list:
                    if subj == "013":
                        continue
                    # load tfr data
                    tfr_a = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{tmin}_{tmax}_{study!s}{suffix_induced}-tfr.h5"),
                        condition=0,
                    )
                    tfr_b = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_b_name}_{tmin}_{tmax}_{study!s}{suffix_induced}-tfr.h5"),
                        condition=0,
                    )
                    tfr_a_all.append(tfr_a)
                    tfr_b_all.append(tfr_b)
            elif study == 1:
                for subj in id_list:
                    # load tfr data
                    tfr_a = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{tmin}_{tmax}_{study!s}{suffix_induced}-tfr.h5"),
                        condition=0,
                    )
                    tfr_b = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_b_name}_{tmin}_{tmax}_{study!s}{suffix_induced}-tfr.h5"),
                        condition=0,
                    )
                    tfr_a_all.append(tfr_a)
                    tfr_b_all.append(tfr_b)
        
        tfr_a_cond.append(tfr_a_all)
        tfr_b_cond.append(tfr_b_all)

    return tfr_a_cond, tfr_b_cond, cond


def plot_tfr_cluster_test_output(
    cond: str = 'probability',
    threed_test: bool = False,
    cond_a_name: str = 'high',
    cond_b_name: str = 'low',
    channel_names: list = ['CP4'],
    tmin: float = -0.7,
    tmax: float = 0,
):
    """
    Plot cluster permutation test output for tfr data (time and frequency cluster).

    Args:
    ----
    cond : str, info: which condition to analyze: "probability" or "prev_resp"
    tfr_a_cond : list of tfr objects
        tfr data for condition a
    tfr_b_cond : list of tfr objects
        tfr data for condition b
    threed_test : boolean,
        info: whether to run a 3D cluster test or not
    cond_a_name : str
        condition a
    cond_b_name : str
        condition b
    channel_names: list of char
        channels to analyze
    ----------

    Returns:
    -------
    None

    """
    tfr_a_cond, tfr_b_cond, _ = load_tfr_conds()

    # Create a 2x3 grid of plots (2 rows, 3 columns)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Adjust the space between the subplots
    plt.subplots_adjust(wspace=0.7)
    # which time windows to plot
    time_windows = [(tmin, tmax), (tmin, tmax)]

    # loop over time windows and axes
    for idx, t in enumerate(time_windows):
        if threed_test:
            if cond == "probability":
                if idx == 0:
                    tfr_a = np.array(tfr_a[idx])
                    tfr_b = np.array(tfr_b[idx])
                    x_conds = []
                    for conds in range(tfr_a.shape[1]):
                        # contrast data
                        x = np.array([
                            h.copy().crop(tmin=t[0], tmax=t[1]).data
                            - l_.copy().crop(tmin=t[0], tmax=t[1]).data
                            for h, l_ in zip(tfr_a[:, conds], tfr_b[:, conds])
                        ])
                        x_conds.append(x)

                    # average across conditions (get rid of previous trial response)
                    x = np.mean(
                        np.array(list(x_conds)), axis=0
                    )  # should be shape participants x frequencies x timepoints
                else:
                    # contrast data
                    x = np.array([
                        h.copy().crop(tmin=t[0], tmax=t[1]).data
                        - l_.copy().crop(tmin=t[0], tmax=t[1]).data
                        for h, l_ in zip(tfr_a[idx], tfr_b[idx])
                    ])
            elif cond == "prev_resp":
                if idx == 1:
                    tfr_a = np.array(tfr_a[idx])
                    tfr_b = np.array(tfr_b[idx])
                    x_conds = []
                    for conds in range(tfr_a.shape[1]):  # exclude prevfa
                        # contrast data
                        x = np.array([
                            h.copy().crop(tmin=t[0], tmax=t[1]).data
                            - l_.copy().crop(tmin=t[0], tmax=t[1]).data
                            for h, l_ in zip(tfr_a[:, conds], tfr_b[:, conds])
                        ])
                        x_conds.append(x)

                    # average across conditions (get rid of previous trial response)
                    x = np.mean(
                        np.array(list(x_conds)), axis=0
                    )  # should be shape participants x frequencies x timepoints
                else:
                    # contrast data
                    x = np.array([
                        h.copy().crop(tmin=t[0], tmax=t[1]).data
                        - l_.copy().crop(tmin=t[0], tmax=t[1]).data
                        for h, l_ in zip(tfr_a[idx], tfr_b[idx])
                    ])

            # define adjacency for channels
            ch_adjacency, _ = mne.channels.find_ch_adjacency(tfr_a[0][0].info, ch_type="eeg")

            # visualize channel adjacency
            mne.viz.plot_ch_adjacency(tfr_a[0][0].info, ch_adjacency, tfr_a[0][0].ch_names)
            # our data at each observation is of shape frequencies x times x channels
            tfr_adjacency = mne.stats.combine_adjacency(
                len(tfr_a[0][0].freqs), len(tfr_b[0][0].crop(tmin=t[0], tmax=t[1]).times), ch_adjacency
            )

            x = x.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1])
            print(
                f"Shape of array for cluster test should be"
                f" participants x frequencies x timepoints x channels: {x.shape}"
            )

            # run cluster test over time, frequencies and channels
            t_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(
                x, n_jobs=-1, n_permutations=1000, threshold=None, tail=0, 
                adjacency=tfr_adjacency
            )

        else: # 2D test
            if cond == "probability":
                if idx == 0:
                    tfr_a = np.array(tfr_a_cond[idx])
                    tfr_b = np.array(tfr_b_cond[idx])
                    x_conds = []
                    for conds in range(tfr_a.shape[1]):
                        # contrast data
                        x = np.array([
                            h.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                            - l_.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                            for h, l_ in zip(tfr_a[:, conds], tfr_b[:, conds])
                        ])
                        x_conds.append(x)

                    # average across conditions (get rid of previous trial response)
                    x = np.mean(
                        np.array(list(x_conds)), axis=0
                    )  # should be shape participants x frequencies x timepoints
                else:
                    # contrast data
                    x = np.array([
                        h.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                        - l_.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                        for h, l_ in zip(tfr_a_cond[idx], tfr_b_cond[idx])
                    ])
            elif cond == "prev_resp":
                if idx == 1:
                    tfr_a = np.array(tfr_a_cond[idx])
                    tfr_b = np.array(tfr_b_cond[idx])
                    x_conds = []
                    for conds in range(tfr_a.shape[1]):  # exclude prevfa
                        # contrast data
                        x = np.array([
                            h.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                            - l_.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                            for h, l_ in zip(tfr_a[:, conds], tfr_b[:, conds])
                        ])
                        x_conds.append(x)

                    # average across conditions (get rid of previous trial response)
                    x = np.mean(
                        np.array(list(x_conds)), axis=0
                    )  # should be shape participants x frequencies x timepoints
                else:
                    # contrast data
                    x = np.array([
                        h.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                        - l_.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                        for h, l_ in zip(tfr_a_cond[idx], tfr_b_cond[idx])
                    ])
            else:
                # contrast data
                x = np.array([
                    h.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                    - l_.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                    for h, l_ in zip(tfr_a_cond[idx], tfr_b_cond[idx])
                ])
            
            # pick channel or average over channels
            x = np.mean(x, axis=1) if len(channel_names) > 1 else np.squeeze(x)
            print(f"Shape of array for cluster test should be participants x frequencies x timepoints: {x.shape}")

            # define threshold dictionary for threshold-free cluster enhancement
            threshold_tfce = dict(start=0, step=0.1)

            # run cluster test over time and frequencies (no need to define adjacency)
            t_obs, cluster, cluster_p, ho = mne.stats.permutation_cluster_1samp_test(
                x, n_jobs=-1, n_permutations=1000, threshold=threshold_tfce, tail=0,
                seed=42
            )
            
            # print stats for manuscript
            print(f"smallest cluster p-value: {min(cluster_p)}")

            # get index of cluster with lowest p value
            lowest_p_idx = np.argmin(cluster_p)

            stats = []

            # Reshape for later: participants x (freq*time)
            x_reshaped = x.reshape(x.shape[0], -1)
            t_obs_flat = t_obs.flatten()

            mask = cluster[lowest_p_idx][0].flatten()  # cluster mask over freqs x times
            cluster_size = np.sum(mask)

            # Peak t-value within the cluster
            peak_t = np.max(np.abs(t_obs_flat[mask]))

            # Extract participant data for that cluster
            x_cluster = x_reshaped[:, mask]
            subject_means = np.mean(x_cluster, axis=1)

            # Cohen's d
            mean_diff = np.mean(subject_means)
            std_diff = np.std(subject_means, ddof=1)
            cohen_d = mean_diff / std_diff if std_diff != 0 else np.nan

            stats.append({
                "Cluster Index": lowest_p_idx + 1,
                "Cluster p-value": round(cluster_p[lowest_p_idx], 4),
                "Cluster Size": int(cluster_size),
                "Peak |t|": round(peak_t, 3),
                "Cohen's d": round(cohen_d, 3),
                "H0": "mean = 0"
            })

            df = pd.DataFrame(stats)
            df.to_csv(Path(paths.figures.manuscript.figure3, f"cluster_stats_with_power_{cond}_{str(idx)}.csv", index=False))

            # set mask for plotting sign. points that belong to the cluster
            mask = np.array(cluster_p).reshape(t_obs.shape[0], t_obs.shape[1])
            mask = mask < params.alpha  # boolean for masking sign. voxels

            # plot t contrast and sign. cluster contour
            plot_cluster_contours(
                data=x,
                t_obs=t_obs,
                tfr_a_cond=tfr_a_cond,
                fmin=3,
                fmax=35,
                idx=idx,
                t=t,
                axs=axs,
                mask=mask,
                cond=cond,
            )

    plt.tight_layout()
    # now save the figure to disk as png and svg
    for fm in ["svg", "png"]:
        fig.savefig(
            Path(
                paths.figures.manuscript.figure3,
                f"fig3_tfr_{cond}_{tmin}_{tmax}_{channel_names[0]}.{fm}",
            ),
            dpi=300,
            format=fm,
        )
    plt.show()

    return "Done with cluster test", t_obs, cluster, cluster_p, ho


def plot_cluster_contours(
    data: np.ndarray,
    t_obs: np.ndarray,
    tfr_a_cond: np.ndarray,
    fmin: float,
    fmax: float,
    idx: int,
    t: float,
    axs: tuple,
    mask: np.ndarray,
    cond: str,
) -> str:
    """
    Plot cluster permutation test output.

    The cluster is highlighted via a contour around it, code adapted Gimpert et al.
    x and p ticks and labels, as well as colorbar are not plotted

    Args:
    ----
    data: array, info: data to plot in heatmap
    t_obs: array, info: t-values
    tfr_a_cond: array, info: tfr data for condition a
    fmin: float, info: minimum frequency to plot
    fmax: float, info: maximum frequency to plot
    idx: int, info: index of the subplot
    t: float, info: time window to plot
    axs: tuple, info: axes to plot
    mask: array, info: mask for the cluster

    """
    if cond == "probability":
        # Define timepoints and frequencies
        times = tfr_a_cond[1][5].copy().crop(tmin=t[0], tmax=t[1]).times
    else:
        times = tfr_a_cond[0][5].copy().crop(tmin=t[0], tmax=t[1]).times

    freqs = np.arange(fmin, fmax, 1)

    # Set custom x labels and ticks (timepoints)
    x_ticks, x_labels = [], []

    # Iterate through the list and select every 3rd entry starting from the first entry
    for x_idx, x_value in enumerate(times):
        if x_idx % 25 == 0:
            x_ticks.append(x_idx)
            x_labels.append(x_value)

    # Set custom y labels and ticks (frequencies)
    y_ticks, y_labels = [], []

    # Iterate through the list and select every 3rd entry starting from the first entry
    for y_idx, y_value in enumerate(freqs):
        if y_idx % 3 == 0:
            y_ticks.append(y_idx)
            y_labels.append(y_value)

    res = scipy.stats.ttest_1samp(data, popmean=0)

    t_val = np.squeeze(res[0])
    if cond == "probability":
        vmin_val = -6 #min(t_val)
        vmax_val = 6 #abs(vmin_val)
    else:
        vmin_val = -3
        vmax_val = 3
    # Plot tmap as heatmap, use the same vmin and vmax for all plots
    sns.heatmap(t_val, cbar=True, cmap="coolwarm", robust=True, ax=axs[idx], vmin=vmin_val, vmax=vmax_val)

    # Customize the font size and family for various elements
    plt.rcParams["font.family"] = "Arial"  # Set the font family for the entire plot to Arial
    plt.rcParams["font.size"] = 16  # Set the default font size to 12

    # Draw the cluster outline
    for i in range(mask.shape[0]):  # frequencies
        for j in range(mask.shape[1]):  # time
            if mask[i, j]:
                if i > 0 and not mask[i - 1, j]:
                    axs[idx].plot([j - 0.5, j + 0.5], [i, i], color="white", linewidth=2)
                if i < mask.shape[0] - 1 and not mask[i + 1, j]:
                    axs[idx].plot([j - 0.5, j + 0.5], [i + 1, i + 1], color="white", linewidth=2)
                if j > 0 and not mask[i, j - 1]:
                    axs[idx].plot([j - 0.5, j - 0.5], [i, i + 1], color="white", linewidth=2)
                if j < mask.shape[1] - 1 and not mask[i, j + 1]:
                    axs[idx].plot([j + 0.5, j + 0.5], [i, i + 1], color="white", linewidth=2)

    axs[idx].invert_yaxis()
    axs[idx].axvline([t_obs.shape[1]], color="white", linestyle="dotted", linewidth=5)  # stimulation onset
    axs[idx].set(xlabel="Time (s)", ylabel="Frequency (Hz)")
    # Set the font size for xlabel and ylabel
    axs[idx].xaxis.label.set_fontsize(20)  # Adjust the font size for the xlabel
    axs[idx].yaxis.label.set_fontsize(20)  # Adjust the font size for the ylabel

    # Set custom the x and y-axis ticks and labels
    axs[idx].set_xticks(x_ticks)
    axs[idx].set_xticklabels(x_labels, rotation=0)
    axs[idx].set_yticks(y_ticks)
    axs[idx].set_yticklabels(y_labels, rotation=0)
    axs[idx].yaxis.labelpad = 15  # Adjust the distance between the y-axis label and the y-axis ticks
    axs[idx].xaxis.labelpad = 15  # Adjust the distance between the x-axis label and the y-axis ticks

    # Increase font size for x and y ticks
    axs[idx].tick_params(axis='both', which='major', labelsize=16)  # Adjust font size

    return "Plotted and saved figure"


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
