#!/usr/bin/python3
"""
Provide functions that compute time-frequency representations (TFRs).

Moreover, run a cluster test in 2D space (time and frequency)

This script produces figure 3 in Forster et al.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2024
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
from expecon_ms.utils import zero_pad_or_mirror_data, drop_trials

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
    cycles = freqs / 4.0

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

        # avoid leakage and edge artifacts by zero padding the data
        if mirror:
            metadata = epochs.metadata

            epoch_data = epochs.get_data()

            # zero pad = False = mirror the data on both ends
            data_mirror = zero_pad_or_mirror_data(epoch_data, zero_pad=False)

            # put back into the epoch structure
            epochs = mne.EpochsArray(data_mirror, epochs.info, tmin=tmin * 2)

            # add metadata back
            epochs.metadata = metadata

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
                        & (epochs.metadata.prevresp == 0)
                    )
                ]
                epochs_b = epochs[
                    (
                        (epochs.metadata.cue == params.low_p)
                        & (epochs.metadata.previsyes == 0)
                        & (epochs.metadata.prevresp == 0)
                    )
                ]
                cond_a_name = f"high_prevcr_{tmin}_{tmax}_induced"
                cond_b_name = f"low_prevcr_{tmin}_{tmax}_induced"

            elif study == 2:  # noqa: PLR2004
                epochs_a = epochs[(epochs.metadata.cue == params.high_p)]
                epochs_b = epochs[(epochs.metadata.cue == params.low_p)]
                cond_a_name = f"high_{tmin}_{tmax}_induced"
                cond_b_name = f"low_{tmin}_{tmax}_induced"

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
                cond_a_name = f"prevyesresp_highprob_prevstim_{tmin}_{tmax}_induced"
                cond_b_name = f"prevnoresp_highprob_prevstim_{tmin}_{tmax}_induced"
            elif study == 2:
                epochs_a = epochs[
                    ((epochs.metadata.prevresp == 1) & (epochs.metadata.prevcue == epochs.metadata.cue) &
                     (epochs.metadata.cue == params.high_p))
                ]

                epochs_b = epochs[
                    ((epochs.metadata.prevresp == 0) & (epochs.metadata.prevcue == epochs.metadata.cue) &
                     (epochs.metadata.cue == params.high_p))

                ]

                cond_a_name = f"prevyesresp_samecue_highprob_{tmin}_{tmax}_induced"
                cond_b_name = f"prevnoresp_samecue_highprob_{tmin}_{tmax}_induced"
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
            cond_a_name = f"hit_highprob_{tmin}_{tmax}_induced"
            cond_b_name = f"miss_highprob_{tmin}_{tmax}_induced"
        else:
            raise ValueError("input should be 'probability' or 'prev_resp' or 'hitmiss'")

        # make sure we have an equal number of trials in both conditions
        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        n_epochs = len(epochs_a.events)
        n_ave.append(n_epochs)

        # set tfr path
        if (Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{study!s}-tfr.h5")).exists():
            print("TFR already exists")
        else:
            tfr_a = mne.time_frequency.tfr_multitaper(
                epochs_a, freqs=freqs, n_cycles=cycles, return_itc=False, n_jobs=-1, average=True
            )

            tfr_b = mne.time_frequency.tfr_multitaper(
                epochs_b, freqs=freqs, n_cycles=cycles, return_itc=False, n_jobs=-1, average=True
            )

            # save tfr data
            tfr_a.save(fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{study!s}-tfr.h5"))
            tfr_b.save(fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_b_name}_{study!s}-tfr.h5"))

    # save number of epochs per participant as csv file
    pd.DataFrame(n_ave).to_csv(Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"n_ave_{cond_a_name}_{study!s}.csv"))

    return "Done with tfr/erp computation", cond_a_name, cond_b_name


def load_tfr_conds(
    studies: list = [1, 2], cond: str = 'hitmiss', cond_a_name: str = "hit_highprob_-0.7_0_induced", 
                cond_b_name: str = "miss_highprob_-0.7_0_induced", cond_a_names: list = None, cond_b_names: list = None
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
                            fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{c_a}_{study!s}-tfr.h5"),
                            condition=0,
                        )
                        tfr_b = mne.time_frequency.read_tfrs(
                            fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{c_b}_{study!s}-tfr.h5"),
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
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{study!s}-tfr.h5"),
                        condition=0,
                    )
                    tfr_b = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_b_name}_{study!s}-tfr.h5"),
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
                            fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{c_a}_{study!s}-tfr.h5"),
                            condition=0,
                        )
                        tfr_b = mne.time_frequency.read_tfrs(
                            fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{c_b}_{study!s}-tfr.h5"),
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
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{study!s}-tfr.h5"),
                        condition=0,
                    )
                    tfr_b = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_b_name}_{study!s}-tfr.h5"),
                        condition=0,
                    )
                    tfr_a_all.append(tfr_a)
                    tfr_b_all.append(tfr_b)
        elif cond == 'hitmiss':
            if study == 2:
                for subj in id_list:
                    if subj == "013":
                        continue
                    # load tfr data
                    tfr_a = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{study!s}_mirror-tfr.h5"),
                        condition=0,
                    )
                    tfr_b = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_b_name}_{study!s}_mirror-tfr.h5"),
                        condition=0,
                    )
                    tfr_a_all.append(tfr_a)
                    tfr_b_all.append(tfr_b)
            elif study == 1:
                for subj in id_list:
                    # load tfr data
                    tfr_a = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{study!s}_mirror-tfr.h5"),
                        condition=0,
                    )
                    tfr_b = mne.time_frequency.read_tfrs(
                        fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_b_name}_{study!s}_mirror-tfr.h5"),
                        condition=0,
                    )
                    tfr_a_all.append(tfr_a)
                    tfr_b_all.append(tfr_b)
        
        tfr_a_cond.append(tfr_a_all)
        tfr_b_cond.append(tfr_b_all)

    return tfr_a_cond, tfr_b_cond, cond


def perform_stat_tests(power_high: np.array, power_low: np.array, times: np.array):
    """
    Perform statistical tests on power difference and return significant timepoints.
    
    Args:
    -----
    power_high : np.array
        Power across participants for the high condition.
    power_low : np.array
        Power across participants for the low condition.
    ch_index : int
        Index of the channel.
    
    Returns:
    --------
    sign_timepoints : np.array
        Significant timepoints where the p-value is below 0.05.
    """
    diff = power_high - power_low
    t_vals, p_vals, _ = mne.stats.permutation_t_test(diff)
    print(f"Significant time points: {times[np.where(p_vals < 0.05)]}")
    return times[np.where(p_vals < 0.05)]


def compute_power(tfr_a_cond, tfr_b_cond, ch_index: int, freq_band: tuple, study: int, cond: str,
                    idx_cond: int, tmin: float, tmax: float):
    """
    Load the data and compute power for the given frequency band.

    Args:
    -----
    tfr_a_cond : list
        TFR data for high condition.
    tfr_b_cond : list
        TFR data for low condition.
    ch_index : int
        Index of the channel to analyze.
    freq_band : tuple
        Frequency band for power computation (start_freq, end_freq).
    study : str
        Study name for labeling.
    cond : str
        Condition to analyze: either "probability" or "prev_resp".

    Returns:
    --------
    high_mean : np.array
        Mean power for the high condition.
    high_sem : np.array
        SEM for the high condition.
    low_mean : np.array
        Mean power for the low condition.
    low_sem : np.array
        SEM for the low condition.
    """
    start_freq, end_freq = freq_band
    end_freq += 1  # Include the last frequency in the range

    if cond == 'probability':
        if study == 'stable':
                # Compute band power only for the specified channel for all participants in the stable condition
                high_power = np.array([scipy.integrate.trapezoid(a[idx_cond].crop(tmin,tmax).data[ch_index, start_freq:end_freq, :], axis=0) for a in tfr_a_cond])
                low_power = np.array([scipy.integrate.trapezoid(a[idx_cond].crop(tmin,tmax).data[ch_index, start_freq:end_freq, :], axis=0) for a in tfr_b_cond])
        else:
            # Compute band power only for the specified channel for all participants in the volatile condition
            high_power = np.array([scipy.integrate.trapezoid(a.crop(tmin,tmax).data[ch_index, start_freq:end_freq, :], axis=0) for a in tfr_a_cond])
            low_power = np.array([scipy.integrate.trapezoid(a.crop(tmin,tmax).data[ch_index, start_freq:end_freq, :], axis=0) for a in tfr_b_cond])
    else:
        if study == 'stable':
            # Compute band power only for the specified channel for all participants in the stable condition
            high_power = np.array([scipy.integrate.trapezoid(a.crop(tmin,tmax).data[ch_index, start_freq:end_freq, :], axis=0) for a in tfr_a_cond])
            low_power = np.array([scipy.integrate.trapezoid(a.crop(tmin,tmax).data[ch_index, start_freq:end_freq, :], axis=0) for a in tfr_b_cond])
        else:
            # Compute band power only for the specified channel for all participants in the volatile condition
            high_power = np.array([scipy.integrate.trapezoid(a[idx_cond].crop(tmin,tmax).data[ch_index, start_freq:end_freq, :], axis=0) for a in tfr_a_cond])
            low_power = np.array([scipy.integrate.trapezoid(a[idx_cond].crop(tmin,tmax).data[ch_index, start_freq:end_freq, :], axis=0) for a in tfr_b_cond])

    # Calculate mean and SEM for both conditions
    high_mean = np.mean(high_power, axis=0)
    high_sem = scipy.stats.sem(high_power, axis=0)
    low_mean = np.mean(low_power, axis=0)
    low_sem = scipy.stats.sem(low_power, axis=0)

    return high_mean, high_sem, low_mean, low_sem, high_power, low_power


def calc_freq_band_power_over_time(ch_name: str = 'CP4', cond: str = 'probability', tmin=-0.7, tmax=0, 
                                    fmin_alpha: int = 8, fmax_alpha: int = 12, fmin_beta: int = 15, fmax_beta: int = 30):
    """
    Plot the power for both high and low conditions (not the difference) for the alpha and beta bands.
    
    Args:
    ----
    ch_name : str
        Name of the channel to analyze (e.g., 'CP4').
    cond : str
        Condition to analyze: either "probability" or "prev_resp".
    
    Returns:
    -------
    None
    """
    if cond == "probability":
        # Load data for probability condition
        tfr_a_cond, tfr_b_cond, cond = load_tfr_conds(
            studies=[1, 2],
            cond="probability",
            cond_a_name="high_-0.7_0_induced",
            cond_b_name="low_-0.7_0_induced",
            cond_a_names=["high_prevhit_-0.7_0_induced", "high_prevmiss_-0.7_0_induced", "high_prevcr_-0.7_0_induced"],
            cond_b_names=["low_prevhit_-0.7_0_induced", "low_prevmiss_-0.7_0_induced", "low_prevcr_-0.7_0_induced"],
        )
        cond_name = 'high vs low'
    else:
        # Load data for previous response condition
        tfr_a_cond, tfr_b_cond, cond = load_tfr_conds(
            studies=[1, 2],
            cond="prev_resp",
            cond_a_name="prevyesresp_highprob_prevstim_-0.7_0_induced",
            cond_b_name="prevnoresp_highprob_prevstim_-0.7_0_induced",
            cond_a_names=["prevyesresp_samecue_lowprob_-0.7_0_induced", "prevyesresp_samecue_highprob_-0.7_0_induced"],
            cond_b_names=["prevnoresp_samecue_lowprob_-0.7_0_induced", "prevnoresp_samecue_highprob_-0.7_0_induced"],
        )
        cond_name = 'previous yes vs previous no response'

    if cond == "probability":
        times = tfr_a_cond[1][0].crop(tmin, tmax).times
        ch_index = tfr_a_cond[1][0].ch_names.index(ch_name)
    else:
        times = tfr_a_cond[0][0].crop(tmin, tmax).times
        ch_index = tfr_a_cond[0][0].ch_names.index(ch_name)
        
    for idx, (cond_a, cond_b) in enumerate(zip(tfr_a_cond, tfr_b_cond)):
        if cond == "probability":
            if idx == 1:
                conditions = None
                idx_cond = None
                study = 'volatile'

                # Alpha and Beta power computation for high and low conditions
                beta_high_mean, beta_high_sem, beta_low_mean, beta_low_sem, high_power_beta, low_power_beta = compute_power(tfr_a_cond[idx], tfr_b_cond[idx],
                                                                                ch_index, (fmin_beta, fmax_beta), study, 'probability', idx_cond, tmin, tmax)
                alpha_high_mean, alpha_high_sem, alpha_low_mean, alpha_low_sem, high_power_alpha, low_power_alpha = compute_power(tfr_a_cond[idx], tfr_b_cond[idx], 
                                                                                    ch_index, (fmin_alpha, fmax_alpha), study, 'probability', idx_cond, tmin, tmax)
                
                # get significant time points for alpha and  beta
                significant_times_alpha = perform_stat_tests(high_power_alpha, low_power_alpha, times)
                significant_times_beta = perform_stat_tests(high_power_beta, low_power_beta, times)

                # Plot results
                plot_freq_band_over_time(
                    ch_index, high_power_alpha, low_power_alpha, 
                    high_power_beta, low_power_beta,
                    times, cond, study, conditions, tmin, tmax, idx_cond, 
                    significant_times_alpha=significant_times_alpha,
                    significant_times_beta=significant_times_beta
                )

            else:
                conditions = ['previous hit', 'previous miss', 'previous correct rejection']
                study = 'stable'

                for idx_cond in range(len(conditions)):
                    # Alpha and Beta power computation for high and low conditions
                    beta_high_mean, beta_high_sem, beta_low_mean, beta_low_sem, high_power_beta, low_power_beta = compute_power(tfr_a_cond[idx], tfr_b_cond[idx], 
                                                                                    ch_index, (fmin_beta, fmax_beta), study, 'probability', idx_cond, tmin, tmax)
                    alpha_high_mean, alpha_high_sem, alpha_low_mean, alpha_low_sem, high_power_alpha, low_power_alpha = compute_power(
                                                                                    tfr_a_cond[idx], tfr_b_cond[idx], 
                                                                                    ch_index, (fmin_alpha, fmax_alpha), study, 'probability', idx_cond, tmin, tmax)

                    significant_times_alpha = perform_stat_tests(high_power_alpha, low_power_alpha, times)
                    significant_times_beta = perform_stat_tests(high_power_beta, low_power_beta, times)

                    # Plot results
                    plot_freq_band_over_time(
                        ch_index, high_power_alpha, low_power_alpha, 
                        high_power_beta, low_power_beta,
                        times, cond, study, conditions, tmin, tmax, idx_cond, 
                        significant_times_alpha=significant_times_alpha,
                        significant_times_beta=significant_times_beta
                    )
        else:
            if idx == 0:

                conditions = None
                study = 'stable'
                idx_cond = 0

                # Alpha and Beta power computation for high and low conditions
                beta_high_mean, beta_high_sem, beta_low_mean, beta_low_sem, high_power_beta, low_power_beta = compute_power(tfr_a_cond[idx], tfr_b_cond[idx], 
                                                                                ch_index, (fmin_beta, fmax_beta), study, 'prev_resp', idx_cond, tmin, tmax)
                alpha_high_mean, alpha_high_sem, alpha_low_mean, alpha_low_sem, high_power_alpha, low_power_alpha = compute_power(tfr_a_cond[idx], tfr_b_cond[idx], 
                                                                                ch_index, (fmin_alpha, fmax_alpha), study, 'prev_resp', idx_cond, tmin, tmax)

                significant_times_alpha = perform_stat_tests(high_power_alpha, low_power_alpha, times)
                significant_times_beta = perform_stat_tests(high_power_beta, low_power_beta, times)

                # Plot results
                plot_freq_band_over_time(
                    ch_index, high_power_alpha, low_power_alpha,
                    high_power_beta, low_power_beta,
                    times, cond, study, conditions, tmin, tmax, idx_cond, 
                    significant_times_alpha=significant_times_alpha,
                    significant_times_beta=significant_times_beta
                )
        
            else:
                conditions = ['high probability', 'low probability']
                study = 'volatile'

                for idx_cond in range(len(conditions)):
                    # Alpha and Beta power computation for high and low conditions
                    beta_high_mean, beta_high_sem, beta_low_mean, beta_low_sem, high_power_beta, low_power_beta = compute_power(tfr_a_cond[idx], tfr_b_cond[idx], 
                                                                                    ch_index, (fmin_beta, fmax_beta), study, 'prev_resp', idx_cond, tmin, tmax)
                    alpha_high_mean, alpha_high_sem, alpha_low_mean, alpha_low_sem, high_power_alpha, low_power_alpha = compute_power(tfr_a_cond[idx], tfr_b_cond[idx], 
                                                                                    ch_index, (fmin_alpha, fmax_alpha), study, 'prev_resp', idx_cond, tmin, tmax)

                    significant_times_alpha = perform_stat_tests(high_power_alpha, low_power_alpha, times)
                    significant_times_beta = perform_stat_tests(high_power_beta, low_power_beta, times)

                    # Plot results
                    plot_freq_band_over_time(
                        ch_index, high_power_alpha, low_power_alpha,
                        high_power_beta, low_power_beta,
                        times, cond, study, conditions, tmin, tmax, idx_cond, 
                        significant_times_alpha=significant_times_alpha,
                        significant_times_beta=significant_times_beta
                    )

    return "Done plotting freq bands over time"


def plot_freq_band_over_time(ch_name, alpha_high, alpha_low, beta_high, beta_low,
                             times, cond, study, conditions, tmin, tmax, idx_cond: int = 0,
                             significant_times_alpha=None, significant_times_beta=None):
    """
    Plot the power for both high and low probability conditions for the alpha and beta bands,
    and mark significant time points with horizontal lines.

    Args:
    ----
    ch_name : str
        Name of the channel to plot.
    alpha_high : np.array
        Power for the high probability condition in the alpha band.
    alpha_low : np.array
        Power for the low probability condition in the alpha band.
    beta_high : np.array
        Power for the high probability condition in the beta band.
    beta_low : np.array
        Power for the low probability condition in the beta band.
    times : np.array
        Time points for the x-axis.
    cond : str
        Condition name for labeling.
    study : str
        Study name for labeling.
    conditions : list of str
        List of conditions for the previous response condition.
    idx_cond : int
        Index of the current condition.
    significant_times : np.array or list, optional
        Array of time points where the difference between conditions is statistically significant (p < 0.05).

    Returns:
    -------
    None
    """

    def calculate_confidence_intervals(mean, sem, n, confidence_level=0.95):
        """Calculate the confidence intervals for a given mean and standard error."""
        from scipy.stats import t
        t_value = t.ppf((1 + confidence_level) / 2, n - 1)  # Use degrees of freedom (n - 1)
        ci_upper = mean + t_value * sem
        ci_lower = mean - t_value * sem
        return ci_lower, ci_upper

    def plot_difference_with_ci(ax, times, high, low, n_high, n_low, title, band_label, color_diff='green'):
        """
        Function to plot the difference between high and low power conditions 
        with shaded error (confidence intervals).
        """
        # Compute the difference between high and low for each participant
        diff = high - low

        # now calculate the mean difference over participants and the SEM
        diff_mean = np.mean(diff, axis=0)
        diff_sem = scipy.stats.sem(diff, axis=0)

        # Calculate the confidence intervals for the difference
        diff_ci_lower, diff_ci_upper = calculate_confidence_intervals(diff_mean, diff_sem, min(n_high, n_low))

        # Plot the difference with CI
        ax.plot(times, diff_mean, color=color_diff)
        ax.fill_between(times, diff_ci_lower, diff_ci_upper, color=color_diff, alpha=0.2)

        # Customize plot appearance
        ax.set_title(title, fontsize=20)
        ax.legend(loc='lower right', fontsize=12)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(band_label)


    def set_common_axis_properties(ax, x_label, y_label):
        """Set common axis properties for both subplots."""
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim(tmin, tmax)

    # Create the figure and subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Define the figure title based on study and condition
    if study == 'volatile':
        fig_title = f'{study.capitalize()} environment:' ' High minus Low Stimulus Probability' if cond == 'probability' else f'{study.capitalize()} environment: Previous Yes minus No Response \n{conditions[idx_cond]}'
    else:
        fig_title = f'{study.capitalize()} environment:' f' High minus Low Stimulus Probability \n{conditions[idx_cond]}' if cond == 'probability' else f'{study.capitalize()} environment: Previous Yes minus No Response'

    fig.suptitle(fig_title, fontsize=24, y=1.05)

    # Plot alpha band (High and Low Probability)
    plot_difference_with_ci(axs[0], times, alpha_high, alpha_low,
              len(alpha_high), len(alpha_low), "Alpha Frequency Band [8-12 Hz]", "Alpha band")

    # Add red vertical lines for significant time points (p < 0.05)
    if significant_times_alpha is not None:
        for sig_time in significant_times_alpha:
            plt.axvline(x=sig_time, color='red', linestyle='--', linewidth=1, alpha=0.3)

    # Plot beta band (High and Low Probability)
    plot_difference_with_ci(axs[1], times, beta_high, beta_low,
              len(beta_high), len(beta_low), "Beta Frequency Band [15-30 Hz]", "Beta band")

    # Add red vertical lines for significant time points (p < 0.05)
    if significant_times_beta is not None:
        for sig_time in significant_times_beta:
            plt.axvline(x=sig_time, color='red', linestyle='--', linewidth=1, alpha=0.3)

    # Calculate global y-limits based on both subplots
    alpha_ymin, alpha_ymax = axs[0].get_ylim()
    beta_ymin, beta_ymax = axs[1].get_ylim()
    global_ymin = min(alpha_ymin, beta_ymin)
    global_ymax = max(alpha_ymax, beta_ymax)

    # Set the same y-limits across both subplots
    axs[0].set_ylim(global_ymin, global_ymax)
    axs[1].set_ylim(global_ymin, global_ymax)

    # Set common axis properties
    set_common_axis_properties(axs[0], "Time (s)", "Power")
    set_common_axis_properties(axs[1], "Time (s)", "Power")

    figure3_savepath = Path(paths.figures.manuscript.figure3)

    # Determine the correct save path based on the condition and study
    file_base = f"{figure3_savepath}/fig3_tfr_freqbands_over_time_{ch_name}_{cond}"

    # Handle the study-specific part of the path
    if study == 'volatile':
        # Handle volatile case
        if cond == 'probability':
            save_path = f"{file_base}_{study}.svg"
        else:
            save_path = f"{file_base}_{conditions[idx_cond]}_{study}.svg"
    else:
        # Handle non-volatile case
        if cond == 'probability':
            save_path = f"{file_base}_{conditions[idx_cond]}_{study}.svg"
        else:
            save_path = f"{file_base}_{study}.svg"

    # Save both SVG and PNG versions of the figure
    svg_save_path = save_path
    png_save_path = save_path.replace(".svg", ".png")
    
    # save figure in this path
    # Assuming `fig` is your matplotlib figure object:
    fig.savefig(svg_save_path, format='svg')
    fig.savefig(png_save_path, format='png')

    print(f"Figure saved as SVG: {svg_save_path}")
    print(f"Figure saved as PNG: {png_save_path}")

    # Plot the label legend in the middle upper corner for the 'prev_resp' condition
    if cond == 'prev_resp':
        plt.legend(loc='upper center', ncol=2, fontsize=16)
    else:
        plt.legend(loc='lower right', ncol=2, fontsize=16)

    plt.tight_layout()
    plt.show()


def plot_tfr_cluster_test_output(
    cond: str = 'hitmiss',
    threed_test: bool = False,
    cond_a_name: str = 'hit',
    cond_b_name: str = 'miss',
    channel_names: list = ['CP4'],
    tmin: float = -0.7,
    tmax: float = 0.0,
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
    tfr_a_cond, tfr_b_cond, cond = load_tfr_conds()

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
            t_obs, _, cluster_p, _ = mne.stats.permutation_cluster_1samp_test(
                x, n_jobs=-1, n_permutations=10000, threshold=None, tail=0, adjacency=tfr_adjacency
            )

            print(f"smallest cluster p-value: {min(cluster_p)}")
            cluster_p = np.around(cluster_p, 2)

        else:
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
            t_obs, _, cluster_p, _ = mne.stats.permutation_cluster_1samp_test(
                x, n_jobs=-1, n_permutations=10000, threshold=threshold_tfce, tail=0
            )

            print(f"smallest cluster p-value: {min(cluster_p)}")
            cluster_p = np.around(cluster_p, 2)

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
                f"fig3_tfr_tvals_{cond_a_name}_{cond_b_name}_coolwarm_robust_samevminvmax_{channel_names[0]}.{fm}",
            ),
            dpi=300,
            format=fm,
        )
    plt.show()

    return "Done with cluster test"


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

    vmin_val = np.min(t_val)
    vmax_val = abs(vmin_val)

    # Plot tmap as heatmap, use the same vmin and vmax for all plots
    sns.heatmap(t_val, cbar=True, cmap="coolwarm", robust=True, ax=axs[idx], vmin=vmin_val, vmax=vmax_val)

    # Customize the font size and family for various elements
    plt.rcParams["font.family"] = "Arial"  # Set the font family for the entire plot to Arial
    plt.rcParams["font.size"] = 12  # Set the default font size to 12

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
    axs[idx].xaxis.label.set_fontsize(16)  # Adjust the font size for the xlabel
    axs[idx].yaxis.label.set_fontsize(16)  # Adjust the font size for the ylabel

    # Set custom the x and y-axis ticks and labels
    axs[idx].set_xticks(x_ticks)
    axs[idx].set_xticklabels(x_labels, rotation=0)
    axs[idx].set_yticks(y_ticks)
    axs[idx].set_yticklabels(y_labels, rotation=0)
    axs[idx].yaxis.labelpad = 15  # Adjust the distance between the y-axis label and the y-axis ticks
    axs[idx].xaxis.labelpad = 15  # Adjust the distance between the x-axis label and the y-axis ticks

    # Create a colorbar for the current subplot
    # cbar = axs[idx].collections[0].colorbar
    # Set the colorbar ticks to only include the minimum and maximum values of the data
    # min_val = np.ceil(min(map(min, t_val)))
    # max_val = np.ceil(max(map(max, t_val)))
    # cbar.set_ticks([min_val, max_val])

    return "Plotted and saved figure"


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
