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
    laplace: bool,
    induced: bool,
    mirror: bool,
    drop_bads: bool,
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
        else:
            raise ValueError("input should be 'probability' or 'prev_resp'")

        # make sure we have an equal number of trials in both conditions
        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        n_epochs = len(epochs_a.events)
        n_ave.append(n_epochs)

        # set tfr path
        if (Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{study!s}_mirror-tfr.h5")).exists():
            print("TFR already exists")
        else:
            tfr_a = mne.time_frequency.tfr_multitaper(
                epochs_a, freqs=freqs, n_cycles=cycles, return_itc=False, n_jobs=-1, average=True
            )

            tfr_b = mne.time_frequency.tfr_multitaper(
                epochs_b, freqs=freqs, n_cycles=cycles, return_itc=False, n_jobs=-1, average=True
            )

            # save tfr data
            tfr_a.save(fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_a_name}_{study!s}_mirror-tfr.h5"))
            tfr_b.save(fname=Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"{subj}_{cond_b_name}_{study!s}_mirror-tfr.h5"))

    # save number of epochs per participant as csv file
    pd.DataFrame(n_ave).to_csv(Path(paths.data.eeg.sensor.tfr.tfr_contrasts, f"n_ave_{cond_a_name}_{study!s}.csv"))

    return "Done with tfr/erp computation", cond_a_name, cond_b_name


def load_tfr_conds(
    studies: list, cond: str, cond_a_name: str, cond_b_name: str, cond_a_names: list, cond_b_names: list
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

        tfr_a_cond.append(tfr_a_all)
        tfr_b_cond.append(tfr_b_all)

    return tfr_a_cond, tfr_b_cond


def calc_freq_band_power_over_time(ch_name: str = 'CP4', cond: str = 'probability'):
    """
    Plot the difference in power between high and low probability conditions for the alpha and beta bands.
    
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
    
    def load_and_compute_power(tfr_a_cond, tfr_b_cond, ch_index, freq_band, study):
        """
        Load the data and compute power difference and SEM for the given frequency band.
        
        Args:
        -----
        tfr_a_cond : list
            TFR data for condition A.
        tfr_b_cond : list
            TFR data for condition B.
        ch_index : int
            Index of the channel to analyze.
        freq_band : tuple
            Frequency band for power computation (start_freq, end_freq).
        study : str
            Study name for labeling.
        
        Returns:
        --------
        mean_diff : np.array
            Mean difference in power for the given frequency band.
        sem_diff : np.array
            Standard error of the mean for the power difference.
        """
        start_freq, end_freq = freq_band
        
        if study == 'stable':
            # Compute band power difference for all participants
            high_power = [scipy.integrate.trapezoid(a[idx_cond].data[:, start_freq:end_freq, :], axis=1) for a in tfr_a_cond]
            low_power = [scipy.integrate.trapezoid(a[idx_cond].data[:, start_freq:end_freq, :], axis=1) for a in tfr_b_cond]
        else:
            # Compute band power difference for all participants
            high_power = [scipy.integrate.trapezoid(a.data[:, start_freq:end_freq, :], axis=1) for a in tfr_a_cond]
            low_power = [scipy.integrate.trapezoid(a.data[:, start_freq:end_freq, :], axis=1) for a in tfr_b_cond]
        
        # Calculate the difference and SEM
        power_diff = [h - l for h, l in zip(high_power, low_power)]
        sem_diff = scipy.stats.sem(np.array(power_diff), axis=0)
        mean_diff = np.mean(np.array(power_diff), axis=0)
        
        return np.array(power_diff), mean_diff, sem_diff
    
    def perform_stat_tests(power_diff, ch_index):
        """
        Perform statistical tests on power difference and return significant timepoints.
        
        Args:
        -----
        power_diff : np.array
            Power difference across participants for a given frequency band.
        ch_index : int
            Index of the channel.
        
        Returns:
        --------
        sign_timepoints : np.array
            Significant timepoints where the p-value is below 0.05.
        """
        power_ch = power_diff[:, ch_index, :]
        t_vals, p_vals, _ = mne.stats.permutation_t_test(power_ch)
        return times[np.where(p_vals < 0.05)]
    
    if cond == "probability":
        # Load data for probability condition
        tfr_a_cond, tfr_b_cond = load_tfr_conds(
            studies=[1, 2],
            cond="probability",
            cond_a_name="high_-0.7_0_induced",
            cond_b_name="low_-0.7_0_induced",
            cond_a_names=["high_prevhit_-0.7_0_induced", "high_prevmiss_-0.7_0_induced", "high_prevcr_-0.7_0_induced"],
            cond_b_names=["low_prevhit_-0.7_0_induced", "low_prevmiss_-0.7_0_induced", "low_prevcr_-0.7_0_induced"],
        )
        cond_name = 'high minus low'
        cond = 'prob'
    else:
        # Load data for previous response condition
        tfr_a_cond, tfr_b_cond = load_tfr_conds(
            studies=[1, 2],
            cond="prev_resp",
            cond_a_name="prevyesresp_highprob_prevstim_-0.7_0_induced",
            cond_b_name="prevnoresp_highprob_prevstim_-0.7_0_induced",
            cond_a_names=["prevyesresp_samecue_lowprob_-0.7_0_induced", "prevyesresp_samecue_highprob_-0.7_0_induced"],
            cond_b_names=["prevnoresp_samecue_lowprob_-0.7_0_induced", "prevnoresp_samecue_highprob_-0.7_0_induced"],
        )
        cond_name = 'previous yes minus previous no response'
        cond = 'prev_resp'

    times = tfr_a_cond[1][0].copy().times
    ch_index = tfr_a_cond[1][0].ch_names.index(ch_name)
    
    for idx, (cond_a, cond_b) in enumerate(zip(tfr_a_cond, tfr_b_cond)):
        if idx == 1:
            conditions=None
            study = 'Volatile'

            # Alpha and Beta power computation
            beta_diff, beta_mean, beta_sem = load_and_compute_power(tfr_a_cond[idx], tfr_b_cond[idx], ch_index, (12, 28), 'volatile')
            alpha_diff, alpha_mean, alpha_sem = load_and_compute_power(tfr_a_cond[idx], tfr_b_cond[idx], ch_index, (8, 12), 'volatile')

            # Perform stats
            sign_timepoints_beta = perform_stat_tests(beta_diff, ch_index)
            sign_timepoints_alpha = perform_stat_tests(alpha_diff, ch_index)

            # Plot results
            plot_freq_band_over_time(
                ch_index, alpha_mean, alpha_sem, beta_mean, beta_sem,
                sign_timepoints_alpha, sign_timepoints_beta, times, cond, study, conditions, idx_cond
            )

        else:
            conditions = ['previous hit', 'previous miss', 'previous correct rejection']
            study = 'Stable'

            for idx_cond in range(len(conditions)):
                # Alpha and Beta power computation
                beta_diff, beta_mean, beta_sem = load_and_compute_power(tfr_a_cond[idx], tfr_b_cond[idx], ch_index, (12, 28), 'stable')
                alpha_diff, alpha_mean, alpha_sem = load_and_compute_power(tfr_a_cond[idx], tfr_b_cond[idx], ch_index, (8, 12), 'stable')

                # Perform stats
                sign_timepoints_beta = perform_stat_tests(beta_diff, ch_index)
                sign_timepoints_alpha = perform_stat_tests(alpha_diff, ch_index)

                # Plot results
                plot_freq_band_over_time(
                    ch_index, alpha_mean, alpha_sem, beta_mean, beta_sem,
                    sign_timepoints_alpha, sign_timepoints_beta, times, cond, study, conditions, idx_cond=idx_cond
                )
    
    return "Done plotting freq bands over time"


def plot_freq_band_over_time(ch_name, alpha_mean, alpha_sem, beta_mean, 
                             beta_sem, sign_timepoints_alpha, sign_timepoints_beta, 
                             times, cond, study, conditions, idx_cond: int = 0):
    """
    Plot the difference in power between high and low probability conditions for the alpha and beta bands.

    Args:
    ----
    ch_name : str
        Name of the channel to plot.
    alpha_mean : np.array
        Mean power difference for the alpha band.
    alpha_sem : np.array
        Standard error of the mean for the alpha band.
    beta_mean : np.array
        Mean power difference for the beta band.
    beta_sem : np.array
        Standard error of the mean for the beta band.
    sign_timepoints_alpha : np.array
        Significant timepoints for the alpha band.
    sign_timepoints_beta : np.array
        Significant timepoints for the beta band.
    times : np.array
        Time points for the x-axis.
    cond : str
        Condition name for labeling.
    study : str
        Study name for labeling.
    conditions : list of str
        List of conditions for the previous response condition.

    Returns:
    -------
    None
    """
    
    def plot_band(ax, times, mean_vals, sem_vals, title, band_label, color='black'):
        """Helper function to plot a frequency band with shaded error."""
        ax.plot(times, mean_vals, color=color, label=band_label)
        ax.fill_between(times,
                        mean_vals - sem_vals,
                        mean_vals + sem_vals,
                        color=color, alpha=0.3)
        ax.set_title(title, fontsize=20)

    def set_common_axis_properties(ax, x_label, y_label):
        """Set common axis properties for both subplots."""
        ax.set_xlabel(x_label, fontsize=20)
        ax.set_ylabel(y_label, fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_xlim(-0.7, 0)

    # Create the figure and subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))
    if study == 'Volatile':
        fig_title = f"{study.capitalize()} environment: Difference Amplitude {cond_name}"
    else:
        fig_title = f"{study.capitalize()} environment: Difference Amplitude {cond_name}\n{conditions[idx_cond]}"
    fig.suptitle(fig_title, fontsize=24, y=1.05)

    # Plot alpha and beta bands
    plot_band(axs[0], times, alpha_mean[ch_name, :], alpha_sem[ch_name, :], "Alpha Frequency Band [8-12 Hz]", "Alpha band")
    plot_band(axs[1], times, beta_mean[ch_name, :], beta_sem[ch_name, :], "Beta Frequency Band [15-30 Hz]", "Beta band")

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

    # Mark significant timepoints after y-limits are set
    if len(sign_timepoints_alpha) > 0:
        min_time_alpha, max_time_alpha = min(sign_timepoints_alpha), max(sign_timepoints_alpha)
        axs[0].plot([min_time_alpha, max_time_alpha], [global_ymin, global_ymin], color="red", linestyle="--", linewidth=8)

    if len(sign_timepoints_beta) > 0:
        min_time_beta, max_time_beta = min(sign_timepoints_beta), max(sign_timepoints_beta)
        axs[1].plot([min_time_beta, max_time_beta], [global_ymin, global_ymin], color="red", linestyle="--", linewidth=8)

    # Save the figure
    if study == 'Volatile':
        save_path = f"fig3_tfr_freqbands_over_time_{ch_name}_robust_samevminvmax_{cond}_{study}.svg"
    else:
        save_path = f"fig3_tfr_freqbands_over_time_{ch_name}_robust_samevminvmax_{cond}_{conditions[idx_cond]}_{study}.svg"
    fig.savefig(Path(paths.figures.manuscript.figure3, save_path), dpi=300, format="svg")

    plt.show()



def plot_tfr_cluster_test_output(
    cond: str,
    tfr_a_cond: list,
    tfr_b_cond: list,
    threed_test: bool,
    cond_a_name: str,
    cond_b_name: str,
    channel_names: list,
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
    # Create a 2x3 grid of plots (2 rows, 3 columns)
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 6))

    # Adjust the space between the subplots
    plt.subplots_adjust(wspace=0.7)
    # which time windows to plot
    time_windows = [(-0.7, 0.0), (-0.7, 0.0)]

    # loop over time windows and axes
    for idx, t in enumerate(time_windows):
        if threed_test:
            # contrast data
            x = np.array([
                h.copy().crop(tmin=t[0], tmax=t[1]).data - l_.copy().crop(tmin=t[0], tmax=t[1]).data
                for h, l_ in zip(tfr_a_cond[idx], tfr_b_cond[idx])
            ])
            # define adjacency for channels
            ch_adjacency, _ = mne.channels.find_ch_adjacency(tfr_a_cond[idx][0].info, ch_type="eeg")

            # visualize channel adjacency
            mne.viz.plot_ch_adjacency(tfr_a_cond[0][0].info, ch_adjacency, tfr_a_cond[0][0].ch_names)
            # our data at each observation is of shape frequencies x times x channels
            tfr_adjacency = mne.stats.combine_adjacency(
                len(tfr_a_cond[idx][0].freqs), len(tfr_b_cond[idx][0].crop(tmin=t[0], tmax=t[1]).times), ch_adjacency
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

    res = stats.ttest_1samp(data, popmean=0)

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
