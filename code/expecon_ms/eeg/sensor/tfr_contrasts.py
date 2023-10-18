#!/usr/bin/python3
"""
Provide functions that compute time-frequency representations (TFRs).

Moreover, run a cluster test in 2D space (time and frequency)

This script produces figure 6.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023
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

from expecon_ms.behav import figure1  # TODO(simon): what is this imported for?
from expecon_ms.configs import PROJECT_ROOT, config, params, path_to

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# %matplotlib qt  # for plots in new windows

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/sensor/tfr_contrasts.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# set font to Arial and font size to 22  # TODO(simon): below is size=14
plt.rcParams.update({"font.size": 14, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# set paths
tfr_path = Path(path_to.data.eeg.sensor.tfr)

# participant IDs
id_list = config.participants.ID_list

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def compute_tfr(
    tmin: float = -0.4,
    tmax: float = 0,
    fmax: float = 35,
    fmin: float = 3,
    laplace: bool = False,
    induced: bool = False,
    mirror_data: bool = False,
    drop_bads: bool = True,
):
    """
    Calculate the time-frequency representations per trial (induced power) using multitaper method.

    Data is then saved in a tfr object per subject and stored to disk as a .h5 file.

    Args:
    ----
    tmin : float
    tmax : float
    fmin: float
    fmax: float
    laplace: boolean, info: apply current source density transform
    induced : boolean, info: subtract evoked response from each epoch
    mirror_data : boolean, info: mirror the data on both sides to avoid edge artifacts
    drop_bads : boolean: drop epochs with abnormal strong signal (> 200 micro-volts)

    Returns:
    -------
    None
    """
    # Define frequencies and cycles for multitaper method
    freqs = np.arange(fmin, fmax, 1)
    cycles = freqs / 4.0

    # store behavioral data and spectra
    df_all = []

    # now loop over participants
    for idx, subj in enumerate(id_list):
        # print participant ID
        print("Analyzing " + subj)

        # load cleaned epochs (after ica component rejection)

        epochs = mne.read_epochs(
            Path("./data/eeg/prepro_ica/clean_epochs_corr", f"P{subj}_epochs_after_ica_0.1Hzfilter-epo.fif")
        )

        # kick out blocks with too high or low hit-rates and false alarm rates
        # TODO(simon): consider to move the following to the config.toml
        ids_to_delete = [10, 12, 13, 18, 26, 30, 32, 32, 39, 40, 40, 30]
        blocks_to_delete = [6, 6, 4, 3, 4, 3, 2, 3, 3, 2, 5, 6]

        # Check if the participant ID is in the list of IDs to delete
        if pd.unique(epochs.metadata.ID) in ids_to_delete:
            # Get the corresponding blocks to delete for the current participant
            participant_blocks_to_delete = [
                block for id_, block in zip(ids_to_delete, blocks_to_delete) if id_ == pd.unique(epochs.metadata.ID)
            ]

            # Drop the rows with the specified blocks from the dataframe
            epochs = epochs[~epochs.metadata.block.isin(participant_blocks_to_delete)]

        # remove trials with rts >= 2.5 (no response trials) and trials with reaction times < 0.1
        epochs = epochs[epochs.metadata.respt1 >= 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # load behavioral data
        data = pd.read_csv(Path(path_to.data.behavior.behavior_df, "prepro_behav_data.csv"))

        # behavioral data for current participant
        subj_data = data[idx + 7 == data.ID]

        # get drop log from epochs
        drop_log = epochs.drop_log

        search_string = "IGNORED"

        indices = [index for index, tpl in enumerate(drop_log) if tpl and search_string not in tpl]

        # drop bad epochs (too late recordings)
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
        if mirror_data:
            metadata = epochs.metadata
            # zero pad the data on both sides to avoid leakage and edge artifacts
            data = epochs.get_data()

            data = zero_pad_or_mirror_data(data, zero_pad=False)

            # put back into epochs structure
            epochs = mne.EpochsArray(data, epochs.info, tmin=tmin * 2)

            # add metadata back
            epochs.metadata = metadata

        # subtract evoked response from each epoch
        if induced:
            epochs = epochs.subtract_evoked()

        # assign a sequential count for each row within each 'block' and 'subblock' group
        epochs.metadata["trial_count"] = epochs.metadata.groupby(["block", "subblock"]).cumcount()

        df_all.append(epochs.metadata)

        # create conditions
        epochs_a = epochs[
            (epochs.metadata.cue == 0.75)
        ]  # TODO(simon): consider to drop the values as params in config
        epochs_b = epochs[(epochs.metadata.cue == 0.25)]

        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        # set tfr path
        if (tfr_path / f"{subj}_high_mirror-tfr.h5").exists():
            print("TFR already exists")
        else:
            tfr_a = mne.time_frequency.tfr_multitaper(
                epochs_a, freqs=freqs, n_cycles=cycles, return_itc=False, n_jobs=-1, average=True
            )

            tfr_b = mne.time_frequency.tfr_multitaper(
                epochs_b, freqs=freqs, n_cycles=cycles, return_itc=False, n_jobs=-1, average=True
            )

            tfr_a.save(fname=tfr_path / f"{subj}_high_mirror-tfr.h5", overwrite=True)
            tfr_b.save(fname=tfr_path / f"{subj}_low_mirror-tfr.h5", overwrite=True)

        if (tfr_path / f"{subj}_single_trial_power-tfr.h5").exists():
            print("TFR already exists")
        else:
            tfr = mne.time_frequency.tfr_multitaper(
                epochs, freqs=freqs, n_cycles=cycles, return_itc=False, n_jobs=-1, average=False, decim=2
            )

            tfr.save(fname=tfr_path / f"{subj}_single_trial_power-tfr.h5", overwrite=True)

    return "Done with tfr/erp computation"


def load_tfr_conds(cond_a: str = "high", cond_b: str = "low", mirror: bool = True):
    """
    Load tfr data for the two conditions.

    Args:
    ----
    cond_a : str
        which condition tfr to load
    cond_b : str
        which condition tfr to load
    mirror : boolean
            whether to load mirrored data

    Returns:
    -------
    tfr_a_all: list: list of tfr objects for condition a
    tfr_b_all: list: list of tfr objects for condition b
    """
    tfr_a_all, tfr_b_all = [], []

    for subj in id_list:
        # load tfr data
        sfx = "_mirror" if mirror else ""
        tfr_a = mne.time_frequency.read_tfrs(fname=tfr_path / f"{subj}_{cond_a}{sfx}-tfr.h5", condition=0)
        tfr_b = mne.time_frequency.read_tfrs(fname=tfr_path / f"{subj}_{cond_b}{sfx}-tfr.h5", condition=0)
        tfr_a_all.append(tfr_a)
        tfr_b_all.append(tfr_b)

    return tfr_a_all, tfr_b_all


# TODO(simon): dont use mutables as default arguments
def plot_tfr_cluster_test_output(channel_names=["CP4"], fmin: int = 3, fmax: int = 35):
    """
    Plot cluster permutation test output for tfr data (time and frequency cluster).

    Args:
    ----
    channel_names : list of char
        channels to analyze
    fmin : int
        minimum frequency to plot
    fmax : int
        maximum frequency to plot
    ----------

    Returns:
    -------
    None
    """
    # load tfr data
    tfr_a_all, tfr_b_all = visualize_contrasts()  # TODO(simon): where is this function?

    # average over participants
    gra_a = mne.grand_average(tfr_a_all)  # high
    gra_b = mne.grand_average(tfr_b_all)  # low

    # difference between conditions (2nd level)
    diff = gra_a - gra_b
    diff.data = diff.data * 10**11

    # Create a 2x3 grid of plots (2 rows, 3 columns)
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))

    # which time windows to plot
    time_windows = [(-1, -0.4), (-0.4, 0)]
    # for mirrored data
    # time_windows = [(-0.4, 0)]

    # which axes to plot the time windows
    axes_first_row = [(0, 0), (0, 1)]
    axes_second_row = [(1, 0), (1, 1)]

    # now populate first row with tfr contrasts
    for t, a in zip(time_windows, axes_first_row):
        # TODO(simon): not used?!
        tfr_fig = (
            diff.copy()
            .crop(tmin=t[0], tmax=t[1])
            .plot(picks=channel_names, cmap=plt.cm.bwr, axes=axs[a[0], a[1]], show=False)[0]
        )

    # now plot the cluster permutation output in the second row
    ch_index = [tfr_a_all[0].ch_names.index(c) for c in channel_names]  # pick channels
    # TODO(simon): not used?!

    for t, a in zip(time_windows, axes_second_row):
        # contrast data
        x = np.array(
            [
                h.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                - l.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                for h, l in zip(tfr_a_all, tfr_b_all)
            ]
        )

        # pick channel
        x = np.mean(x, axis=1) if len(channel_names) > 1 else np.squeeze(x)
        print(x.shape)  # should be participants x frequencies x timepoints

        threshold_tfce = dict(start=0, step=0.1)

        # run cluster test over time and frequencies (no need to define adjacency)
        t_obs, clusters, cluster_p, h0 = mne.stats.permutation_cluster_1samp_test(
            x, n_jobs=-1, n_permutations=10000, threshold=threshold_tfce, tail=0
        )

        if len(cluster_p) > 0:
            print(f"The minimum p-value is {min(cluster_p)}")

            good_cluster_inds = np.where(cluster_p < params.alpha)

            if len(good_cluster_inds[0]) > 0:
                # Find the index of the overall minimum value
                min_index = np.unravel_index(np.argmin(t_obs), t_obs.shape)  # TODO(simon): not used?!

                freqs = np.arange(fmin, fmax, 1)  # TODO(simon): not used?!
                times = tfr_a_all[0].copy().crop(t[0], t[1]).times  # TODO(simon): not used?!

        # run function to plot significant cluster in time and frequency space
        plot_cluster_test_output(
            tobs=t_obs,
            cluster_p_values=cluster_p,
            clusters=clusters,
            fmin=fmin,
            fmax=fmax,
            data_cond=tfr_a_all,
            tmin=t[0],
            tmax=t[1],
            ax0=a[0],
            ax1=a[1],
        )

    # finally, save the figure
    for fm in ["svg", "png"]:
        # TODO(simon): cond_a & cond_b are not defined
        plt.savefig(
            Path(path_to.figures.manuscript.figure6) / f"fig6_{cond_a}_{cond_b}_tfr_{channel_names[0]}_mirrored.{fm}",
            dpi=300,
            format=fm,
        )


def plot_cluster_test_output(
    t_obs=None, cluster_p_values=None, clusters=None, fmin=7, fmax=35, data_cond=None, tmin=0, tmax=0.5, ax0=0, ax1=0
) -> None:
    """Plot cluster."""
    freqs = np.arange(fmin, fmax, 1)

    times = 1e3 * data_cond[0].copy().crop(tmin, tmax).times

    t_obs_plot = np.nan * np.ones_like(t_obs)
    for c, p_val in zip(clusters, cluster_p_values):
        if p_val <= params.alpha:
            t_obs_plot[c] = t_obs[c]

    min_v = np.max(np.abs(t_obs))

    vmin = -min_v
    vmax = min_v

    # TODO(simon): fig1 not used & axs not defined
    fig1 = axs[ax0, ax1].imshow(
        t_obs,
        cmap=plt.cm.bwr,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        alpha=0.5,
    )

    fig2 = axs[ax0, ax1].imshow(
        t_obs_plot,
        cmap=plt.cm.bwr,
        extent=[times[0], times[-1], freqs[0], freqs[-1]],
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
    )

    # Add colorbar for each subplot
    # TODO(simon): fig not defined
    fig.colorbar(fig2, ax=axs[ax0, ax1])

    axs[ax0, ax1].set_xlabel("Time (ms)")
    axs[ax0, ax1].set_ylabel("Frequency (Hz)")


def zero_pad_or_mirror_data(data, zero_pad: bool = False):
    """
    Zero-pad or mirror data on both sides to avoid edge artifacts.

    Args:
    ----
    data: data array with the structure epochs x channels x time

    Returns:
    -------
    array with mirrored data on both ends = data.shape[2]
    """
    if zero_pad:
        padded_list = []

        zero_pad = np.zeros(data.shape[2])
        # loop over epochs
        for epoch in range(data.shape[0]):
            ch_list = []
            # loop over channels
            for ch in range(data.shape[1]):
                # zero padded data at beginning and end
                ch_list.append(np.concatenate([zero_pad, data[epoch][ch], zero_pad]))
            padded_list.append(list(ch_list))
    else:
        padded_list = []

        # loop over epochs
        for epoch in range(data.shape[0]):
            ch_list = []
            # loop over channels
            for ch in range(data.shape[1]):
                # mirror data at beginning and end
                ch_list.append(np.concatenate([data[epoch][ch][::-1], data[epoch][ch], data[epoch][ch][::-1]]))
            padded_list.append(list(ch_list))

    return np.squeeze(np.array(padded_list))


def permutate_trials(n_permutations: int = 500, power_a=None, power_b=None) -> None:
    """Permutate trials between two conditions and equalize trial counts."""
    # store permutations
    power_a_perm, power_b_perm = [], []

    for i in range(n_permutations):
        if power_b.data.shape[0] > power_a.data.shape[0]:
            random_sample = power_a.data.shape[0]
            idx_list = list(range(power_b.data.shape[0]))

            power_b_idx = random.sample(idx_list, random_sample)

            power_b.data = power_b.data[power_b_idx, :, :, :]
        else:
            random_sample = power_b.data.shape[0]
            idx_list = list(range(power_a.data.shape[0]))

            power_a_idx = random.sample(idx_list, random_sample)

            power_a.data = power_a.data[power_a_idx, :, :, :]

        evoked_power_a = power_a.average()
        evoked_power_b = power_b.average()

        power_a_perm.append(evoked_power_a)
        power_b_perm.append(evoked_power_b)

    # average across permutations
    evoked_power_a_perm_arr = np.mean(np.array([p.data for p in power_a_perm]), axis=0)
    evoked_power_b_perm_arr = np.mean(np.array([p.data for p in power_b_perm]), axis=0)

    # put back into the TFR object
    # TODO(simon): power not defined, evoked_power_a evoked_power_b are not used
    evoked_power_a = mne.time_frequency.AverageTFR(
        power.info, evoked_power_a_perm_arr, power.times, power.freqs, power_a.data.shape[0]
    )
    evoked_power_b = mne.time_frequency.AverageTFR(
        power.info, evoked_power_b_perm_arr, power.times, power.freqs, power_b.data.shape[0]
    )
