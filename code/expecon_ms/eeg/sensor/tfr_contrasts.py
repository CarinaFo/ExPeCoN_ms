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

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from expecon_ms.configs import PROJECT_ROOT, config, params, path_to

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# %matplotlib qt  # for plots in new windows

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/sensor/tfr_contrasts.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# set font to Arial and font size to 14
plt.rcParams.update({"font.size": 14, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# set paths
# clean epochs
dir_clean_epochs = Path(path_to.data.eeg.preprocessed.ica.ICA)
dir_clean_epochs_expecon2 = Path(path_to.data.eeg.preprocessed.ica.clean_epochs_expecon2)

# save tfr solutions
tfr_path = Path(path_to.data.eeg.sensor.tfr)

# participant IDs
id_list_expecon1 = config.participants.ID_list_expecon1
id_list_expecon2 = config.participants.ID_list_expecon2

# pilot data counter
pilot_counter = config.participants.pilot_counter

# data_cleaning parameters defined in config.toml
rt_max = config.behavioral_cleaning.rt_max
rt_min = config.behavioral_cleaning.rt_min
hitrate_max = config.behavioral_cleaning.hitrate_max
hitrate_min = config.behavioral_cleaning.hitrate_min
farate_max = config.behavioral_cleaning.farate_max
hit_fa_diff = config.behavioral_cleaning.hit_fa_diff
# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def compute_tfr(
    study: int = 2,
    cond: str = "prev_resp",
    tmin: float = -1,
    tmax: float = 0,
    fmax: float = 35,
    fmin: float = 3,
    laplace: bool = False,
    induced: bool = False,
    mirror_data: bool = False,
    drop_bads: bool = True
):
    """
    Calculate the time-frequency representations per trial (induced power) 
    using multitaper method.
    Data is then saved in a tfr object per subject and stored to disk
    as a .h5 file.
    Args:
    ----
    study : int, info: which study to analyze: 1 (block, stable environment) or 2 (trial,
    volatile environment)
    cond : str, info: which condition to analyze: "probability" or "prev_resp"
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

    # store behavioral data
    df_all = []

    if study == 1:
        id_list = id_list_expecon1
        # load behavioral data
        data = pd.read_csv(Path(path_to.data.behavior, "prepro_behav_data.csv"))

    elif study == 2:
        id_list = id_list_expecon2
        # load behavioral data
        data = pd.read_csv(Path(path_to.data.behavior,
                                "prepro_behav_data_expecon2.csv"))
    else:
        raise ValueError("input should be 1 or 2 for the respective study")


    # now loop over participants
    for idx, subj in enumerate(id_list):

        # print participant ID
        print("Analyzing " + subj)

        # load clean epochs (after ica component rejection)
        if study == 1:
            epochs = mne.read_epochs(
                Path(dir_clean_epochs / f"P{subj}_icacorr_0.1Hz-epo.fif"))
        elif study == 2:
            # skip ID 13
            if subj == '013':
                continue
            epochs = mne.read_epochs(dir_clean_epochs_expecon2 / f"P{subj}_icacorr_0.1Hz-epo.fif")
            # rename columns
            epochs.metadata = epochs.metadata.rename(columns ={"resp1_t": "respt1",
                                                               "stim_type": "isyes",
                                                               "resp1": "sayyes"})
        else:
            raise ValueError("input should be 1 or 2 for the respective study")

        # clean epochs (remove blocks based on hit and false alarm rates, reaction times, etc.)
        epochs = drop_trials(data=epochs)

        # get behavioral data for current participant
        if study == 1:
            subj_data = data[idx + pilot_counter == data.ID]
        elif study == 2:
            subj_data = data[idx + 1 == data.ID]
        else:
            raise ValueError("input should be 1 or 2 for the respective study")

        # get drop log from epochs
        drop_log = epochs.drop_log

        # Ignored bad epochs are those defined by the user as bad epochs
        search_string = "IGNORED"
        # remove trials where epochs are labelled as too short
        indices = [index for index, tpl in enumerate(drop_log) if tpl and search_string not in tpl]

        # drop trials without corresponding epoch
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
            data = epochs.get_data()

            # zero pad = False = mirror the data on both ends
            data = zero_pad_or_mirror_data(data, zero_pad=False)

            # put back into epochs structure
            epochs = mne.EpochsArray(data, epochs.info, tmin=tmin * 2)

            # add metadata back
            epochs.metadata = metadata

        # subtract evoked response from each epoch
        if induced:
            epochs = epochs.subtract_evoked()

        # store behavioral data
        df_all.append(epochs.metadata)

        if cond == "probability":
            epochs_a = epochs[
                (epochs.metadata.cue == 0.75)
            ]
            epochs_b = epochs[(epochs.metadata.cue == 0.25)]
            cond_a = "high"
            cond_b = "low"
        elif cond == "prev_resp":
            epochs_a = epochs[
                (epochs.metadata.prevresp == 1)
            ]
            epochs_b = epochs[(epochs.metadata.prevresp == 0)]
            cond_a = "prevyesresp"
            cond_b = "prevnoresp"
        else:
            raise ValueError("input should be 'probability' or 'prev_resp'")

        # make sure we have an equal amount of trials in both conditions
        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        # set tfr path
        if (tfr_path / f"{subj}_{cond}_{str(study)}-tfr.h5").exists():
            print("TFR already exists")
        else:
            tfr_a = mne.time_frequency.tfr_multitaper(
                epochs_a, freqs=freqs, n_cycles=cycles, 
                return_itc=False, n_jobs=-1, average=True
            )

            tfr_b = mne.time_frequency.tfr_multitaper(
                epochs_b, freqs=freqs, n_cycles=cycles, 
                return_itc=False, n_jobs=-1, average=True
            )
            
            # save tfr data
            if cond == "probability":
                tfr_a.save(fname=tfr_path / f"{subj}_{cond_a}_{str(study)}-tfr.h5")
                tfr_b.save(fname=tfr_path / f"{subj}_{cond_b}_{str(study)}-tfr.h5")
            elif cond == "prev_resp":
                tfr_a.save(fname=tfr_path / f"{subj}_{cond_a}_{str(study)}-tfr.h5")
                tfr_b.save(fname=tfr_path / f"{subj}_{cond_b}_{str(study)}-tfr.h5")
            else:
                raise ValueError("input should be 'probability' or 'prev_resp'")

        # calculate tfr for all trials
        if (tfr_path / f"{subj}_single_trial_power_{str(study)}-tfr.h5").exists():
            print("TFR already exists")
        else:
            tfr = mne.time_frequency.tfr_multitaper(
                epochs, freqs=freqs, n_cycles=cycles, return_itc=False, 
                n_jobs=-1, average=False, decim=2
            )

            tfr.save(fname=tfr_path / f"{subj}_single_trial_power_{str(study)}-tfr.h5", 
                     overwrite=True)

    return "Done with tfr/erp computation", cond_a, cond_b


def load_tfr_conds(study: int = 1, 
                   cond_a: str = "high",
                   cond_b: str = "low",
                   mirror: bool = False):
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

    if study == 1:
        id_list = id_list_expecon1
    elif study == 2:
        id_list = id_list_expecon2
    else:
        raise ValueError("input should be 1 or 2 for the respective study")

    for subj in id_list:
        # load tfr data
        sfx = "_mirror" if mirror else ""
        tfr_a = mne.time_frequency.read_tfrs(fname=tfr_path / f"{subj}_{cond_a}_{str(study)}-tfr.h5", condition=0)
        tfr_b = mne.time_frequency.read_tfrs(fname=tfr_path / f"{subj}_{cond_b}_{str(study)}-tfr.h5", condition=0)
        tfr_a_all.append(tfr_a)
        tfr_b_all.append(tfr_b)

    return tfr_a_all, tfr_b_all


# TODO(simon): dont use mutables as default arguments
def plot_tfr_cluster_test_output(channel_names=["CP4"],
                                 fmin: int = 3,
                                 fmax: int = 35):
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
    #time_windows = [(-0.4, 0)]

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

        threshold_tfce = dict(start=0, step=0.05)

        # run cluster test over time and frequencies (no need to define adjacency)
        t_obs, clusters, cluster_p, h0 = mne.stats.permutation_cluster_1samp_test(
            x, n_jobs=-1, n_permutations=10000, threshold=threshold_tfce, tail=0,
        out_type='mask')

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
            t_obs=t_obs,
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

def plot_cluster_contours():

    """plot cluster permutation test output
    cluster is highlighted via a contour around it,
    code from Magdalena Gimpert"""

    fig = plt.figure()
    ax = sns.heatmap(t_obs, center=0,
                    cbar=True)
    # Draw the cluster outline
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j]:
                if i > 0 and not mask[i - 1, j]:
                    plt.plot([j - 0.5, j + 0.5], [i, i], color='black', linewidth=1)
                if i < mask.shape[1] - 1 and not mask[i + 1, j]:
                    plt.plot([j - 0.5, j + 0.5], [i + 1, i + 1], color='black', linewidth=1)
                if j > 0 and not mask[i, j - 1]:
                    plt.plot([j - 0.5, j - 0.5], [i, i + 1], color='black', linewidth=1)
                if j < mask.shape[2] - 1 and not mask[i, j + 1]:
                    plt.plot([j + 0.5, j + 0.5], [i, i + 1], color='black', linewidth=1)
    ax.invert_yaxis()
    ax.axvline([126], color='black', linestyle='dotted', linewidth=1) # stimulation onset
    ax.set(xlabel='Time (s)', ylabel='Frequency (Hz)',
        title=f'TFR contrast including significant cluster, {channel_name}')
    plt.show()

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


#### Helper functions


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
    data = data[data.metadata.respt1 != rt_max]
    data = data[data.metadata.respt1 > rt_min]

    # print rt trials dropped
    rt_trials_removed = before_rt_cleaning - len(data.metadata)

    print("Removed trials based on reaction time: ", rt_trials_removed)
    # Calculate hit rates per participant
    signal = data[data.metadata.isyes == 1]
    hitrate_per_subject = signal.metadata.groupby(['ID'])['sayyes'].mean()

    print(f"Mean hit rate: {np.mean(hitrate_per_subject):.2f}")

    # Calculate hit rates by participant and block
    hitrate_per_block = signal.metadata.groupby(['ID', 'block'])['sayyes'].mean()

    # remove blocks with hitrates > 90 % or < 20 %
    filtered_groups = hitrate_per_block[(hitrate_per_block > hitrate_max) | (hitrate_per_block < hitrate_min)]
    print('Blocks with hit rates > 0.9 or < 0.2: ', len(filtered_groups))

    # Extract the ID and block information from the filtered groups
    remove_hitrates = filtered_groups.reset_index()

    # Calculate false alarm rates by participant and block
    noise = data[data.metadata.isyes == 0]
    farate_per_block = noise.metadata.groupby(['ID', 'block'])['sayyes'].mean()

    # remove blocks with false alarm rates > 0.4
    filtered_groups = farate_per_block[farate_per_block > farate_max]
    print('Blocks with false alarm rates > 0.4: ', len(filtered_groups))

    # Extract the ID and block information from the filtered groups
    remove_farates = filtered_groups.reset_index()

    # Hitrate should be > false alarm rate
    filtered_groups = hitrate_per_block[hitrate_per_block - farate_per_block < hit_fa_diff]
    print('Blocks with hit rates < false alarm rates: ', len(filtered_groups))
          
    # Extract the ID and block information from the filtered groups
    hitvsfarate = filtered_groups.reset_index()

    # Concatenate the dataframes
    combined_df = pd.concat([remove_hitrates, remove_farates, hitvsfarate])

    # Remove duplicate rows based on 'ID' and 'block' columns
    unique_df = combined_df.drop_duplicates(subset=['ID', 'block'])

    # Merge the big dataframe with unique_df to retain only the non-matching rows
    data.metadata = data.metadata.merge(unique_df, on=['ID', 'block'], how='left',
                             indicator=True,
                             suffixes=('', '_y'))
    
    data = data[data.metadata['_merge'] == 'left_only']

    # Remove the '_merge' column
    data.metadata = data.metadata.drop('_merge', axis=1)

    return data