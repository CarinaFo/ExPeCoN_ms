#!/usr/bin/python3
"""
Provide functions that compute time-frequency representations (TFRs).

Moreover, run a cluster test in 2D space (time and frequency)

This script produces figure 4.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023
"""
# %% Import
from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import mne
import numpy as np
import pandas as pd
import random
import scipy.stats as stats

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

# clean epochs
dir_clean_epochs_expecon1 = Path(path_to.data.eeg.preprocessed.ica.clean_epochs_expecon1)
dir_clean_epochs_expecon2 = Path(path_to.data.eeg.preprocessed.ica.clean_epochs_expecon2)

# save tfr solutions
tfr_path = Path(path_to.data.eeg.sensor.tfr.tfr_contrasts)
tfr_path.mkdir(parents=True, exist_ok=True)

figure4_path = Path(path_to.figures.manuscript.figure4)
figure4_path.mkdir(parents=True, exist_ok=True)

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
    study: int,
    cond: str,
    tmin: float,
    tmax: float,
    fmin: float,
    fmax: float,
    laplace: bool,
    induced: bool,
    mirror: bool,
    drop_bads: bool
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
    # store epoch count per participant
    n_ave = []

    if study == 1:
        id_list = id_list_expecon1
        # load behavioral data
        data = pd.read_csv(Path(path_to.data.behavior, "prepro_behav_data_expecon1.csv"))

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
                Path(dir_clean_epochs_expecon1 / f"P{subj}_icacorr_0.1Hz-epo.fif"))
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
        if mirror:
            metadata = epochs.metadata

            epoch_data = epochs.get_data()

            # zero pad = False = mirror the data on both ends
            data_mirror = zero_pad_or_mirror_data(epoch_data, zero_pad=False)

            # put back into epochs structure
            epochs = mne.EpochsArray(data_mirror, epochs.info, tmin=tmin * 2)

            # add metadata back
            epochs.metadata = metadata

        # subtract evoked response from each epoch
        if induced:
            epochs = epochs.subtract_evoked()

        # store behavioral data
        df_all.append(epochs.metadata)

        if cond == "probability":
            epochs_a = epochs[(epochs.metadata.cue == 0.75)]
            epochs_b = epochs[(epochs.metadata.cue == 0.25)]
            cond_a_name = "high"
            cond_b_name = "low"

            if mirror:
                cond_a_name = f'{cond_a_name}_mirror'
                cond_b_name = f'{cond_b_name}_mirror'

        elif cond == "prev_resp":
            epochs_a = epochs[((epochs.metadata.prevresp == 1) &
                               (epochs.metadata.previsyes == 1) &
                               (epochs.metadata.cue == 0.75))]
            epochs_b = epochs[((epochs.metadata.prevresp == 0) &
                               (epochs.metadata.previsyes == 1) &
                                (epochs.metadata.cue == 0.75))]
            cond_a_name = "prevyesresp_highprob_stim"
            cond_b_name = "prevnoresp_highprob_stim"

            if mirror:
                cond_a_name = f'{cond_a_name}_mirror'
                cond_b_name = f'{cond_b_name}_mirror'

        else:
            raise ValueError("input should be 'probability' or 'prev_resp'")

        # make sure we have an equal amount of trials in both conditions
        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        n_epochs = len(epochs_a.events)
        n_ave.append(n_epochs)

        # set tfr path
        if (tfr_path / f"{subj}_{cond_a_name}_{str(study)}-tfr.h5").exists():
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
            tfr_a.save(fname=tfr_path / f"{subj}_{cond_a_name}_{str(study)}-tfr.h5")
            tfr_b.save(fname=tfr_path / f"{subj}_{cond_b_name}_{str(study)}-tfr.h5")

        # calculate tfr for all trials
        if (tfr_path / f"{subj}_single_trial_power_mirror_{str(study)}-tfr.h5").exists():
            print("TFR already exists")
        else:
            tfr = mne.time_frequency.tfr_multitaper(
                epochs, freqs=freqs, n_cycles=cycles, return_itc=False,
                n_jobs=-1, average=False, decim=2
            )
            if mirror:
                tfr.save(fname=tfr_path / f"{subj}_single_trial_power_mirror_{str(study)}-tfr.h5")
            else:
                tfr.save(fname=tfr_path / f"{subj}_single_trial_power_{str(study)}-tfr.h5", 
                        overwrite=True)

    # save number of epochs per participant as csv file
    pd.DataFrame(n_ave).to_csv(f'{tfr_path}{Path("/")}n_ave_{cond_a_name}_{str(study)}.csv')

    return "Done with tfr/erp computation", cond_a_name, cond_b_name


def load_tfr_conds(studies: list,
                   cond: str,
                   cond_a_name: str,
                   cond_b_name: str,
                   cond_a_names: list,
                   cond_b_names: list):
    """
    Load tfr data for the two conditions.

    Args:
    ----
    study : list, info: load data from one study or both
    cond : str, info: which condition to analyze: "probability" or "prev_resp"
    cond_a : str
        which condition tfr to load: high or low or prevyesresp or prevnoresp
    cond_b : str
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

    tfr_a_cond, tfr_b_cond = [], [] # store condition data

    for study in studies:
        tfr_a_all, tfr_b_all = [], [] # store participant data

        # adjust id list
        if study == 1:
            id_list = id_list_expecon1
        elif study == 2:
            id_list = id_list_expecon2
        else:
            raise ValueError("input should be 1 or 2 for the respective study")
            # now load data for each participant and each condition
        if cond == 'probability':
            if study == 1:
                for subj in id_list:
                    tfr_a_all_conds, tfr_b_all_conds = [], [] # store the conditions
                    for c_a, c_b in zip(cond_a_names, cond_b_names):
                        # load tfr data
                        tfr_a = mne.time_frequency.read_tfrs(fname=tfr_path / f"{subj}_{c_a}_{str(study)}-tfr.h5",
                                                            condition=0)
                        tfr_b = mne.time_frequency.read_tfrs(fname=tfr_path / f"{subj}_{c_b}_{str(study)}-tfr.h5",
                                                            condition=0)
                        tfr_a_all_conds.append(tfr_a)
                        tfr_b_all_conds.append(tfr_b)
                    tfr_a_all.append(tfr_a_all_conds)
                    tfr_b_all.append(tfr_b_all_conds)
            elif study == 2:
                for subj in id_list:
                    if subj == '013':
                        continue
                    # load tfr data
                    tfr_a = mne.time_frequency.read_tfrs(fname=tfr_path / f"{subj}_{cond_a_name}_{str(study)}-tfr.h5",
                                                        condition=0)
                    tfr_b = mne.time_frequency.read_tfrs(fname=tfr_path / f"{subj}_{cond_b_name}_{str(study)}-tfr.h5",
                                                        condition=0)
                    tfr_a_all.append(tfr_a)
                    tfr_b_all.append(tfr_b)

        elif cond == 'prev_resp':
            for subj in id_list:
                if study == 2 and subj == '013':
                    continue
                # load tfr data
                tfr_a = mne.time_frequency.read_tfrs(fname=tfr_path / f"{subj}_{cond_a_name}_{str(study)}-tfr.h5",
                                                    condition=0)
                tfr_b = mne.time_frequency.read_tfrs(fname=tfr_path / f"{subj}_{cond_b_name}_{str(study)}-tfr.h5",
                                                    condition=0)
                tfr_a_all.append(tfr_a)
                tfr_b_all.append(tfr_b)

        tfr_a_cond.append(tfr_a_all)
        tfr_b_cond.append(tfr_b_all)

    return tfr_a_cond, tfr_b_cond


def plot_tfr_cluster_test_output(cond: str, 
                                 tfr_a_cond: list,
                                 tfr_b_cond: list,
                                 threed_test: bool,
                                 cond_a_name: str,
                                 cond_b_name: str,
                                 channel_names: list):
    """
    Plot cluster permutation test output for tfr data (time and frequency cluster).

    Args:
    ----
    cond : str, info: which condition to analyze: "probability" or "prev_resp"
    data_a : list of tfr objects
        tfr data for condition a
    data_b : list of tfr objects
        tfr data for condition b
    3d_test: boolean, info: whether to run a 3d cluster test (time, frequency, channels)
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
    time_windows = [(-0.4, 0.0), (-0.4, 0.0)]

    # loop over time windows and axes
    for idx, t in enumerate(time_windows):

        if threed_test:
            # contrast data
            x = np.array(
                [
                    h.copy().crop(tmin=t[0], tmax=t[1]).data
                    - l.copy().crop(tmin=t[0], tmax=t[1]).data
                    for h, l in zip(tfr_a_cond[idx], tfr_b_cond[idx])
                ]
            )
            # define adjacency for channels
            ch_adjacency, _ = mne.channels.find_ch_adjacency(tfr_a_cond[idx][0].info, ch_type="eeg")

            # visualize channel adjacency
            mne.viz.plot_ch_adjacency(tfr_a_cond[0][0].info, ch_adjacency, tfr_a_cond[0][0].ch_names)
            # our data at each observation is of shape frequencies × times × channels
            tfr_adjacency = mne.stats.combine_adjacency(len(tfr_a_cond[idx][0].freqs), 
                                                        len(tfr_b_cond[idx][0].
                                                            crop(tmin=t[0], tmax=t[1]).times),
                                                         ch_adjacency)

            x = x.reshape(x.shape[0], x.shape[2], x.shape[3], x.shape[1])
            print(f"Shape of array for cluster test should be participants x frequencies x timepoints x channels: {x.shape}")

            # run cluster test over time, frequencies and channels
            t_obs, _ , cluster_p, _ = mne.stats.permutation_cluster_1samp_test(
                x, n_jobs=-1, n_permutations=10000, threshold=None,
                tail=0, adjacency=tfr_adjacency)

            print(f'smallest cluster p-value: {min(cluster_p)}')
            cluster_p = np.around(cluster_p, 2)
        else:
            if cond == 'probability':
                if idx == 0:
                    tfr_a = np.array(tfr_a_cond[idx])
                    tfr_b = np.array(tfr_b_cond[idx])
                    x_conds = []
                    for conds in range(tfr_a.shape[1]-1): # exclude prevfa
                        # contrast data
                        x = np.array(
                            [
                                h.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                                - l.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                                for h, l in zip(tfr_a[:, conds], tfr_b[:, conds])
                            ]
                        )
                        x_conds.append(x)

                    # average across conditions (get rid of previous trial response)
                    x = np.mean(np.array(list(x_conds)), axis=0)  # should be shape participants x frequencies x timepoints
                else:
                    # contrast data
                    x = np.array(
                        [
                            h.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                            - l.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                            for h, l in zip(tfr_a_cond[idx], tfr_b_cond[idx])
                        ]
                    )
            elif cond == 'prev_resp':
                # contrast data
                x = np.array(
                        [
                            h.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                            - l.copy().crop(tmin=t[0], tmax=t[1]).pick_channels(channel_names).data
                            for h, l in zip(tfr_a_cond[idx], tfr_b_cond[idx])
                        ]
                )
            
            # pick channel or average over channels
            x = np.mean(x, axis=1) if len(channel_names) > 1 else np.squeeze(x)
            print(f"Shape of array for cluster test should be participants x frequencies x timepoints: {x.shape}")

            # define threshold dictionary for threshold-free cluster enhancement
            threshold_tfce = dict(start=0, step=0.1)

            # run cluster test over time and frequencies (no need to define adjacency)
            t_obs, _, cluster_p, _ = mne.stats.permutation_cluster_1samp_test(
                x, n_jobs=-1, n_permutations=10000, threshold=threshold_tfce,
                tail=0)

            print(f'smallest cluster p-value: {min(cluster_p)}')
            cluster_p = np.around(cluster_p, 2)

            # set mask for plotting sign. points that belong to the cluster
            mask = np.array(cluster_p).reshape(t_obs.shape[0], t_obs.shape[1])
            mask = mask < params.alpha  # boolean for masking sign. voxels

            # plot t contrast and sign. cluster contour
            plot_cluster_contours(data=x, t_obs=t_obs,
                                  tfr_a_cond=tfr_a_cond,
                                  fmin=3, fmax=35,
                                  idx=idx, t=t,
                                  axs=axs, mask=mask)

    plt.tight_layout()
    # now save the figure to disk as png and svg
    for fm in ["svg", "png"]:
        fig.savefig(figure4_path / f"fig4_tfr_tvals_{cond_a_name}_"
            f"{cond_b_name}_{channel_names[0]}.{fm}",
            dpi=300,
            format=fm,
        )
    plt.show()

    return t_obs, cluster_p


def plot_cluster_contours(data: np.ndarray,
                          t_obs: np.ndarray,
                          tfr_a_cond: np.ndarray,
                          fmin: float,
                          fmax: float,
                          idx: int,
                          t: float,
                          axs: tuple,
                          mask: np.ndarray):
    """Plot cluster permutation test output.

    Cluster is highlighted via a contour around it, code adapted Gimpert et al.
    x and p ticks and labels, as well as colorbar are not plotted

    Args:
    ----
    data: array, info: data to plot in heatmap
    fmin: float, info: minimum frequency to plot
    fmax: float, info: maximum frequency to plot
    """
    # define timepoints and frequencies
    times = tfr_a_cond[1][5].copy().crop(tmin=t[0], tmax=t[1]).times

    freqs = np.arange(fmin, fmax, 1)

    # set custom x labels and ticks (timepoints)
    x_ticks, x_labels = [], []

    # Iterate through the list and select every 3rd entry starting from the first entry
    for x_idx, x_value in enumerate(times):
        if x_idx % 25 == 0:
            x_ticks.append(x_idx)
            x_labels.append(x_value)

    # set custom y labels and ticks (frequencies)
    y_ticks, y_labels = [], []

    # Iterate through the list and select every 3rd entry starting from the first entry
    for y_idx, y_value in enumerate(freqs):
        if y_idx % 3 == 0:
            y_ticks.append(y_idx)
            y_labels.append(y_value)

    res = stats.ttest_1samp(data, popmean=0)

    t_val = np.squeeze(res[0])

    # plot tmap as heatmap
    sns.heatmap(t_val,
                cbar=True,
                cmap="viridis",
                ax=axs[idx])

    # Draw the cluster outline
    for i in range(mask.shape[0]): # frequencies
        for j in range(mask.shape[1]): # time
            if mask[i, j]:
                if i > 0 and not mask[i - 1, j]:
                    axs[idx].plot([j - 0.5, j + 0.5], [i, i], color='white', linewidth=2)
                if i < mask.shape[0] - 1 and not mask[i + 1, j]:
                    axs[idx].plot([j - 0.5, j + 0.5], [i + 1, i + 1], color='white', linewidth=2)
                if j > 0 and not mask[i, j - 1]:
                    axs[idx].plot([j - 0.5, j - 0.5], [i, i + 1], color='white', linewidth=2)
                if j < mask.shape[1] - 1 and not mask[i, j + 1]:
                    axs[idx].plot([j + 0.5, j + 0.5], [i, i + 1], color='white', linewidth=2)
    
    axs[idx].invert_yaxis()
    axs[idx].axvline([t_obs.shape[1]], color='white', 
                     linestyle='dotted', linewidth=5) # stimulation onset
    axs[idx].set(xlabel='Time (s)', ylabel='Frequency (Hz)')
    # Set the font size for xlabel and ylabel
    axs[idx].xaxis.label.set_fontsize(20)  # Adjust the font size for the xlabel
    axs[idx].yaxis.label.set_fontsize(20)  # Adjust the font size for the ylabel

    # Set custom the x and y axis ticks and labels
    axs[idx].set_xticks(x_ticks)
    axs[idx].set_xticklabels(x_labels, fontsize=20)
    axs[idx].set_yticks(y_ticks)
    axs[idx].set_yticklabels(y_labels, fontsize=20)

    return "Plotted and saved figure"

def zero_pad_or_mirror_data(data, zero_pad: bool):
    """
    Zero-pad or mirror data on both sides to avoid edge artifacts.

    Args:
    ----
    data: data array with the structure epochs x channels x time
    zero_pad: boolean, info: whether to zero pad or mirror the data
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


def permute_trials(n_permutations: float, power_a: None, power_b: None):

    """Permute trials between two conditions and equalize trial counts.
    Args:
    ----
    n_permutations: int, info: number of permutations
    power_a: mne.time_frequency.EpochsTFR, info: tfr data for condition a
    power_b: mne.time_frequency.EpochsTFR, info: tfr data for condition b
    Returns:
    -------
    evoked_power_a: mne.time_frequency.AverageTFR, info: evoked power for condition a
    evoked_power_b: mne.time_frequency.AverageTFR, info: evoked power for condition b
    """

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
    evoked_power_a = mne.time_frequency.AverageTFR(
        power_a.info, evoked_power_a_perm_arr, power_a.times, power_a.freqs,
          power_a.data.shape[0]
    )
    evoked_power_b = mne.time_frequency.AverageTFR(
        power_b.info, evoked_power_b_perm_arr, power_b.times, power_b.freqs,
          power_b.data.shape[0]
    )

    return evoked_power_a, evoked_power_b


# Helper functions

def plot_mirrored_data(subj: str):

    """Plot mirrored data on both ends
    Args:
    ----
    subj: str, info: participant ID
    Returns:
    -------
    None
    """

    # load epochs for a single participant
    epochs = mne.read_epochs(dir_clean_epochs / f"P{subj}_icacorr_0.1Hz-epo.fif")
    
    # crop the data in prestimulus window
    epochs.crop(tmin=-0.4, tmax=0)

    # get length of timeseries after cropping the data
    len_timeseries = epochs.get_data().shape[2]

    # how many epochs?
    len_epochs = epochs.get_data().shape[0]

    # select random epoch to plot
    random_epoch = random.randint(0, len_epochs)

    # get data as an numpy array
    epoch_data = epochs.get_data()

    # zero pad = False = mirror the data on both ends
    data_mirror = zero_pad_or_mirror_data(epoch_data, zero_pad=False)

    # select channel CP4
    sens_channel = epochs.ch_names.index('CP4')
    # we pick a random  epoch to plot the data and channel CP4
    plt.plot(data_mirror[random_epoch, sens_channel, :])
    plt.axvspan(0, len_timeseries, color='green', alpha=0.5)
    plt.axvspan(len_timeseries*2, len_timeseries*3, 
                color='green', alpha=0.5)
    plt.xlabel('Timepoints', fontsize=20)
    plt.ylabel('Amplitude in mV', fontsize=20)
    plt.savefig(Path(path_to.figures.manuscript.figure4) / 'mirrored_data_CP4.svg', 
                dpi=300)
    plt.show()


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
# %%
