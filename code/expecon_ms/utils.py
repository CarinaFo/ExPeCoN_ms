#!/usr/bin/python3
"""
utils for expecon_ms.

Author: Carina Forster
Years: 2024
"""

# %% Import
import numpy as np
import random
import string
from pathlib import Path
import mne
import matplotlib.pyplot as plt

import pandas as pd


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

    # remove blocks with hit rates > 90 % or < 20 %
    filtered_groups = hit_rate_per_block[
        (hit_rate_per_block > params.behavioral_cleaning.hitrate_max)
        | (hit_rate_per_block < params.behavioral_cleaning.hitrate_min)
    ]
    print(
        f"Blocks with hit rates > {params.behavioral_cleaning.hitrate_max} or "
        f" {params.behavioral_cleaning.hitrate_min}: ",
        len(filtered_groups),
    )

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

def randomize_behav_data(data_path: str = None, save_path: str = None):
    """
    Randomize behavioral data.

    Returns:
    -------
    data: pd.DataFrame, randomized data

    """
    # load data
    data = pd.read_csv(Path(data_path))

    # create a random id consisting of 5 digits for
    # each participant in the data
    def random_id():
        return "".join(random.choice(string.digits) for _ in range(5))

    # create a list of random ids
    random_ids = [random_id() for _ in range(len(pd.unique(data.ID)))]

    # now replace the initial IDs with the random IDs for each participant
    data["ID"] = data["ID"].replace(pd.unique(data.ID), random_ids)

    # drop the age column and the unnamed column
    data = data.drop(["age", "Unnamed: 0"], axis=1)

    # now we shuffle the participants data based on the randoms IDs but
    # making sure that the trial data is still in the correct order
    data = data.sort_values(["ID", "trial"])

    # save the anonymized data
    data.to_csv(Path(save_path), index=False)

    return data



def permute_trials(n_permutations: int, power_a: None, power_b: None):
    """
    Permute trials between two conditions and equalize trial counts.

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

    for _ in range(n_permutations):
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
        power_a.info, evoked_power_a_perm_arr, power_a.times, power_a.freqs, power_a.data.shape[0]
    )
    evoked_power_b = mne.time_frequency.AverageTFR(
        power_b.info, evoked_power_b_perm_arr, power_b.times, power_b.freqs, power_b.data.shape[0]
    )

    return evoked_power_a, evoked_power_b


def plot_mirrored_data(subj: str):
    """
    Plot mirrored data on both ends.

    Args:
    ----
    subj: str, info: participant ID

    Returns:
    -------
    None

    """
    # load epochs for a single participant
    # load data for a single participant from a single study (doesn't matter which one, we just need the data structure)
    epochs = mne.read_epochs(
        Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon1, f"P{subj}_icacorr_0.1Hz-epo.fif")
    )

    # crop the data in the pre-stimulus window
    epochs.crop(tmin=-0.4, tmax=0)

    # get length of timeseries after cropping the data
    len_timeseries = epochs.get_data().shape[2]

    # how many epochs?
    len_epochs = epochs.get_data().shape[0]

    # select random epoch to plot
    random_epoch = random.randint(0, len_epochs)  # noqa: S311

    # get data as an numpy array
    epoch_data = epochs.get_data()

    # zero pad = False = mirror the data on both ends
    data_mirror = zero_pad_or_mirror_data(epoch_data, zero_pad=False)

    # select channel CP4
    sens_channel = epochs.ch_names.index("CP4")
    # we pick a random epoch to plot the data and channel CP4
    plt.plot(data_mirror[random_epoch, sens_channel, :])
    plt.axvspan(0, len_timeseries, color="green", alpha=0.5)
    plt.axvspan(len_timeseries * 2, len_timeseries * 3, color="green", alpha=0.5)
    plt.xlabel("Timepoints", fontsize=20)
    plt.ylabel("Amplitude in mV", fontsize=20)
    plt.savefig(Path(paths.figures.manuscript.figure4) / "mirrored_data_CP4.svg", dpi=300)
    plt.show()

    
# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END

if __name__ == "__main__":
    pass
