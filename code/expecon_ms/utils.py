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
from scipy.stats import pearsonr

from expecon_ms.configs import PROJECT_ROOT, config, params, paths


def zero_pad_or_mirror_epochs(epochs, zero_pad: bool, pad_length: int = 100):
    """
    Zero-pad or mirror an MNE.Epochs object on both sides to avoid edge artifacts.
    
    Args:
    ----
    epochs: mne.Epochs
        The input MNE Epochs object to be padded or mirrored.
    zero_pad: bool
        If True, zero-pad the data. If False, mirror the data.
    pad_length: int, optional
        Number of time points to pad (default is 100).

    Returns:
    -------
    padded_epochs: mne.EpochsArray
        The padded or mirrored MNE.EpochsArray object with adjusted time dimensions.
    """
    data = epochs.get_data()  # Extract the data from the epochs object
    n_epochs, n_channels, n_times = data.shape
    sfreq = epochs.info['sfreq']  # Get the sampling frequency
    times = epochs.times  # Original time array
    # save metadata
    metadata = epochs.metadata

    # Initialize list to collect padded/mirrored data
    padded_list = []

    if zero_pad:
        # Create a zero-padding array with shape (pad_length,)
        zero_pad_array = np.zeros(pad_length)

        # Loop over epochs and channels to zero-pad the data
        for epoch in range(n_epochs):
            ch_list = []
            for ch in range(n_channels):
                # Zero pad at beginning and end
                ch_list.append(np.concatenate([zero_pad_array, data[epoch][ch], zero_pad_array]))
            padded_list.append(ch_list)

    else:
        # Mirror data at the edges with a length equal to `pad_length`
        for epoch in range(n_epochs):
            ch_list = []
            for ch in range(n_channels):
                # Mirror pad at the beginning and end using `pad_length`
                ch_list.append(np.concatenate([
                    data[epoch][ch][:pad_length][::-1],  # Mirror the first `pad_length` points
                    data[epoch][ch],  # Original data
                    data[epoch][ch][-pad_length:][::-1]  # Mirror the last `pad_length` points
                ]))
            padded_list.append(ch_list)

    # Convert the list back to a numpy array with the correct shape
    padded_data = np.array(padded_list)

    # Adjust the time array to match the new data length
    time_step = times[1] - times[0]  # Time step based on sampling frequency
    new_times = np.arange(times[0] - pad_length * time_step, times[-1] + pad_length * time_step, time_step)

    # Create a new MNE EpochsArray with the padded/mirrored data
    padded_epochs = mne.EpochsArray(padded_data, epochs.info, tmin=new_times[0], events=epochs.events, event_id=epochs.event_id)

    # Add metadata back to the epochs object
    padded_epochs.metadata = metadata
    
    return padded_epochs


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


def compute_acf(sequence, lag=1):
    """
    Compute Autocorrelation Function (ACF) for a binary sequence at a given lag.
    
    Parameters:
    sequence (array-like): Binary sequence (0s and 1s).
    lag (int): Lag at which to compute autocorrelation.

    Returns:
    float: Autocorrelation coefficient at the specified lag.
    """
    # Convert binary sequence to numpy array
    sequence = np.array(sequence)
    
    # Compute ACF using Pearson correlation coefficient
    acf, _ = pearsonr(sequence[:-lag], sequence[lag:])
    return acf

def monte_carlo_permutation_test(observed_sequence, num_permutations=1000, lag=1):
    """
    Perform Monte Carlo permutation testing to assess autocorrelation using ACF.
    
    Parameters:
    observed_sequence (array-like): Observed binary sequence (0s and 1s).
    num_permutations (int): Number of permutations to generate (default is 1000).
    lag (int): Lag at which to compute autocorrelation.

    Returns:
    float: p-value indicating significance of observed autocorrelation.
    """
    # Compute ACF for observed sequence
    observed_acf = compute_acf(observed_sequence, lag)
    
    # Generate random permuted sequences and compute ACF
    permuted_acfs = []
    for _ in range(num_permutations):
        permuted_sequence = np.random.permutation(observed_sequence)
        permuted_acf = compute_acf(permuted_sequence, lag)
        permuted_acfs.append(permuted_acf)
    
    # Calculate p-value based on how extreme observed_acf is compared to permuted_acfs
    num_extreme = np.sum(np.abs(permuted_acfs) >= np.abs(observed_acf))
    p_value = (num_extreme + 1) / (num_permutations + 1)  # Add 1 for continuity correction
    
    return p_value

def monte_carlo_permutation_test_by_group(study: int = 1, num_permutations=5000, lag=1):
    """
    Perform Monte Carlo permutation testing to assess autocorrelation using ACF for each participant and block.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing columns 'block', 'ID', and 'isyes'.
    num_permutations (int): Number of permutations to generate (default is 1000).
    lag (int): Lag at which to compute autocorrelation.

    Returns:
    pd.DataFrame: DataFrame with columns 'block', 'ID', 'p-value'.
    """
    # load dataframe
    df = pd.read_csv(Path(f"E:/expecon_ms/data/behav/behav_cleaned_for_eeg_expecon{study}.csv")) 
    results = []

    # Group by 'ID' and perform permutation testing for each participant
    for subject, group in df.groupby('ID'):
        observed_sequence = group['isyes'].values
        p_value = monte_carlo_permutation_test(observed_sequence, num_permutations, lag)
        
        results.append({
            'ID': subject,
            'p-value': p_value
        })
    
    
    results_df = pd.DataFrame(results)
    return results_df

def plot_p_values(results_df):
    """
    Plot p-values for each participant and block in ascending order, marking p-values < 0.05.
    
    Parameters:
    results_df (pd.DataFrame): DataFrame with columns 'block', 'ID', 'p-value'.
    """
    # Sort results by p-value in ascending order
    results_df_sorted = results_df.sort_values(by='p-value')
    
    # Plotting
    plt.figure(figsize=(10, 6))
    xticks = np.arange(len(results_df_sorted))
    plt.bar(xticks, results_df_sorted['p-value'], align='center', alpha=0.7)
    plt.xticks(xticks, results_df_sorted['ID'], rotation=45, ha='right')
    plt.ylabel('P-value')
    plt.title('P-values for Autocorrelation Assessment by Participant')
    
    # Mark significant p-values < 0.05
    significant_indices = results_df_sorted['p-value'] < 0.05
    plt.plot(xticks[significant_indices], results_df_sorted.loc[significant_indices, 'p-value'], 'ro', label='p < 0.05')
    
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    pass