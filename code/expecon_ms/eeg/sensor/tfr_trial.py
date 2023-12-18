#!/usr/bin/python3
"""
The script averages power per trial in specific frequency bands.

Moreover, it saves single trial power as well as behavioral data in a csv file.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023
"""
# %% Import
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from expecon_ms.behav import figure1
from expecon_ms.configs import PROJECT_ROOT, config, path_to

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/sensor/tfr_trial.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# set font to Arial and font size to 14
plt.rcParams.update({"font.size": 14, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# set directory paths
behav_dir = Path(path_to.data.behavior)
tfr_dir = Path(path_to.data.eeg.sensor.tfr)

# participant IDs
id_list_expecon1 = config.participants.ID_list_expecon1
id_list_expecon2 = config.participants.ID_list_expecon2

# frequencies from tfr calculation
freq_list = np.arange(3, 35, 1)

# define frequency bands we are interested in
freq_bands = {"alpha": (7, 13), "beta": (15, 25)}

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

def save_band_power_per_trial(study: int, time_intervals: dict,
                              channel_name: list):
    """
    Save the power per trial per frequency band in a csv file.

    The power is calculated for the time interval specified and averaged over
    the specified channel and frequency band.
    Power is calculated for the following frequency bands:
        - alpha (7-13 Hz)
        - beta (15-25 Hz)

    Args:
    ----
    study: int: Study number: 1 = block design, 2 = trial design
    time_intervals: dict: Dictionary with column names as keys and values
        as tuples with the time interval of interest.
    channel_name: list: List with channel names of interest.

    Return:
    ------
    csv file with power per trial per frequency band
    """
    # save single subject dataframes in a list
    brain_behav = []

    if study == 1:
        id_list = id_list_expecon1
    elif study == 2:
        id_list = id_list_expecon2

    # loop over subjects
    for subj in id_list:
        if ((study == 2) & (subj == "013")):
            continue
        # load single trial power
        power = mne.time_frequency.read_tfrs(tfr_dir / f"{subj}_single_trial_power_{str(study)}-tfr.h5")[0]

        # get behavioral data
        behav_data = power.metadata

        # at which time window do we want to look at?
        if study == 1:
            power_crop = power.copy().crop(tmin=time_intervals['pre'][0][0],
                 tmax=time_intervals['pre'][0][1]).pick_channels(channel_name)
        else:
            power_crop = power.copy().crop(tmin=time_intervals['pre'][1][0],
                 tmax=time_intervals['pre'][1][1]).pick_channels(channel_name)

        # now we average over time and channels
        power_crop.data = np.mean(power_crop.data, axis=(1, 3))
        # using list comprehension + enumerate()

        # extract the indices of the frequency bands
        alpha_indices = [idx for idx, val in enumerate(freq_list) if val >= freq_bands['alpha'][0]
                            and val <= freq_bands['alpha'][1]]
        beta_indices = [idx for idx, val in enumerate(freq_list) if val >= freq_bands['beta'][0]
                        and val <= freq_bands['beta'][1]]

        # now we average over the frequency band
        behav_data[f"pre_alpha"] = np.mean(power_crop.data[:, alpha_indices], axis=1)  # 7-13 Hz
        behav_data[f"pre_beta"] = np.mean(power_crop.data[:, beta_indices], axis=1)  # 15-25 Hz

        # save the data in a list
        brain_behav.append(behav_data)

    # concatenate the list of dataframes and save as csv
    pd.concat(brain_behav).to_csv(behav_dir / f"brain_behav_{str(study)}.csv")

    return brain_behav


def power_criterion_corr(study: int) -> None:
    """Correlate power vs. criterion difference.

    Plots regression line and calculates spearman correlation.

    Args:
    ----
    study: int: Study number: 1 = block design, 2 = trial design

    Return:
    ------
    None
    """
    # load brain behav dataframe
    df_brain_behav = pd.read_csv(behav_dir / f"brain_behav_{study}.csv")

    freqs = ["prestim_alpha", "prestim_beta"]

    diff_p_list = []

    # now get power per participant and per condition
    for f in freqs:

        power = df_brain_behav.groupby(["ID", "cue"])[f].mean()

        low = power.unstack()[0.25].reset_index()
        high = power.unstack()[0.75].reset_index()

        # log transform and calculate the difference
        diff_p = np.log(np.array(low[0.25])) - np.log(np.array(high[0.75]))

        diff_p_list.append(diff_p)

    # add the power difference to a dictionary
    power_dict = {"alpha": diff_p_list[0], "beta": diff_p_list[1]}

    # load behavioral SDT data
    out = figure1.prepare_for_plotting(exclude_high_fa = False, expecon = 1)

    # get sdt parameters
    sdt = out[0][0]

    # calculate criterion difference between conditions
    c_diff = np.array(sdt.criterion[sdt.cue == 0.25]) - np.array(sdt.criterion[sdt.cue == 0.75])

    # dprime difference between conditions
    dprime_diff = np.array(sdt.dprime[sdt.cue == 0.25]) - np.array(sdt.dprime[sdt.cue == 0.75])

    # add the sdt data to a dictionary
    sdt_params = {"dprime": dprime_diff, "criterion": c_diff}

    for freq_band, power_values in power_dict.items():  # loop over the power difference
        for sdt_param, sdt_values in sdt_params.items():  # loop over the criterion and dprime difference
            # plot regression line
            fig = sns.regplot(x=power_values, y=sdt_values)

            plt.xlabel(f"{freq_band} power difference")
            plt.ylabel(f"{sdt_param} difference")
            plt.show()
            # calculate correlation (spearman is non parametric)
            print(scipy.stats.spearmanr(power_values, sdt_values))
            # save figure
            fig.figure.savefig(Path(path_to.figures.manuscript.figure3, f"{freq_band}_{sdt_param}.svg"))
            plt.show()


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
