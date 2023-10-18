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

# set font to Arial and font size to 22
plt.rcParams.update({"font.size": 14, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# set directory paths
behav_dir = Path(path_to.data.behavior.behavior_df)
tfr_dir = Path(path_to.data.eeg.sensor.tfr)

id_list = config.participants.ID_list

# frequencies from tfr calculation
freq_list = np.arange(3, 35, 1)

# define frequency bands we are interested in
freq_bands = {"alpha": (7, 13), "beta": (15, 25)}


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


# TODO(simon): defaults should not be mutable
def save_band_power_per_trial(time_intervals={"pre": (-0.2, 0)}, channel_name=["CP4"]):
    """
    Save the power per trial per frequency band in a csv file.

    The power is calculated for the time interval specified and averaged over
    the specified channel and frequency band.
    Power is calculated for the following frequency bands:
        - alpha (7-13 Hz)
        - beta (15-25 Hz)

    Args:
    ----
    time_intervals: dict: Dictionary with time intervals of interest.
    channel_name: list: List with channel names of interest.

    Return:
    ------
    csv file with power per trial per frequency band
    """
    # save single subject dataframes in a list
    brain_behav = []

    # loop over subjects
    for subj in id_list:
        # load single trial power
        power = mne.time_frequency.read_tfrs(tfr_dir / f"{subj}_single_trial_power-tfr.h5")[0]

        # get behavioral data
        behav_data = power.metadata

        # loop over the time intervals
        for keys, time in time_intervals.items():
            power_crop = power.copy().crop(tmin=time[0], tmax=time[1]).pick_channels(channel_name)

            # now we average over time and channels
            power_crop.data = np.mean(power_crop.data, axis=(1, 3))

            # now we average over the frequency band and time interval
            # and add the column to the behavioral data frame
            behav_data[f"{keys}_alpha"] = np.mean(power_crop.data[:, 4:11], axis=1)  # 7-13 Hz
            behav_data[f"{keys}_beta"] = np.mean(power_crop.data[:, 12:23], axis=1)  # 15-25 Hz

        # save the data in a list
        brain_behav.append(behav_data)

    # concatenate the list of dataframes and save as csv
    pd.concat(brain_behav).to_csv(behav_dir / "brain_behav.csv")

    return brain_behav


def power_criterion_corr():
    """
    Correlate the power difference between low and high expectations trials.

    Difference is measured in dprime and criterion for different frequency bands.
    TODO(simon): improve slightly
    Plot a regression plot.
    """
    # load brain behav dataframe
    df_brain_behav = pd.read_csv(behav_dir / "brain_behav.csv")

    freqs = ["pre_alpha", "pre_beta"]

    diff_p_list = []

    # now get power per participant and per condition
    for f in freqs:
        power = df_brain_behav.groupby(["ID", "cue"]).mean()[f]

        low = power.unstack()[0.25].reset_index()
        high = power.unstack()[0.75].reset_index()

        # log transform and calculate the difference
        diff_p = np.log(np.array(low[0.25])) - np.log(np.array(high[0.75]))

        diff_p_list.append(diff_p)

    # add the power difference to a dictionary
    power_dict = {"alpha": diff_p_list[0], "beta": diff_p_list[1]}

    # load random effects
    # re = pd.read_csv(Path("./data/behav/mixed_models/brms/random_effects.csv"))  # TODO(simon): rm if not needed

    # load behavioral SDT data
    out = figure1.prepare_for_plotting()

    sdt = out[0][0]
    # calculate criterion difference
    c_diff = np.array(sdt.criterion[sdt.cue == 0.25]) - np.array(sdt.criterion[sdt.cue == 0.75])
    # dprime difference between conditions
    dprime_diff = np.array(sdt.dprime[sdt.cue == 0.25]) - np.array(sdt.dprime[sdt.cue == 0.75])

    # add the sdt data to a dictionary
    sdt_params = {"dprime": dprime_diff, "criterion": c_diff}

    for p_key, p_value in power_dict.items():  # loop over the power difference
        for keys, values in sdt_params.items():  # loop over the criterion and dprime difference
            # plot regression line
            fig = sns.regplot(x=p_value, y=values)

            plt.xlabel(f"{p_key} power difference")
            plt.ylabel(f"{keys} difference")
            plt.show()
            # calculate correlation
            print(scipy.stats.spearmanr(p_value, values))
            # save figure
            fig.figure.savefig(Path(path_to.figures.manuscript.figure7, f"{p_key}_{keys}_.svg"))
            # TODO(simon): do you want the final '_' here?
            plt.show()


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
