#!/usr/bin/python3
"""
The script contains functions for preprocessing of EEG data using the MNE toolbox.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023
"""
# %% Import
from __future__ import annotations

import copy
import subprocess
from pathlib import Path

import mne
import pandas as pd
from autoreject import AutoReject, Ransac  # Jas et al., 2016

from expecon_ms.configs import PROJECT_ROOT, config, path_to

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/preprocessing/prepro.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# set save paths
save_dir_concatenated_raw = Path(path_to.data.eeg.RAW)
# TODO(simon): adapt Paths to config style (see above and other py-files)
save_dir_stim = Path("./data/eeg/prepro_stim/filter_0.1Hz/")
save_dir_cue = Path("./data/eeg/prepro_cue/")

# EEG cap layout file
filename_montage = Path("./data/eeg/prepro_stim/CACS-64_REF.bvef")

# raw behavioral data
behav_path = Path(path_to.data.behavior.behavior_df)

# participant IDs
id_list = config.participants.ID_list


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def concatenate():
    """
    Concatenate raw data files for each participant.

    Take raw EEG data (brainvision files), read the event triggers from annotations, and concatenate the raw data
    files for each experimental block to one big .fif file.
    """
    trigger_per_block = []

    for subj in enumerate(id_list):
        # if file exists, skip
        if (save_dir_concatenated_raw / f"P{subj}_concatenated_raw.fif").exists():
            continue

        if subj == "018":
            raw_fname1 = f"expecon_EEG_{subj}_train.vhdr"  # wrongly named the first experimental block
            raw_fname2 = f"expecon_EEG_{subj}_02.vhdr"
            raw_fname3 = f"expecon_EEG_{subj}_03.vhdr"
            raw_fname4 = f"expecon_EEG_{subj}_04.vhdr"
            raw_fname5 = f"expecon_EEG_{subj}_05.vhdr"

        elif subj == "031":  # only four blocks
            raw_fname1 = f"expecon_EEG_{subj}_01.vhdr"
            raw_fname2 = f"expecon_EEG_{subj}_02.vhdr"
            raw_fname3 = f"expecon_EEG_{subj}_03.vhdr"
            raw_fname4 = f"expecon_EEG_{subj}_04.vhdr"

        else:
            raw_fname1 = f"expecon_EEG_{subj}_01.vhdr"
            raw_fname2 = f"expecon_EEG_{subj}_02.vhdr"
            raw_fname3 = f"expecon_EEG_{subj}_03.vhdr"
            raw_fname4 = f"expecon_EEG_{subj}_04.vhdr"
            raw_fname5 = f"expecon_EEG_{subj}_05.vhdr"

        # extract events from annotations and store trigger counts in a list

        if subj == "031":
            raw_1 = mne.io.read_raw_brainvision(raw_fname1)
            raw_2 = mne.io.read_raw_brainvision(raw_fname2)
            raw_3 = mne.io.read_raw_brainvision(raw_fname3)
            raw_4 = mne.io.read_raw_brainvision(raw_fname4)

            events_1, _ = mne.events_from_annotations(raw_1, regexp="Stimulus/S  2")
            events_2, _ = mne.events_from_annotations(raw_2, regexp="Stimulus/S  2")
            events_3, _ = mne.events_from_annotations(raw_3, regexp="Stimulus/S  2")
            events_4, _ = mne.events_from_annotations(raw_4, regexp="Stimulus/S  2")

            trigger_per_block.append([len(events_1), len(events_2), len(events_3), len(events_4), len(events_5)])

            raw = mne.concatenate_raws([raw_1, raw_2, raw_3, raw_4])

        else:
            raw_1 = mne.io.read_raw_brainvision(raw_fname1)
            raw_2 = mne.io.read_raw_brainvision(raw_fname2)
            raw_3 = mne.io.read_raw_brainvision(raw_fname3)
            raw_4 = mne.io.read_raw_brainvision(raw_fname4)
            raw_5 = mne.io.read_raw_brainvision(raw_fname5)

            events_1, _ = mne.events_from_annotations(raw_1, regexp="Stimulus/S  2")
            events_2, _ = mne.events_from_annotations(raw_2, regexp="Stimulus/S  2")
            events_3, _ = mne.events_from_annotations(raw_3, regexp="Stimulus/S  2")
            events_4, _ = mne.events_from_annotations(raw_4, regexp="Stimulus/S  2")
            events_5, _ = mne.events_from_annotations(raw_5, regexp="Stimulus/S  2")

            # check if we have 144 triggers per block (trials)
            # if not I forgot to turn on the EEG recording (or turned it on too late)

            trigger_per_block.append([len(events_1), len(events_2), len(events_3), len(events_4), len(events_5)])

            raw = mne.concatenate_raws([raw_1, raw_2, raw_3, raw_4, raw_5])

        # save concatenated raw data
        raw.save(f"{save_dir_concatenated_raw}P{subj}_concatenated_raw.fif")

    return trigger_per_block


def remove_trials(filename: str | Path = "raw_behav_data.csv"):
    """
    Drop trials from the behavioral data that do not have a matching trigger in the EEG recording due to human error.

    Args:
    ----
    filename: .csv file that contains behavioral data

    Returns:
    -------
    None
    """
    # load the preprocessed dataframe from R (already removed the training block)
    df_behav = pd.read_csv(behav_path / filename)  # TODO(simon): why is it called "raw_*" if preprocessed

    # remove trials where I started the EEG recording too late
    # (35 trials in total)

    # Create a list of tuples to store each condition
    conditions = [
        (7, 2, [1]),
        (8, 3, [1]),
        (9, 5, [1]),
        (10, 4, [1, 2, 3, 4, 5]),
        (12, 2, [1, 2, 3]),
        (12, 6, [1, 2]),
        (16, 3, [1]),
        (16, 5, [1, 2]),
        (18, 5, [1, 2, 3]),
        (20, 3, [1]),
        (22, 3, [1, 2, 3]),
        (24, 3, [1]),
        (24, 4, [1, 2, 3, 4]),
        (24, 5, [1, 2, 3]),
        (24, 6, [1]),
        (28, 5, [1]),
        (35, 2, [1]),
        (42, 5, [1]),
    ]

    # Iterate over the list of conditions and drop the rows for each condition
    for cond in conditions:
        df_behav = df_behav.drop(
            df_behav.index[
                (df_behav["ID"] == cond[0]) & (df_behav["block"] == cond[1]) & (df_behav["trial"].isin(cond[2]))
            ]
        )

    df_behav.to_csv(behav_path / "behav_cleaned_for_eeg.csv")


def prepro(
    trigger: str = "stimulus",
    l_freq: float = 0.1,
    h_freq: int = 40,
    tmin: float = -1.5,
    tmax: float = 1.5,
    resample_rate: int = 250,
    sf: int = 2500,
    detrend: int = 1,
    ransac: int = 1,
    autoreject: int = 0,
):
    """
    Bandpass-filter the data using a finite response filter.

    As implemented in MNE, add channel locations according to the 10/10 system, load a
    specified behavioral data file (.csv) and add events as metadata to each epoch,
    inspect data for bad channels and bad epochs using RANSAC from the autoreject package.
    To ensure the same amount of channels for all subjects, we interpolate bad
    channels, after interpolating the bad channels, the data is epoched
    to the stimulus or cue trigger events and saved as an -epo.fif file.

    Args:
    ----
    trigger: lock the data to the "stimulus" or "cue" onset
    IMPORTANT:
    -data can be epoched to stimulus(0) or cue onset (1)
    -autoreject only works on epoched data
    UPDATES:
    - downsample and filter after epoching (downsampling before epoching
      might create trigger jitter)

    # TODO(simon): add other args to docstring
    """
    # load the cleaned behavioral data for EEG preprocessing (kicked out trials with
    # no matching trigger in the EEG recording)
    df_cleaned = pd.read_csv(behav_path / "behav_cleaned_for_eeg.csv")

    # set eeg channel layout for topo plots
    montage = mne.channels.read_custom_montage(filename_montage)

    # store how many channels were interpolated per participant
    # and annotations (trigger) information
    ch_interp, annot = [], []

    # loop over participants
    for index, subj in enumerate(id_list):
        # if file exists, skip
        if (save_dir_stim / f"P{subj}_epochs_stim_0.1Hzfilter-epo.fif").exists():
            print(f"{subj} already exists")
            continue

        # load raw data concatenated for all blocks
        raw = mne.io.read_raw_fif(fname=save_dir_concatenated_raw / f"P{subj}_concatenated_raw.fif", preload=True)

        # save the annotations (trigger) information
        annot.append(raw.annotations.to_data_frame())

        raw.set_channel_types({"VEOG": "eog", "ECG": "ecg"})

        # setting montage from brainvision montage file
        raw.set_montage(montage)

        # bandpass filter the data
        raw.filter(l_freq=l_freq, h_freq=h_freq, method="iir")

        # load stimulus trigger events
        events, _ = mne.events_from_annotations(raw, regexp="Stimulus/S  2")

        # add stimulus onset cue as a trigger to event structure
        if trigger == "cue":
            cue_timings = [i - int(0.4 * sf) for i in events[:, 0]]

            # subtract 0.4*sampling frequency to get the cue time stamps
            cue_events = copy.deepcopy(events)
            cue_events[:, 0] = cue_timings
            cue_events[:, 2] = 2

        # add dataframe as metadata to epochs
        df_sub = df_cleaned[index + 7 == df_cleaned.ID]  # the first six subjects were pilot subjects
        # TODO(simon): consider [param] usage from config.toml for "7" here. Why +7 if only 6 pilots?

        metadata = df_sub

        # lock the data to the specified trigger and create epochs
        if trigger == "cue":
            # stimulus onset cue
            epochs = mne.Epochs(
                raw,
                cue_events,
                event_id=2,
                tmin=tmin,
                tmax=tmax,
                preload=True,
                baseline=None,
                detrend=detrend,
                metadata=metadata,
            )
        # or stimulus
        else:
            epochs = mne.Epochs(
                raw,
                events,
                event_id=1,
                tmin=tmin,
                tmax=tmax,
                preload=True,
                baseline=None,
                detrend=detrend,
                metadata=metadata,
            )

        # downsample the data (resampling before epoching jitters the trigger)
        epochs.resample(resample_rate)

        # pick only EEG channels for Ransac bad channel detection
        picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

        # use RANSAC to detect bad channels
        # (autoreject interpolates bad channels and detects bad
        # epochs, takes quite long and removes a lot of epochs due to
        # blink artifacts)
        if ransac:
            print(f"Run ransac for {subj}")

            ransac = Ransac(verbose="progressbar", picks=picks, n_jobs=3)

            # which channels have been marked bad by RANSAC

            epochs = ransac.fit_transform(epochs)

            print("\n".join(ransac.bad_chs_))

            ch_interp.append(ransac.bad_chs_)

        # detect bad epochs
        # now feed the clean channels into Autoreject to detect bad trials
        if autoreject:
            ar = AutoReject()

            epochs, reject_log = ar.fit_transform(epochs)

            reject_log.save(save_dir_stim / "f'P_{subj}_reject_log.npz'")

        if trigger == "cue":
            epochs.save(save_dir_cue / f"P{subj}_epochs_cue-epo.fif")

        else:
            epochs.save(save_dir_stim / "f'P{subj}_epochs_stim_{l_freq}Hz_filter-epo.fif'")

            print(f"saved epochs for participant {subj}")

    # for methods part: how many channels were interpolated per participant
    ch_df = pd.DataFrame(ch_interp)

    ch_df.to_csv(save_dir_stim / "interpolated_channels.csv")

    print("Done with preprocessing and creating clean epochs")

    return ch_interp, annot


def channels_interp() -> None:
    """
    Calculate the mean, std, min and max of channels interpolated across participants.

    Returns
    -------
    None.
    """
    df_inter_ch = pd.read_csv(f'{save_dir_stim}{Path("/")}interpolated_channels.csv')

    # TODO(simon): Check this out: https://stackoverflow.com/questions/36519086/how-to-get-rid-of-unnamed-0-column-in-a-pandas-dataframe-read-in-from-csv-fil
    df_inter_ch = df_inter_ch.drop(["Unnamed: 0"], axis=1)
    df_inter_ch["count_ch"] = df_inter_ch.count(axis=1)

    print(f"mean channels interpolated {df_inter_ch['count_ch'].mean()}")
    print(f"std of channels interpolated: {df_inter_ch['count_ch'].std()}")
    print(f"min channels interpolated: {df_inter_ch['count_ch'].min()}")
    print(f"max channels interpolated: {df_inter_ch['count_ch'].max()}")


# Unused functions
def add_reaction_time_trigger() -> None:
    """Add the reaction time as a trigger to the event structure."""
    # reset index
    # TODO(simon): metadata, sf, and events are not defined
    metadata_index = metadata.reset_index()

    # get rt per trial
    rt = metadata_index.respt1

    # add event trigger
    rt_timings = [event + int(rt[index] * sf) for index, event in enumerate(events[:, 0])]

    rt_n_ts = copy.deepcopy(events)
    rt_n_ts[:, 0] = rt_timings
    rt_n_ts[:, 2] = 3


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
