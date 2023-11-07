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

import expecon_ms.configs
from expecon_ms.configs import PROJECT_ROOT, config, path_to

# if you change the config.toml file instead of reloading the kernel you can 
# uncomment and execute the following lines of code:

# from importlib import reload
# reload(expecon_ms)
# reload(expecon_ms.configs)
# from expecon_ms.configs import path_to
# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/preprocessing/prepro.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# raw concatenated eeg data
save_dir_concatenated_raw = Path(path_to.data.eeg.RAW)

# stimulus locked
save_dir_stim = Path(path_to.data.eeg.preprocessed.stimulus)

# cue locked
save_dir_cue = Path(path_to.data.eeg.preprocessed.cue)

# EEG cap layout file
filename_montage = Path(path_to.data.templates)

# raw behavioral data
behav_path = Path(path_to.data.behavior)

# participant IDs
id_list = config.participants.ID_list

# pilot data counter
pilot_counter = config.participants.pilot_counter

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def prepro(
    trigger: str,
    l_freq: float,
    h_freq: int,
    tmin: float,
    tmax: float,
    resample_rate: float,
    sf: int,
    detrend: int,
    ransac: int,
    autoreject: int,
):
    """
    Bandpass-filter the data using a finite response filter.
    As implemented in MNE, add channel locations according to the 10/10 system,
    load a specified behavioral data file (.csv) and add events as metadata to 
    each epoch, inspect data for bad channels and bad epochs using RANSAC from the 
    autoreject package. To ensure the same amount of channels for all subjects,
    we interpolate bad channels, after interpolating the bad channels, the data is 
    epoched to the stimulus or cue trigger events and saved as an -epo.fif file.

    Args:
    ----
   
    trigger: str
        Specify whether to epoch the data to the stimulus or cue trigger.
        Options: "stimulus" or "cue"
    l_freq: float
        Low cut-off frequency in Hz.
    h_freq: int
        High cut-off frequency in Hz.
    tmin: float
        Start time before event in seconds.
    tmax: float
        End time after event in seconds.
    resample_rate: float
        New sampling frequency in Hz.
    sf: int
        Sampling frequency of the raw data in Hz.
    detrend: int
        If 1, the data is detrended (linear detrending)
    ransac: int
        If 1, RANSAC is used to detect bad channels.
    autoreject: int
    Returns:
    -------
    ch_interp: list
        List with the number of interpolated channels per participant.
    annot: list
        List with the annotations (trigger) information.
    """
    # load the cleaned behavioral data for EEG preprocessing (kicked out trials with
    # no matching trigger in the EEG recording)
    df_cleaned = pd.read_csv(behav_path / "behav_cleaned_for_eeg.csv")

    # set eeg channel layout for topo plots
    montage = mne.channels.read_custom_montage(filename_montage / "CACS-64_REF.bvef")

    # store how many channels were interpolated per participant
    # and annotations (trigger) information
    ch_interp, annot = [], []

    if trigger == "stimulus":
        save_dir = save_dir_stim
    else:
        save_dir = save_dir_cue

    # loop over participants
    for index, subj in enumerate(id_list):
        # if file exists, skip
        if (save_dir / f"P{subj}_epochs_{l_freq}Hz-epo.fif").exists():
            print(f"{subj} already exists")
            continue

        # load raw data concatenated for all blocks
        raw = mne.io.read_raw_fif(save_dir_concatenated_raw / f"P{subj}_concatenated_raw.fif",
                                   preload=True)

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
        metadata = df_cleaned[index + pilot_counter == df_cleaned.ID]

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

            reject_log.save(save_dir / f'P_{subj}_reject_log_{l_freq}.npz')
        
        #save epochs to disk
        epochs.save(save_dir / f'P{subj}_epochs_{l_freq}Hz-epo.fif')

        print(f"saved epochs for participant {subj}")

    # for methods part: how many channels were interpolated per participant
    ch_df = pd.DataFrame(ch_interp)

    ch_df.to_csv(save_dir / f"interpolated_channels_{l_freq}.csv", 
                 index=False)

    print("Done with preprocessing and creating clean epochs")

    return ch_interp, annot


def channels_interp(trigger: str, l_freq: float) -> None:
    """
    Calculate the mean, std, min and max of channels interpolated across participants.

    Parameters
    ----------
    trigger : str
        Specify whether to epoch the data to the stimulus or cue trigger.
        Options: "stimulus" or "cue"
    l_freq : float
        Low cut-off frequency in Hz.
    Returns
    -------
    None.
    """
    
    if trigger == "stimulus":
        save_dir = save_dir_stim
    else:
        save_dir = save_dir_cue

    # load channel interpolation data
    df_inter_ch = pd.read_csv(f'{save_dir}{Path("/")}interpolated_channels_{l_freq}.csv')

    # (simon): Check this out: https://stackoverflow.com/questions/36519086/how-to-get-rid-of-unnamed-0-column-in-a-pandas-dataframe-read-in-from-csv-fil
    # Merci, this is very useful!
    df_inter_ch["count_ch"] = df_inter_ch.count(axis=1)

    print(f"mean channels interpolated {df_inter_ch['count_ch'].mean()}")
    print(f"std of channels interpolated: {df_inter_ch['count_ch'].std()}")
    print(f"min channels interpolated: {df_inter_ch['count_ch'].min()}")
    print(f"max channels interpolated: {df_inter_ch['count_ch'].max()}")


# Unused functions

def add_reaction_time_trigger(metadata=None, sf: int, events=None) -> None:

    """Add the reaction time as a trigger to the event structure.
    
    Parameters
    ----------
    metadata : pd.DataFrame
        Dataframe with behavioral data.
    sf : int
        Sampling frequency of the raw data in Hz.
    events : np.array
        Array with the event structure.
    Returns
    -------
    None.
    """

    # reset index
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
