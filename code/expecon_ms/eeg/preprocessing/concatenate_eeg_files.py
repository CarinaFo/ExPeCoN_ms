# function to concatenate raw EEG data files per block to one big file
# function to remove trials from the behavioral data that do not have a
#  matching trigger in the EEG recording due to human error (too late recordings)

import mne
import pandas as pd
from pathlib import Path

# set paths
# raw eeg data per block
save_dir_concatenated_raw = Path(path_to.data.eeg.RAW_blocks)

# raw concatenated eeg data
save_dir_concatenated_raw = Path(path_to.data.eeg.RAW)

def concatenate():
    """
    Concatenate raw data files for each participant.

    Take raw EEG data (brainvision files), read the event triggers from annotations, and concatenate the raw data
    files for each experimental block to one big .fif file.
    """
    trigger_per_block = []

    for subj in id_list:
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
    # COMMENT: the data is not preprocessed, only the trainings block is removed :)

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
                (df_behav["ID"] == cond[0]) & (df_behav["block"] == cond[1])
                & (df_behav["trial"].isin(cond[2]))
            ]
        )

    df_behav.to_csv(behav_path / "behav_cleaned_for_eeg.csv")
