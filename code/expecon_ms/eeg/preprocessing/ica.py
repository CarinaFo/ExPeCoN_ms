#!/usr/bin/python3
"""
Function to run ICA on epoched data and save the ICA solution.

Function that selects ICA components based on template matching or ICA label and rejects them.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023
"""
# %% Import
from __future__ import annotations

import pickle
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import pandas as pd
from mne_icalabel import label_components

from expecon_ms.configs import PROJECT_ROOT, config, path_to

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/preprocessing/ica.py")  # == __file__

last_commit_date = subprocess.check_output(
    ["git", "log", "-1", "--format=%cd", "--follow", __file__path]
).decode("utf-8").strip()
print("Last Commit Date for", __file__path, ":", last_commit_date)

# directory where to find the cleaned, epoched data
epochs_for_data_analysis = Path(path_to.data.eeg.preprocessed.stimulus.filter_0_1hz)
epochs_for_ica_dir = Path(path_to.data.eeg.preprocessed.stimulus.filter_1hz)

# directory where to save the ICA cleaned epochs
save_dir_ica = Path(path_to.data.eeg.preprocessed.ICA)


# directory of the ICA solution
save_dir_ica_sol = Path(path_to.data.eeg.preprocessed.ica.ICA_solution)
save_dir_ica_comps = Path(path_to.data.eeg.preprocessed.ica.ICA_components)

# raw EEG data
raw_dir = Path(path_to.data.eeg.RAW)

# participant IDs
id_list = config.participants.ID_list

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def run_ica(infomax: int = 1, save_psd=0):
    """
    Run ICA on epoched data and save the ICA solution.

    Args:
    ----
        infomax (int): Flag indicating whether to use the infomax method (default: 1).
        save_psd (int): Flag indicating whether to save the power spectral density (PSD) plot (default: 0).

    Returns:
    -------
        None
    """
    for subj in id_list:

        # Read the epoch data for the current participant (1Hz filtered data for ICA)
        epochs = mne.read_epochs(epochs_for_ica_dir / f"P{subj}_epochs_stim-epo.fif")

        # Pick EEG channels for ICA
        picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

        if save_psd:
            # Compute and plot the power spectral density (PSD)
            epochs.compute_psd(fmax=40, picks=picks).plot(show=False)
            plt.savefig(Path(path_to.data.eeg.preprocessed.ica.PSD, f"PSD_{subj}.png"))

        if infomax == 1:
            # Fit ICA using infomax method with extended parameters
            ica = mne.preprocessing.ICA(method="infomax",
                                        fit_params=dict(extended=True)).fit(
                                        epochs, picks=picks)
        else:
            # Fit ICA using fastica method
            ica = mne.preprocessing.ICA(method="fastica").fit(
                epochs, picks=picks)

        save_data(data=ica, identifier=subj)  # save ICA solution

    return "Done with ICA"


def label_ica_correction():
    """
    Perform template matching for blink and cardiac artifact detection.

    (correlate ICA components with eye movements and cardiac related activity.

    To detect blink and cardiac artifacts, we use the mne function detect_artifacts.
    The cleaned data will be saved as an -epo.fif file.
    - referencing to common average after ICA.

    Args:
    ----
        None.

    Returns:
    -------
        comps_removed (list): List containing the number of components removed
        for each participant.
    """
    ch_name_blinks = "VEOG"
    ch_name_ecg = "ECG"

    comps_removed = []

    for subj in id_list:
        file_path = epochs_for_ica_dir / f"P{subj}_epochs_stim-epo.fif"

        # load epochs
        epochs = mne.read_epochs(file_path, preload=True)

        # load ICA solution
        ica_sol = load_pickle(save_dir_ica_sol / f"icas_{subj}.pkl")

        # correlate components with ECG and EOG
        eog_inds, _ = ica_sol.find_bads_eog(epochs, ch_name=ch_name_blinks)
        ecg_inds, _ = ica_sol.find_bads_ecg(epochs, ch_name=ch_name_ecg, method="correlation")

        # combine the sources to exclude
        inds_to_exclude = eog_inds + ecg_inds

        # plot ICs applied to raw data, with ECG matches highlighted
        ica_sol.plot_sources(epochs, show_scrollbars=False, block=False,
                             show=False, picks=list(range(21)))
        # save figures
        plt.savefig(save_dir_ica_comps / f"ica_sources_{subj}")

        ica_sol.plot_components(inst=epochs, show=False, picks=list(range(21)))
        plt.savefig(save_dir_ica_comps / f"ica_comps_{subj}")

        ica_sol.plot_components(inst=epochs, show=False, picks=inds_to_exclude)
        plt.savefig(save_dir_ica_comps / f"ica_del_{subj}")

        ica_sol.exclude = inds_to_exclude

        comps_removed.append(len(inds_to_exclude))

        # now load the highpass filtered data
        filter_path = epochs_for_data_analysis / f"P{subj}_epochs_stim_0.1Hzfilter-epo.fif"

        epochs_filter = mne.read_epochs(filter_path, preload=True)
        # apply an ICA solution (reject components that are not brain related)
        ica_sol.apply(epochs_filter)

        # reference to average
        epochs_filter.set_eeg_reference("average", ch_type="eeg")

        # save the cleaned epochs
        epochs_filter.save(save_dir_ica / f"clean_epochs_corr/P{subj}_epochs_after_ica_0.1Hzfilter-epo.fif")

        print(f"Saved ICA cleaned epochs for participant {subj}.")

    # save a dataframe with info on how many components were removed
    pd.DataFrame(comps_removed).to_csv(save_dir_ica / "ica_components_0.1Hzfilter.csv")

    return comps_removed


def label_iclabel():
    """
    Apply automatic labeling of ICA components using the iclabel method.

    This works now with python 3.10.11 and mne 0.23.0.

    Args:
    ----
        None

    Returns:
    -------
        str: Message indicating the completion of removing ICA components.
    """
    # Store the count of removed ICA components for each participant
    ica_list = []

    for subj in id_list:
        file_path = Path(epochs_for_ica_dir, f"P{subj}_epochs_stim-epo.fif")

        # Load the clean epochs (1-Hz filtered)
        epochs_ica = mne.read_epochs(file_path)

        # load ICA solution
        ica_sol = load_pickle(save_dir_ica_sol / f"icas_{subj}.pkl")

        # use the 'iclabel' method to label components
        label_components(epochs_ica, ica_sol, method="iclabel")

        # Get indices of non-brain or other labeled components to exclude
        all_non_brain = []

        for label in ica_sol.labels_:
            if label not in ["brain", "other"]:
                all_non_brain.append(ica_sol.labels_[label])

        # unpack list of lists
        exclude_idx = [item for sublist in all_non_brain for item in sublist]

        print(f"Excluding these ICA components for participant {subj}: {exclude_idx}")

        # Save the count of excluded components per participant for methods part
        ica_list.append(len(exclude_idx))

        # now load the highpass filtered data
        filter_path = epochs_for_data_analysis / f"P{subj}_epochs_stim_0.1Hzfilter-epo.fif"

        epochs_filter = mne.read_epochs(filter_path, preload=True)

        # set the indices to exclude
        ica_sol.exclude = exclude_idx

        # Remove the non-brain components from the clean epochs
        ica_sol.apply(epochs_filter)

        # reference to average
        epochs_filter.set_eeg_reference("average", ch_type="eeg")

        # save the cleaned epochs
        epochs_filter.save(
            Path(save_dir_ica, "clean_epochs_iclabel", f"{subj}_epochs_after_ica_0.1Hzfilter-epo.fif")
        )
        print(f"Saved ICA cleaned epochs for participant {subj}")

    # save a dataframe with info on how many components were removed

    pd.DataFrame(ica_list).to_csv(Path(save_dir_ica, "clean_epochs_iclabel", "ica_components_0.1Hzfilter.csv"))

    return "Done with removing ICA components"


# Helper functions
def save_data(data, identifier):
    """
    Save data to a pickle file.

    Args:
    ----
        data: The data to be saved.
        identifier: The identifier to be included in the filename.

    Returns:
    -------
        None
    """
    with Path(save_dir_ica, f"icas_{identifier}.pkl").open("wb") as f:
        pickle.dump(data, f)


def load_pickle(file_path: str | Path):
    """
    Load data from a pickle file.

    Args:
    ----
        file_path: The path to the pickle file.

    Returns:
    -------
        The loaded data.
    """
    with Path(file_path).open("rb") as f:
        return pickle.load(f)


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END