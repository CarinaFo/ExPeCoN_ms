#!/usr/bin/python3
"""
Run ICA.

Functions to run ICA (extended infomax or fastica)
and select ICA components semi-automatically using a correlation approach or
a fully automated approach using iclabel.

Data is re-referenced to the common average and saved as a .fif file.

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

from expecon_ms.configs import PROJECT_ROOT, config, paths

# if you change the config.toml file instead of reloading the kernel,
# you can uncomment and execute the following lines of code:

# from importlib import reload # noqa: ERA001
# reload(expecon_ms)  # noqa: ERA001
# reload(expecon_ms.configs) # noqa: ERA001
# from expecon_ms.configs import paths # noqa: ERA001

# for plotting in a new window (copy to interpreter)
# %matplotlib qt
# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/preprocessing/ica.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# directory where to find the cleaned, stimulus locked and epoched data
epochs_for_ica_1 = Path(paths.data.eeg.preprocessed.stimulus_expecon1)
epochs_for_ica_2 = Path(paths.data.eeg.preprocessed.stimulus_expecon2)

# create directory if not existing
Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon1).mkdir(parents=True, exist_ok=True)
Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon2).mkdir(parents=True, exist_ok=True)
Path(paths.data.eeg.preprocessed.ica.ICA_solution1).mkdir(parents=True, exist_ok=True)
Path(paths.data.eeg.preprocessed.ica.ICA_components1).mkdir(parents=True, exist_ok=True)
Path(paths.data.eeg.preprocessed.ica.ICA_solution2).mkdir(parents=True, exist_ok=True)
Path(paths.data.eeg.preprocessed.ica.ICA_components2).mkdir(parents=True, exist_ok=True)
Path(paths.data.eeg.preprocessed.ica.PSD1).mkdir(parents=True, exist_ok=True)
Path(paths.data.eeg.preprocessed.ica.PSD2).mkdir(parents=True, exist_ok=True)

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
# participant IDs
id_list_expecon1 = config.participants.ID_list_expecon1
id_list_expecon2 = config.participants.ID_list_expecon2

# add this line to jupyter script
# run_ica(study=1, infomax=1, save_psd=1) # noqa: ERA001
# label_ica_correlation() # noqa: ERA001


def run_ica(study: int, infomax: int, save_psd: int):
    """
    Run ICA on epoched data and save the ICA solution.

    Args:
    ----
        study (int): Flag indicating whether to use the data from study 1 or study 2.
        infomax (int): Flag indicating whether to use the infomax method (default: 1).
        save_psd (int): Flag indicating whether to save the power spectral density (PSD) plot (default: 0).

    Returns:
    -------
        None

    """
    id_list = id_list_expecon1 if study == 1 else id_list_expecon2

    for subj in id_list:
        file_path = (
            Path(paths.data.eeg.preprocessed.ica.ICA_solution1, f"icas_{subj}.pkl")
            if study == 1
            else Path(paths.data.eeg.preprocessed.ica.ICA_solution2, f"icas_{subj}.pkl")
        )

        if not Path(file_path).exists():
            print("The file does not exist.")

            # Read the epoch data for the current participant (1Hz filtered data for ICA)
            if study == 1:
                epochs = mne.read_epochs(epochs_for_ica_1 / f"P{subj}_epochs_1Hz-epo.fif")
            else:
                if subj == "013":
                    continue
                epochs = mne.read_epochs(epochs_for_ica_2 / f"P{subj}_epochs_1Hz-epo.fif")

            # Pick EEG channels for ICA
            picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

            if save_psd:
                # Compute and plot the power spectral density (PSD)
                epochs.compute_psd(fmin=1, fmax=40, picks=picks).plot(show=False)
                if study == 1:
                    plt.savefig(Path(paths.data.eeg.preprocessed.ica.PSD1, f"PSD_{subj}.png"))
                else:
                    plt.savefig(Path(paths.data.eeg.preprocessed.ica.PSD2, f"PSD_{subj}.png"))

            if infomax == 1:
                # Fit ICA using infomax method with extended parameters
                ica = mne.preprocessing.ICA(method="infomax", fit_params=dict(extended=True)).fit(epochs, picks=picks)
            else:
                # Fit ICA using fastica method
                ica = mne.preprocessing.ICA(method="fastica").fit(epochs, picks=picks)

            # save ICA solution
            save_data(
                path=Path(paths.data.eeg.preprocessed.ica.ICA_solution1), data=ica, identifier=subj
            ) if study == 1 else save_data(
                path=paths.data.eeg.preprocessed.ica.ICA_solution2, data=ica, identifier=subj
            )
        else:
            print("The file exists.")

    return "Done with ICA"


def label_ica_correlation(study: int):
    """
    Perform template matching for blink and cardiac artifact detection.

    (correlate ICA components with eye movements and cardiac related activity.

    To detect blink and cardiac artifacts, we use the mne function detect_artifacts.
    The cleaned data will be saved as an -epo.fif file.
    - referencing to common average after ICA.

    Args:
    ----
        study (int): Flag indicating whether to use the data from study 1 or study 2.

    Returns:
    -------
        comps_removed (list): List containing the number of components removed
        for each participant.

    """
    ch_name_blinks = "VEOG"
    ch_name_ecg = "ECG"

    comps_removed = []

    id_list = id_list_expecon1 if study == 1 else id_list_expecon2

    for subj in id_list:
        # no ica for ID 13 in study 2
        if (study == 2) and (subj == "013"):  # noqa: PLR2004
            continue

        # set the file path for clean epochs (1Hz filtered)
        file_path = (
            epochs_for_ica_1 / f"P{subj}_epochs_1Hz-epo.fif"
            if study == 1
            else epochs_for_ica_2 / f"P{subj}_epochs_1Hz-epo.fif"
        )

        # load epochs (1Hz filtered)
        epochs = mne.read_epochs(file_path, preload=True)

        # load ICA solution
        if study == 1:
            ica_sol = load_pickle(Path(paths.data.eeg.preprocessed.ica.ICA_solution1, f"icas_{subj}.pkl"))
        else:
            ica_sol = load_pickle(Path(paths.data.eeg.preprocessed.ica.ICA_solution2, f"icas_{subj}.pkl"))

        # correlate components with ECG and EOG
        eog_inds, _ = ica_sol.find_bads_eog(epochs, ch_name=ch_name_blinks)
        ecg_inds, _ = ica_sol.find_bads_ecg(epochs, ch_name=ch_name_ecg, method="correlation")

        # combine the sources to exclude
        inds_to_exclude = eog_inds + ecg_inds

        # plot ICs applied to epoched data, with ECG matches highlighted (first 20 components only)
        ica_sol.plot_sources(epochs, show_scrollbars=False, block=False, show=False, picks=list(range(21)))
        # save figures
        if study == 1:
            plt.savefig(Path(paths.data.eeg.preprocessed.ica.ICA_components1, f"ica_sources_{subj}"))
        else:
            plt.savefig(Path(paths.data.eeg.preprocessed.ica.ICA_components2, f"ica_sources_{subj}"))
        # plot the components
        ica_sol.plot_components(inst=epochs, show=False, picks=list(range(21)))

        # save figures
        if study == 1:
            plt.savefig(Path(paths.data.eeg.preprocessed.ica.ICA_components1, f"ica_comps_{subj}"))
        else:
            plt.savefig(Path(paths.data.eeg.preprocessed.ica.ICA_components2, f"ica_comps_{subj}"))

        # plot the components to be excluded
        ica_sol.plot_components(inst=epochs, show=False, picks=inds_to_exclude)

        if study == 1:
            plt.savefig(Path(paths.data.eeg.preprocessed.ica.ICA_components1, f"ica_del_{subj}"))
        else:
            plt.savefig(Path(paths.data.eeg.preprocessed.ica.ICA_components2, f"ica_del_{subj}"))

        # set the indices to exclude
        ica_sol.exclude = inds_to_exclude

        comps_removed.append(len(inds_to_exclude))

        # now load the highpass filtered data (0.1 Hz)
        if study == 1:
            filter_path = epochs_for_ica_1 / f"P{subj}_epochs_0.1Hz-epo.fif"
        else:
            if subj == "013":
                continue
            filter_path = epochs_for_ica_2 / f"P{subj}_epochs_0.1Hz-epo.fif"

        epochs_filter = mne.read_epochs(filter_path, preload=True)

        # reject components that are not brain related
        ica_sol.apply(epochs_filter)

        # reference to average
        epochs_filter.set_eeg_reference("average", ch_type="eeg")

        # save the cleaned epochs
        if study == 1:
            epochs_filter.save(
                Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon1, f"P{subj}_icacorr_0.1Hz-epo.fif")
            )
        else:
            epochs_filter.save(
                Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon2, f"P{subj}_icacorr_0.1Hz-epo.fif")
            )

        print(f"Saved ICA cleaned epochs for participant {subj}.")

    # save a dataframe with info on how many components were removed
    if study == 1:
        pd.DataFrame(comps_removed).to_csv(
            Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon1, "ica_components_stats_icacorr.csv"),
            index=False,
        )
    else:
        pd.DataFrame(comps_removed).to_csv(
            Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon2, "ica_components_stats_icacorr.csv"),
            index=False,
        )

    return comps_removed


def label_iclabel(study: int):
    """
    Apply automatic labeling of ICA components using the iclabel method.

    doi:https://doi.org/10.1016/j.neuroimage.2019.05.026.
    Non-brain related ica components are rejected.
    The ica cleaned epochs are averaged to a common reference and saved as .fif files.

    Args:
    ----
        study (int): Flag indicating whether to use the data from study 1 or study 2.

    Returns:
    -------
        str: Message indicating the completion of removing ICA components.

    """
    # Store the count of removed ICA components for each participant
    comps_removed = []

    id_list = id_list_expecon1 if study == 1 else id_list_expecon2

    for subj in id_list:
        # set the file path for clean epochs (1Hz filtered)
        file_path = (
            epochs_for_ica_1 / f"P{subj}_epochs_1Hz-epo.fif"
            if study == 1
            else epochs_for_ica_2 / f"P{subj}_epochs_1Hz-epo.fif"
        )

        # load epochs (1Hz filtered)
        epochs = mne.read_epochs(file_path, preload=True)

        # load ICA solution
        if study == 1:
            ica_sol = load_pickle(Path(paths.data.eeg.preprocessed.ica.ICA_solution1, f"icas_{subj}.pkl"))
        else:
            if subj == "013":
                continue
            ica_sol = load_pickle(Path(paths.data.eeg.preprocessed.ica.ICA_solution2, f"icas_{subj}.pkl"))

        # use the 'iclabel' method to label components
        label_components(epochs, ica_sol, method="iclabel")

        # Get indices of non-brain or other labeled components to exclude
        all_non_brain = []

        for label in ica_sol.labels_:
            if label not in {"brain", "other"}:
                all_non_brain.append(ica_sol.labels_[label])

        # unpack list of lists
        exclude_idx = [item for sublist in all_non_brain for item in sublist]

        print(f"Excluding these ICA components for participant {subj}: {exclude_idx}")

        # Save the count of excluded components per participant for methods
        comps_removed.append(len(exclude_idx))

        # now load the highpass filtered data (0.1 Hz)
        if study == 1:
            filter_path = epochs_for_ica_1 / f"P{subj}_epochs_0.1Hz-epo.fif"
        else:
            if subj == "013":
                continue
            filter_path = epochs_for_ica_2 / f"P{subj}_epochs_0.1Hz-epo.fif"

        epochs_filter = mne.read_epochs(filter_path, preload=True)

        # set the indices to exclude
        ica_sol.exclude = exclude_idx

        # Remove the non-brain components from the clean epochs
        ica_sol.apply(epochs_filter)

        # reference to average
        epochs_filter.set_eeg_reference("average", ch_type="eeg")

        # save the cleaned epochs
        if study == 1:
            epochs_filter.save(
                Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon1, f"P{subj}_iclabel_0.1Hz-epo.fif")
            )
        else:
            epochs_filter.save(
                Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon2, f"P{subj}_iclabel_0.1Hz-epo.fif")
            )

        print(f"Saved ICA cleaned epochs for participant {subj}.")

    # save a dataframe with info on how many components were removed
    if study == 1:
        pd.DataFrame(comps_removed).to_csv(
            Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon1, "ica_components_stats_iclabel.csv"),
            index=False,
        )
    else:
        pd.DataFrame(comps_removed).to_csv(
            Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon2, "ica_components_stats_iclabel.csv"),
            index=False,
        )

    return "Done with removing ICA components", comps_removed


# Helper functions
def save_data(path, data, identifier):
    """
    Save data to a pickle file.

    Args:
    ----
        path: The path to the pickle file.
        data: The data to be saved.
        identifier: The identifier to be included in the filename.

    Returns:
    -------
        None

    """
    with Path(path, f"icas_{identifier}.pkl").open("wb") as f:
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
