#function to run ICA on epoched data and save the ica solution
#function that selects ICA components based on template matching or ica labels and rejects them

# please report bugs
# Author: Carina Forster, forsteca@cbs.mpg.de

# last update: 15.02.2023

import os
import pickle

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from autoreject import Ransac
from mne_icalabel import label_components

# show plots interactively  
# %matplotlib qt

# directory where to find the cleaned, epoched data
clean_epochs_dir = r'D:\expecon_ms\data\eeg\prepro_stim\downsample_after_epoching'

# directory where to save the ica cleaned epochs
save_dir_ica = r'D:\expecon_ms\data\eeg\prepro_ica'
save_dir_psd = r'D:\expecon_ms\data\eeg\prepro_ica\psd'

# directory of ica solution
save_dir_ica_sol = r'D:\expecon_ms\data\eeg\prepro_ica\ica_solution'
save_dir_ica_comps = r'D:\expecon_ms\data\eeg\prepro_ica\ica_comps'

# raw EEG data
raw_dir = r'D:\expecon_EEG\raw'

IDlist = (
    '007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
    '017', '018', '019', '020', '021', '022', '023', '024', '025', '026',
    '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
    '037', '038', '039', '040', '041', '042', '043', '044', '045', '046',
    '047', '048', '049'
)


def run_ica(infomax=1, save_psd=0):

    """
    Run ICA on epoched data and save the ICA solution.

    Args:
        infomax (int): Flag indicating whether to use the infomax method (default: 1).
        save_psd (int): Flag indicating whether to save the power spectral density (PSD) plot (default: 0).

    Returns:
        None

    """

    for counter, subj_id in enumerate(IDlist):

        # Read the epochs data for the current participant
        epochs = mne.read_epochs(f'{clean_epochs_dir}//P{subj_id}_epochs_stim-epo.fif')

        # Pick EEG channels for ICA
        picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

        if save_psd:
            # Compute and plot the power spectral density (PSD)
            epochs.compute_psd(fmax=40, picks=picks).plot(show=False)

            plt.savefig(f'{save_dir_psd}//PSD_{str(counter)}.png')

        if infomax == 1:
            # Fit ICA using infomax method with extended parameters
            ica = mne.preprocessing.ICA(method='infomax',
                                        fit_params=dict(extended=True)).fit(
                                        epochs, picks=picks)
        else:
            # Fit ICA using fastica method
            ica = mne.preprocessing.ICA(method='fastica').fit(
                epochs, picks=picks)

        save_data(data=ica, id=subj_id)  # Save the ICA solution for the participant

    return "Done with ICA"


def label_icacorr():
    """
    Perform template matching for blink and cardiac artifact detection 
    (correlate ICA components with eye movements and cardiac related activity)
    
    Args:
        None.
    
    Returns:
        comps_removed (list): List containing the number of components removed 
        for each participant.
    """

    """
    In order to detect blink and cardiac artifacts, we use the mne function detect_artifacts.
    The cleaned data will be saved as a -epo.fif file.
    Update: added rereferencing after ICA.
    """
    
    ch_name_blinks = "VEOG"
    ch_name_ecg = "ECG"

    comps_removed = []

    for subj_id in IDlist:

        # load epochs
        epochs = mne.read_epochs(f'{clean_epochs_dir}//P{subj_id}_epochs_stim-epo.fif')

        # load ica solution
        ica_sol = load_pickle(f'{save_dir_ica_sol}//icas_{subj_id}.pkl')

        eog_inds, _ = ica_sol.find_bads_eog(epochs, ch_name=ch_name_blinks)
        ecg_inds, _ = ica_sol.find_bads_ecg(epochs, ch_name=ch_name_ecg, 
                                            method='correlation')

        inds_to_exclude = eog_inds + ecg_inds

        # plot ICs applied to raw data, with ECG matches highlighted
        ica_sol.plot_sources(epochs, show_scrollbars=False, block=False, 
                             show=False, picks=list(range(21)))

        os.chdir(save_dir_ica_comps)
        plt.savefig(f'ica_sources_{subj_id}')

        ica_sol.plot_components(inst=epochs, show=False, picks=list(range(21)))
        plt.savefig(f'ica_comps_{subj_id}')

        ica_sol.plot_components(inst=epochs, show=False, picks=inds_to_exclude)
        plt.savefig(f'ica_del_{subj_id}')

        ica_sol.exclude = inds_to_exclude

        comps_removed.append(len(inds_to_exclude))

        ica_sol.apply(epochs)

        # rereference to average
        epochs.set_eeg_reference('average', ch_type="eeg")

        epochs.save(f'{save_dir_ica}//P{i}_epochs_after_ica-epo.fif')

        print(f'Saved ica cleaned epochs for participant {i}.')

    # save a dataframe with info on how many components were removed
    pd.DataFrame(comps_removed).to_csv('ica_components.csv')

    return comps_removed


def label_icalabel():
    """
    Apply automatic labeling of ICA components using the iclabel function.
    
    Args:
        None
    Returns:
        str: Message indicating the completion of removing ICA components.
    """

    # Store the count of removed ICA components for each participant
    icalist = []

    for subj_id in IDlist:

        # Load the clean epochs
        epochs = mne.read_epochs(f'{clean_epochs_dir}//P{subj_id}_epochs_stim-epo.fif')

        # Load the ICA solution
        ica_sol = mne.preprocessing.read_ica(f'{save_dir_ica_sol}/icas_{subj_id}.pkl')

        # Apply icalabel to label the ICA components
        ic_labels = ica_sol.labels_

        # Get indices of non-brain or other labeled components to exclude
        exclude_idx = [idx for idx, label in enumerate(ic_labels) if label not in ["brain", "other"]]

        print(f"Excluding these ICA components for participant {subj_id}: {exclude_idx}")

        # Save the count of excluded components
        icalist.append(len(exclude_idx))

        # Remove the non-brain components from the clean epochs
        ica_sol.apply(epochs)

        # Rereference to average
        epochs.set_eeg_reference('average', ch_type="eeg")

        # Save the cleaned data
        epochs.save(f'{save_dir_ica}//P{subj_id}_epochs_after_ic-label-epo.fif')

        print(f'Saved ICA cleaned epochs for participant {subj_id}')

    # Save a dataframe with info on how many components were removed
    pd.DataFrame(icalist).to_csv('ica_components_deleted_icalabel.csv')

    return "Done with removing ICA components"


def save_data(data, id):
    """
    Save data to a pickle file.

    Args:
        data: The data to be saved.
        id: The identifier to be included in the filename.

    Returns:
        None
    """
    os.chdir(save_dir_ica)
    with open(f'icas_{id}.pkl', 'wb') as f:
        pickle.dump(data, f)


def load_pickle(file_path):
    """
    Load data from a pickle file.

    Args:
        file_path: The path to the pickle file.

    Returns:
        The loaded data.
    """

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
