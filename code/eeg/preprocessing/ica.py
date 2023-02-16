#function to run ICA on epoched data and save the ica solution
#function that selects ICA components based on template matching or ica labels and rejects them

# please report bugs
# Author: Carina Forster, forsteca@cbs.mpg.de

# last update: 15.02.2023

import os

import pandas as pd
import mne
import matplotlib.pyplot as plt
import numpy as np

from autoreject import Ransac
import pickle 
from mne_icalabel import label_components

# directory of clean epoched data

clean_epochs_dir = 'D:\expecon_ms\data\eeg\prepro_stim\downsample_after_epoching'

# directory where to save the ica cleaned epochs

save_dir_ica = 'D:\expecon_ms\data\eeg\prepro_ica'
save_dir_psd = 'D:\expecon_ms\data\eeg\prepro_ica\psd'

# directory of ica solution

save_dir_ica_sol = 'D:\expecon_ms\data\eeg\prepro_ica\ica_solution'
save_dir_ica_comps = 'D:\expecon_ms\data\eeg\prepro_ica\ica_comps'

raw_dir = r'D:\expecon_EEG\raw'

IDlist = ('040', '041', '042', '043', '044', '045', '046', '047', '048', '049')

def run_ica(infomax=0):

    """
    This script takes as input cleaned epoched data (at least high pass filtered).
    It saves the PSD before ICA and computes infomax ica over all non-zero PCA components
    (accounts for interpolated channels)
    output: ica solution for each participant saved as a pickle file
     """

    icas = []

    for counter, i in enumerate(IDlist):

        if i == '040' or i == '045':
            continue

        os.chdir(clean_epochs_dir)

        epochs = mne.read_epochs('P' + i + '_epochs_stim-epo.fif')

        picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

        if counter > 30: 

            epochs.compute_psd(fmax=60, picks=picks).plot(show=False)

            os.chdir(save_dir_psd)

            plt.savefig('PSD_' + str(counter) + '.png')

        if infomax == 1:

            ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True)).fit(
                epochs, picks=picks)
        else:

            ica = mne.preprocessing.ICA(method='fastica').fit(
                epochs, picks=picks)
    
        save_data(data=ica,id = i)

    return "Done with ICA"

# pickle helper function

def save_data(data,id):

    os.chdir(save_dir_ica)

    with open('icas_' + id + '.pkl', 'wb') as f:
        pickle.dump(data, f)

# expecon study: template matching, icalabel didn't work on Windows11

def template_matching(apply_raw=0, save=1, manual=0):

    """
    In order to detect blink and cardiac artifacts we use the mne function detect_artifacts.
    If you choose to manually remove components, set detect_artifacts to "manual".
    This allows you to see all the components and select which one to remove.
    After removing artifact components we check for remaining bad epochs.
    We use autoreject and delete them (as recommended by autoreject).
    The cleaned data will be saved as a -epo.fif file
    Update: added rereferencing after ICA
    """

    ch_name_blinks = "VEOG"
    ch_name_ecg = "ECG"

    comps_removed = []
    eog_ics_list = []
    ecg_ics_list = []

    for counter, i in enumerate(IDlist):

        if i == '040' or i == '045':
            continue

        # load epochs
        os.chdir(clean_epochs_dir)
        epochs = mne.read_epochs('P' + i + '_epochs_stim-epo.fif')
        # load ica solution
        os.chdir(save_dir_ica_sol)
        with open('icas_' + i + '.pkl', 'rb') as f:
            ica_sol = pickle.load(f)
            f.close()

        eog_inds, eog_scores = ica_sol.find_bads_eog(epochs, ch_name=ch_name_blinks)

        ecg_inds, ecg_scores = ica_sol.find_bads_ecg(epochs, ch_name=ch_name_ecg, method='correlation')

        inds_to_exclude = eog_inds + ecg_inds

        # plot ICs applied to raw data, with ECG matches highlighted

        ica_sol.plot_sources(epochs, show_scrollbars=False, show=False, picks=list(range(21)))

        os.chdir(save_dir_ica_comps)

        plt.savefig('ica_sources_' + i)

        ica_sol.plot_components(inst=epochs, show=False, picks=list(range(21)))

        plt.savefig('ica_comps_' + i)

        ica_sol.plot_components(inst=epochs, show=False, picks=inds_to_exclude)
        
        plt.savefig('ica_del_' + i)

        input("Press Enter to continue after you inspected the selected ICs ...")
        
        ica_sol.exclude = inds_to_exclude

        comps_removed.append(len(inds_to_exclude))

        if manual == 1:

            ica_sol.exclude = inds_to_exclude

            ica_sol.plot_components(inst=epochs_Rpeak)

            input("Press Enter to continue after you rejected alll desired components...")

            n_exclude = len(ica_sol.exclude)

            comps_removed.append(n_exclude)

            print("Removed " + str(n_exclude) + " components for participant Nr " + i)

        # kick out components

        if apply_raw == 1:

            raw_clean.load_data()

            print("Removed " + str(len(inds_to_exclude)) + " ica components for subject Nr " + i)

            ica_sol.apply(raw_clean, exclude=inds_to_exclude)

            # rereference to average

            raw_clean.set_eeg_reference('average', ch_type="eeg")

            if save == 1:

                os.chdir(save_dir_raw_ica)

                raw_clean.save('P' + i + '_raw_after_ica-raw.fif', overwrite=True)

                print('Saved ica cleaned raw data for participant ' + i)
        else:

            ica_sol.apply(epochs)

            # rereference to average

            epochs.set_eeg_reference('average', ch_type="eeg")

            if save == 1:

                os.chdir(save_dir_ica)

                epochs.save('P' + i + '_epochs_after_ica-epo.fif', overwrite=True)

                print('Saved ica cleaned epochs for participant ' + i)

    return comps_removed

def icalabel():

    """implementation of new mne feature that allows you to label ica components automatically
    for more info checkout: https://github.com/mne-tools/mne-icalabel
    input: cleaned epochs, ica solution
    output: cleaned epochs after ica component rejection
    csv file that contains the ica labels marked as non brain from icalabel"""

    # store ica components removed

    icalist = []

    #loop over participants

    for counter, i in enumerate(IDlist):
        
        if i == '040' or i == '045':
            continue
        
        os.chdir(clean_epochs_dir)

        # open clean epochs

        epochs = mne.read_epochs('P' + i + '_epochs_stim-epo.fif')

        picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

        # open ica solution

            # open ica solution

        os.chdir(save_dir_ica_sol)

        with open('icas_' + i + '.pkl', 'rb') as f:
            ica_sol = pickle.load(f)
            f.close()

        # run icalabel on the ica components and the clean epoched data for each participant

        ic_labels = label_components(epochs, ica_sol, method="iclabel")

        # We can extract the labels of each component and exclude
        # non-brain classified components, keeping 'brain' and 'other'.
        # "Other" is a catch-all that for non-classifiable components.
        # We will ere on the side of caution and assume we cannot blindly remove these.

        #extract labels
        labels = ic_labels["labels"]

        # get indices of labels that are not brain or other labelled
        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        
        print(f"Excluding these ICA components: {exclude_idx}")

        # save the labelled indices
        icalist.append(len(exclude_idx))

        # remove the non brain components from the clean epochs
        icas[counter].apply(picks)

        # rereference to average
        picks.set_eeg_reference('average', ch_type="eeg")

        os.chdir(save_dir_ica)
        # save the clean data
        picks.save('P' + i + '_epochs_after_ica-epo.fif')

        print('Saved ica cleaned epochs for participant ' + i)

    # save a dataframe with info on how many components were removed
    pd.DataFrame(icalist).to_csv('ica_components.csv')

    return "Done with removing ica components"
