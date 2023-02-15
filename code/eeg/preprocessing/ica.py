#function to run ICA on epoched data and save as pickle
#function that selects ICA components based on template matching or labels and rejects them
#function that applys ICA to raw or epoched data and saves clean data after ICA

import os

import pandas as pd
import mne
import matplotlib.pyplot as plt

from autoreject import Ransac
import pickle 
from mne_icalabel import label_components


#IMPORTANT

# for template matching function: run in terminal with this command:
# ipython --pyvistaqt preprocessing.py
# cd Documents\expecon_EEG_analysis\python
# from ica_mne import template_matching
# template_matching()

#interactive mne plots only work with this settings (Windows11)

# directory of clean epoched data

clean_epochs_dir = 'D:\expecon_ms\data\eeg\prepro_stim'

# directory where to save the ica cleaned epochs

save_dir_ica = 'D:\expecon_ms\data\eeg\prepro_ica'

raw_dir = r'D:\expecon_EEG\raw'

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017', '018', '019', '020', '021',
          '022', '023', '024', '025', '026', '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049')

def run_ica(ica_fast=1, trigger="epochs", idlist=IDlist, plot_stuff=0):
    """
    This script takes as input cleaned epoched data (at least high pass filtered). It saves the PSD before and after ICA
     """

    icas, epochss = [],[]

    for counter, i in enumerate(idlist):

        if i == '040' or i == '045':
            continue

        plt.close('all')

        os.chdir(clean_epochs_dir)

        epochs = mne.read_epochs('P' + i + '_epochs_stim-epo.fif')

        picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

        epochs.plot_psd(average=True, picks=picks, show=False)

        plt.save('PSD_' + str(counter) + '.png')

        ica = mne.preprocessing.ICA(method='infomax', fit_params=dict(extended=True)).fit(
            epochs, picks=picks)

        icas.append(ica)
    
    save_data(data=icas)

    return "Done with ICA"

def save_data(data):

    os.chdir(save_dir_ica)

    with open('icas.pkl', 'wb') as f:
        pickle.dump(data, f)

def icalabel():

    """new mne feature that allows you to label ica components automatically"""

    # store ica components removed

    icalist = []

    # open ica solution

    os.chdir(save_dir_ica)

    with open('icas.pkl', 'rb') as f:
        icas = pickle.load(f)
        f.close()

    for counter, i in enumerate(IDlist):
        
        if i == '040' or i == '045':
            continue

        os.chdir(clean_epochs_dir)

        epochs = mne.read_epochs('P' + i + '_epochs_stim-epo.fif')

        picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

        ic_labels = label_components(picks, icas[counter], method="iclabel")

        # We can extract the labels of each component and exclude
        # non-brain classified components, keeping 'brain' and 'other'.
        # "Other" is a catch-all that for non-classifiable components.
        # We will ere on the side of caution and assume we cannot blindly remove these.

        labels = ic_labels["labels"]

        exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
        
        print(f"Excluding these ICA components: {exclude_idx}")

        icalist.append(len(exclude_idx))

        icas[counter].apply(picks)

        # rereference to average

        picks.set_eeg_reference('average', ch_type="eeg")

        os.chdir(save_dir_ica)

        picks.save('P' + i + '_epochs_after_ica-epo.fif')

        print('Saved ica cleaned epochs for participant ' + i)

    # save a dataframe with info on how many components were removed

    ica_kickedout = pd.DataFrame(icalist)
    ica_kickedout.to_csv('ica_components.csv')

    return "Done with removing ica components"


# Unused functions but might be useful´for others

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

    #open icas and epochs pickle object (list)

    os.chdir(path)

    with open('icas.pkl', 'rb') as f:
        icas = pickle.load(f)
        f.close()

    with open('epochs.pkl', 'rb') as f:
        epochs = pickle.load(f)
        f.close()

    for counter, i in enumerate(IDlist):

        os.chdir(raw_dir)

        raw_clean = mne.io.read_raw_fif('P' + i + '_raw_before_ica.fif')

        ecg_events = mne.preprocessing.find_ecg_events(raw_clean)

        epochs_Rpeak = mne.Epochs(raw_clean, ecg_events[0], tmin=-0.3, tmax=0.5,
                            preload=True, baseline=None)

        ecg_evoked = epochs_Rpeak.average()

        eog_inds, eog_scores = icas[counter].find_bads_eog(raw_clean, ch_name=ch_name_blinks)

        ecg_inds, ecg_scores = icas[counter].find_bads_ecg(raw_clean, ch_name=ch_name_ecg, method='correlation')

        # plot ICs applied to raw data, with ECG matches highlighted

        icas[counter].plot_sources(raw_clean, show_scrollbars=False)

        # plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
        icas[counter].plot_sources(ecg_evoked)

        icas[counter].plot_components(inst=epochs_Rpeak)

        input("Press Enter to continue after you inspected the selected ICs ...")

        inds_to_exclude = icas[counter].exclude

        comps_removed.append(len(inds_to_exclude))

        if manual == 1:

            icas[counter].exclude = inds_to_exclude

            icas[counter].plot_components(inst=epochs_Rpeak)

            input("Press Enter to continue after you rejected alll desired components...")

            n_exclude = len(icas[counter].exclude)

            comps_removed.append(n_exclude)

            print("Removed " + str(n_exclude) + " components for participant Nr " + i)

        # kick out components

        if apply_raw == 1:

            raw_clean.load_data()

            print("Removed " + str(len(inds_to_exclude)) + " ica components for subject Nr " + i)

            icas[counter].apply(raw_clean, exclude=inds_to_exclude)

            # rereference to average

            raw_clean.set_eeg_reference('average', ch_type="eeg")

            if save == 1:

                os.chdir(save_dir_raw_ica)

                raw_clean.save('P' + i + '_raw_after_ica-raw.fif', overwrite=True)

                print('Saved ica cleaned raw data for participant ' + i)
        else:

            icas[counter].apply(epochs[counter])

            # rereference to average

            epochs[counter].set_eeg_reference('average', ch_type="eeg")

            if save == 1:

                os.chdir(save_dir_ica)

                epochs.save('P' + i + '_epochs_after_ica-epo.fif', overwrite=True)

                print('Saved ica cleaned epochs for participant ' + i)

    return comps_removed