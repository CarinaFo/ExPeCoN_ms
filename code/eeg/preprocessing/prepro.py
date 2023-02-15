# file contains functions that does minimal preprocessing for raw eeg data (.fif files)
# using the mne toolbox


# Please report bugs
# Author: Carina Forster
# Last update: 15.02.2023


import os
import copy
import mne
import pandas as pd

# for automatic detection of bad channels and bad epochs
from autoreject import AutoReject, Ransac  # Jas et al., 2016

# set directories

# raw EEG data
raw_dir = r'D:\expecon_EEG\raw'
# EEG cap layout file
filename_montage = r'D:\expecon_EEG\CACS-64_REF.bvef'
# raw behavioral data
behavpath = r'D:\expecon_ms\data\behav'

# save cleaned EEG data
save_dir_cue = r'D:\expecon_ms\data\eeg\prepro_cue'
save_dir_stim = r'D:\expecon_ms\data\eeg\prepro_stim'

IDlist = (
'007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017',
'018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', 
'029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039',
'040','041', '042', '043', '044', '045', '046', '047', '048', '049')


def concatenate():

    """this function takes as input raw EEG data (brainvision files), 
    reads the event triggers from annotations
    and concatenates the raw data files to one big .fif file"""

    for counter, i in enumerate(IDlist):

        os.chdir(raw_dir)

        if os.path.isfile('P' + i + '_concatenated_raw.fif') == 0:

            print('concatenated file does not exist and participant ' + i + ' will be analyzed')

            if i == '018':

                raw_fname1 = 'expecon_EEG_' + i + '_train.vhdr'  # wrongly named the first experimental block
                raw_fname2 = 'expecon_EEG_' + i + '_02.vhdr'
                raw_fname3 = 'expecon_EEG_' + i + '_03.vhdr'
                raw_fname4 = 'expecon_EEG_' + i + '_04.vhdr'
                raw_fname5 = 'expecon_EEG_' + i + '_05.vhdr'

            elif i == '031':  # only 4 blocks

                raw_fname1 = 'expecon_EEG_' + i + '_01.vhdr'
                raw_fname2 = 'expecon_EEG_' + i + '_02.vhdr'
                raw_fname3 = 'expecon_EEG_' + i + '_03.vhdr'
                raw_fname4 = 'expecon_EEG_' + i + '_04.vhdr'

            else:

                raw_fname1 = 'expecon_EEG_' + i + '_01.vhdr'
                raw_fname2 = 'expecon_EEG_' + i + '_02.vhdr'
                raw_fname3 = 'expecon_EEG_' + i + '_03.vhdr'
                raw_fname4 = 'expecon_EEG_' + i + '_04.vhdr'
                raw_fname5 = 'expecon_EEG_' + i + '_05.vhdr'

            datapath = r'D:\expeco_EEG'

            os.chdir(datapath)

            # Construct Epochs

            if i == '031':

                raw_1 = mne.io.read_raw_brainvision(raw_fname1)
                raw_2 = mne.io.read_raw_brainvision(raw_fname2)
                raw_3 = mne.io.read_raw_brainvision(raw_fname3)
                raw_4 = mne.io.read_raw_brainvision(raw_fname4)

                events_1, event_dict = mne.events_from_annotations(raw_1, regexp='Stimulus/S  2')
                events_2, event_dict = mne.events_from_annotations(raw_2, regexp='Stimulus/S  2')
                events_3, event_dict = mne.events_from_annotations(raw_3, regexp='Stimulus/S  2')
                events_4, event_dict = mne.events_from_annotations(raw_4, regexp='Stimulus/S  2')

                print(len(events_1), len(events_2), len(events_3), len(events_4))

                raw = mne.concatenate_raws([raw_1, raw_2, raw_3, raw_4])

            else:

                # get the header to extract events

                raw_1 = mne.io.read_raw_brainvision(raw_fname1)
                raw_2 = mne.io.read_raw_brainvision(raw_fname2)
                raw_3 = mne.io.read_raw_brainvision(raw_fname3)
                raw_4 = mne.io.read_raw_brainvision(raw_fname4)
                raw_5 = mne.io.read_raw_brainvision(raw_fname5)

                events_1, event_dict = mne.events_from_annotations(raw_1, regexp='Stimulus/S  2')
                events_2, event_dict = mne.events_from_annotations(raw_2, regexp='Stimulus/S  2')
                events_3, event_dict = mne.events_from_annotations(raw_3, regexp='Stimulus/S  2')
                events_4, event_dict = mne.events_from_annotations(raw_4, regexp='Stimulus/S  2')
                events_5, event_dict = mne.events_from_annotations(raw_5, regexp='Stimulus/S  2')

                # check if we have 144 trigger per block (trials)
                # if not I forgot to turn on the EEG recording (or turned it on too late)

            print(len(events_1), len(events_2), len(events_3), len(events_4), len(events_5))

            raw = mne.concatenate_raws([raw_1, raw_2, raw_3, raw_4, raw_5])

            os.chdir(raw_dir)

            raw.save('P' + i + '_concatenated_raw.fif', overwrite=True)

        else:

            print('Participant' + i + 'already analyzed')

def remove_trials(filename='updated_behav.csv'):  

    """this function drops trials from the behavioral data that do not
    have a matching trigger in the EEG recording due to human error
    inputs:
    filename= .csv file that contains behavioral data
    return: None
    """

    os.chdir(behavpath)

    # load the preprocessed dataframe from R (already removed the trainingsblock)

    df = pd.read_csv(filename)

    # remove trials where I started the EEG recording too late
    # (38 trials in total, 35 trials without Tilmans toe data)

    df.drop(
        df.index[(df["ID"] == 7) & (df["block"] == 2) & (df["trial"] == 1)],
        inplace=True)
    df.drop(
        df.index[(df["ID"] == 8) & (df["block"] == 3) & (df["trial"] == 1)],
        inplace=True)
    df.drop(
        df.index[(df["ID"] == 9) & (df["block"] == 5) & (df["trial"] == 1)],
        inplace=True)
    df.drop(df.index[(df["ID"] == 10) & (df["block"] == 4) & (
        df["trial"].isin([1, 2, 3, 4, 5]))], inplace=True)
    df.drop(df.index[(df["ID"] == 12) & (df["block"] == 2) & (
        df["trial"].isin([1, 2, 3])) | (df["ID"] == 12) &
                     (df["block"] == 6) & (df["trial"].isin([1, 2]))],
            inplace=True)
    df.drop(df.index[(df["ID"] == 16) & (df["block"] == 3) & (
        df["trial"].isin([1])) | (df["ID"] == 16) &
                     (df["block"] == 5) & (df["trial"].isin([1, 2]))],
            inplace=True)
    df.drop(df.index[(df["ID"] == 18) & (df["block"] == 5) & (
        df["trial"].isin([1, 2, 3]))], inplace=True)
    df.drop(df.index[(df["ID"] == 20) & (df["block"] == 3) & (
        df["trial"].isin([1]))], inplace=True)
    df.drop(df.index[(df["ID"] == 22) & (df["block"] == 3) & (
        df["trial"].isin([1, 2, 3]))], inplace=True)
    df.drop(df.index[(df["ID"] == 24) & (df["block"] == 3) & (
        df["trial"].isin([1])) | (df["ID"] == 24) &
                     (df["block"] == 4) & (df["trial"].isin([1, 2, 3, 4])) | (
                                 df["ID"] == 24) & (
                             df["block"] == 5) &
                     (df["trial"].isin([1, 2, 3])) | (df["ID"] == 24) & (
                                 df["block"] == 6) & (
                         df["trial"].isin([1]))], inplace=True)
    df.drop(df.index[(df["ID"] == 28) & (df["block"] == 5) & (
        df["trial"].isin([1]))], inplace=True)
    df.drop(df.index[(df["ID"] == 35) & (df["block"] == 2) & (
        df["trial"].isin([1]))], inplace=True)
    df.drop(df.index[(df["ID"] == 42) & (df["block"] == 5) & (
        df["trial"].isin([1]))], inplace=True)

    df.to_csv('behav_cleaned_for_eeg.csv')

def prepro(trigger=0, l_freq=1, h_freq=40, filter_method='iir',
           tmin=-1.5, tmax=1.5, resample_rate=250, laplace=0, sf=2500,
           detrend=1, ransac=1, autoreject=0, reject_criteria = dict(eeg=100e-6,eog=200e-6),
           flat_criteria = dict(eeg=1e-6)):

    """
    this function bandpass filters the data using a finite response filter as
    implemented in MNE, adds channel locations according to the 10/10 system, loads a
    specified behavioral data file (.csv) and adds events as metadata to each epoch,
    inspects data for bad channels and bad epochs using RANSAC from the autoreject package.
    To ensure the same amount of channels for all subjecst we interpolate bad
    channels, after interpolating the bad channels, the data is epoched
    to the stimulus or cue trigger events and saved as a -epo.fif file
    trigger: lock the data to the stimulus or cue onset
    IMPORTANT:
    -data can be epoched to stimulus(0) or cue onset (1) 
    -autoreject only works on epoched data
    UPDATES:
    - downsample and filter after epoching (downsampling before epoching
      might create trigger jitter)
      """

    os.chdir(behavpath)

    # load the cleaned behavioral data for EEG preprocessing

    df = pd.read_csv('behav_cleaned_for_eeg.csv')
    
    montage = mne.channels.read_custom_montage(filename_montage)

    # store how many channels were interpolated per participant
    # and EEG annotations

    ch_interp, anot = [], []

    # loop over participants

    for counter, i in enumerate(IDlist):

        # skip those IDs as we anyway exclude them
        #  
        if i == '040' or i == '045':
            continue

        # load raw data concatenated for all blocks

        os.chdir(raw_dir)

        raw = mne.io.read_raw_fif('P' + i + '_concatenated_raw.fif')
        
        # save the annotations (trigger) information

        anot.append(raw.annotations.to_data_frame())

        raw.set_channel_types({'VEOG': 'eog', 'ECG': 'ecg'})

        # setting montage from brainvision montage file

        raw.set_montage(montage)

        # load stimulus trigger events

        events, event_dict = mne.events_from_annotations(raw, regexp='Stimulus/S  2')

        # add cue as a trigger to event structure

        cue_timings = [i - int(0.4 * sf) for i in
                       events[:,
                       0]]

        # subtract 0.4*sampling frequency to get the cue time stamps

        cue_events = copy.deepcopy(events)
        cue_events[:, 0] = cue_timings
        cue_events[:, 2, ] = 2

        # add dataframe as metadata to epochs

        df_sub = df[df.ID == counter + 7]

        metadata = df_sub

        # lock the data to the cue trigger and create epochs

        if trigger == 1:

            epochs = mne.Epochs(raw, cue_events, event_id=2, tmin=tmin,
                                tmax=tmax,
                                preload=True, baseline=None, detrend=detrend,
                                metadata=metadata)
        # or stimulus onset

        else:

            epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin, tmax=tmax,
                                preload=True, baseline=None, detrend=detrend,
                                metadata=metadata)
            
        # resample the data (resampling before epoching jitters the trigger)

        epochs.resample(resample_rate)

        # now filter the epochs 

        epochs.filter(l_freq=l_freq, h_freq=h_freq, method=filter_method)

        # pick only EEG channels for Ransac bad channel detection 

        picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

        # use RANSAC to detect bad channels
        # (autoreject interpolates bad channels and detects bad
        # epochs, takes quite long and removes a lot of epochs due to
        # blink artifacts)

        if ransac == 1:

            ransac = Ransac(verbose='progressbar', picks=picks, n_jobs=5)

            # which channels have been marked bad by RANSAC

            epochs = ransac.fit_transform(epochs)

            print('\n'.join(ransac.bad_chs_))

            print(len(ransac.bad_chs_))

            ch_interp.append(len(ransac.bad_chs_))

        # detect bad epochs

        # now feed the clean channels into Autoreject to detect bad trials

        if autoreject:

            ar = AutoReject()

            epochs, reject_log = ar.fit_transform(epochs)

            os.chdir(save_dir_stim)

            reject_log.save('P_' + i + '_reject_log.npz')

        # laplace filter

        if laplace == 1:

            epochs = mne.preprocessing.compute_current_source_density(epochs)

        if trigger == 1:

            os.chdir(save_dir_cue)

            epochs.save('P' + i + '_epochs_cue-epo.fif', overwrite=True)

        else:

            os.chdir(save_dir_stim)

            epochs.save('P' + i + '_epochs_stim-epo.fif', overwrite=True)

            print('saved epochs for participant ' + i)

    ch_df = pd.Dataframe(ch_interp)

    ch_df.to_csv('interpolated_channels.csv')

    print("Done with preprocessing and creating clean epochs")

    return ch_interp


def add_reaction_time_trigger():

    # add reaction time as trigger event

    # we reset the index because of the individual amount of trials removed in the
    # beginning

    metadata_index = metadata.reset_index()

    rt = metadata_index.respt1

    rt_timings = [event + int(rt[index] * sf) for index, event in
                 enumerate(events[:, 0])]

    rtnts = copy.deepcopy(events)
    rtnts[:, 0] = rt_timings
    rtnts[:, 2, ] = 3

    # concatenate all trigger events

    trigger_events = np.concatenate([events, cue_events, rtnts])

    # save trigger events for each participant in a list

    events_allsubs.append(trigger_events)