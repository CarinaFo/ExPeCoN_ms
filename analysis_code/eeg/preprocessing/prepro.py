# file contains functions that does minimal preprocessing for raw eeg data (.fif files)
# using the mne toolbox

# please report bugs
# Author: Carina Forster, forsteca@cbs.mpg.de

# last update: 15.02.2023

import os
import copy
import mne
import pandas as pd

# for automatic detection of bad channels and bad epochs
from autoreject import AutoReject, Ransac  # Jas et al., 2016

# set directories

# raw EEG data
raw_dir = r'D:\backup_expecon_EEG\expeco_EEG_201021\expecon_eeg'

# EEG cap layout file
filename_montage = r'D:\expecon_ms\data\eeg\prepro_stim\CACS-64_REF.bvef'

# raw behavioral data
behavpath = r'D:\expecon_ms\data\behav\behav_df'

# save cleaned EEG data
save_dir_stim = r'D:\expecon_ms\data\eeg\prepro_stim\filter_0.1Hz'

if not os.path.exists(save_dir_stim):
    os.makedirs(save_dir_stim)
    print("Directory created:", save_dir_stim)

save_dir_cue = r'D:\expecon_ms\data\eeg\prepro_cue'

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016', '017',
          '018', '019', '020', '021', '022', '023', '024', '025', '026', '027', '028', 
          '029', '030', '031', '032', '033', '034', '035', '036', '037', '038', '039',
          '040','041', '042', '043', '044', '045', '046', '047', '048', '049')

def concatenate():

    """this function takes as input raw EEG data (brainvision files), 
    reads the event triggers from annotations
    and concatenates the raw data files to one big .fif file"""

    trigger_per_block = []

    for index, subj in enumerate(IDlist):
        # if file exists, skip
        if os.path.exists(f'D:\expecon_ms\data\eeg\\raw_concatenated\P{subj}_concatenated_raw.fif'):
            continue
        else:
            if subj == '018':
                raw_fname1 = f'expecon_EEG_{subj}_train.vhdr'  # wrongly named the first experimental block
                raw_fname2 = f'expecon_EEG_{subj}_02.vhdr'
                raw_fname3 = f'expecon_EEG_{subj}_03.vhdr'
                raw_fname4 = f'expecon_EEG_{subj}_04.vhdr'
                raw_fname5 = f'expecon_EEG_{subj}_05.vhdr'

            elif subj == '031':  # only 4 blocks
                raw_fname1 = f'expecon_EEG_{subj}_01.vhdr'
                raw_fname2 = f'expecon_EEG_{subj}_02.vhdr'
                raw_fname3 = f'expecon_EEG_{subj}_03.vhdr'
                raw_fname4 = f'expecon_EEG_{subj}_04.vhdr'

            else:
                raw_fname1 = f'expecon_EEG_{subj}_01.vhdr'
                raw_fname2 = f'expecon_EEG_{subj}_02.vhdr'
                raw_fname3 = f'expecon_EEG_{subj}_03.vhdr'
                raw_fname4 = f'expecon_EEG_{subj}_04.vhdr'
                raw_fname5 = f'expecon_EEG_{subj}_05.vhdr'

            # extract events from annotations and store trigger counts in list
            os.chdir(raw_dir)
                
            if subj == '031':

                raw_1 = mne.io.read_raw_brainvision(raw_fname1)
                raw_2 = mne.io.read_raw_brainvision(raw_fname2)
                raw_3 = mne.io.read_raw_brainvision(raw_fname3)
                raw_4 = mne.io.read_raw_brainvision(raw_fname4)

                events_1, event_dict = mne.events_from_annotations(raw_1, regexp='Stimulus/S  2')
                events_2, event_dict = mne.events_from_annotations(raw_2, regexp='Stimulus/S  2')
                events_3, event_dict = mne.events_from_annotations(raw_3, regexp='Stimulus/S  2')
                events_4, event_dict = mne.events_from_annotations(raw_4, regexp='Stimulus/S  2')

                trigger_per_block.append([len(events_1), len(events_2), len(events_3), len(events_4),
                  len(events_5)])

                raw = mne.concatenate_raws([raw_1, raw_2, raw_3, raw_4])

            else:
    
                raw_1 = mne.io.read_raw_brainvision(raw_fname1)
                raw_2 = mne.io.read_raw_brainvision(raw_fname2)
                raw_3 = mne.io.read_raw_brainvision(raw_fname3)
                raw_4 = mne.io.read_raw_brainvision(raw_fname4)
                raw_5 = mne.io.read_raw_brainvision(raw_fname5)

                events_1, event_dict = mne.events_from_annotations(raw_1,
                                                                   regexp='Stimulus/S  2')
                events_2, event_dict = mne.events_from_annotations(raw_2,
                                                                   regexp='Stimulus/S  2')
                events_3, event_dict = mne.events_from_annotations(raw_3,
                                                                   regexp='Stimulus/S  2')
                events_4, event_dict = mne.events_from_annotations(raw_4,
                                                                   regexp='Stimulus/S  2')
                events_5, event_dict = mne.events_from_annotations(raw_5,
                                                                   regexp='Stimulus/S  2')

                # check if we have 144 trigger per block (trials)
                # if not I forgot to turn on the EEG recording (or turned it on too late)

                trigger_per_block.append([len(events_1), len(events_2), len(events_3), len(events_4),
                  len(events_5)])

                raw = mne.concatenate_raws([raw_1, raw_2, raw_3, raw_4, raw_5])

            # save concatenated raw data
            raw.save(f'D:\expecon_ms\data\eeg\\raw_concatenated\P{subj}_concatenated_raw.fif')
    
    return trigger_per_block

def remove_trials(filename='raw_behav_data.csv'):  

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
    # (35 trials in total)

    # Create a list of tuples to store each condition
    conditions = [(7, 2, [1]),
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
                  (42, 5, [1])]

    # Iterate over the list of conditions and drop the rows for each condition
    for cond in conditions:
        df.drop(df.index[(df["ID"] == cond[0]) & (df["block"] == cond[1]) &
                         (df["trial"].isin(cond[2]))], inplace=True)

    df.to_csv('behav_cleaned_for_eeg.csv')

def prepro(trigger=0, l_freq=0.1, h_freq=40,
           tmin=-1.5, tmax=1.5, resample_rate=250, laplace=0, sf=2500,
           detrend=1, ransac=1, autoreject=0):

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

    for index, subj in enumerate(IDlist):

        # if file exists, skip
        if os.path.exists(f'{save_dir_stim}\\P{subj}_epochs_stim_0.1Hzfilter-epo.fif'):
            continue
        
        # load raw data concatenated for all blocks
        raw = mne.io.read_raw_fif(f'D:\expecon_ms\data\eeg\\raw_concatenated\P{subj}_concatenated_raw.fif',
                                  preload=True)
        
        # save the annotations (trigger) information
        anot.append(raw.annotations.to_data_frame())

        raw.set_channel_types({'VEOG': 'eog', 'ECG': 'ecg'})

        # setting montage from brainvision montage file

        raw.set_montage(montage)

        # filter the data
        raw.filter(l_freq=l_freq, h_freq=h_freq, method='iir')

        # load stimulus trigger events
        events, event_dict = mne.events_from_annotations(raw, regexp='Stimulus/S  2')

        # add cue as a trigger to event structure

        if trigger == 1:

            cue_timings = [i - int(0.4 * sf) for i in
                        events[:,
                        0]]

            # subtract 0.4*sampling frequency to get the cue time stamps

            cue_events = copy.deepcopy(events)
            cue_events[:, 0] = cue_timings
            cue_events[:, 2, ] = 2

        # add dataframe as metadata to epochs

        df_sub = df[df.ID == index + 7]

        metadata = df_sub

        # lock the data to the specified trigger and create epochs
        if trigger == 1:

            epochs = mne.Epochs(raw, cue_events, event_id=2, tmin=tmin,
                                tmax=tmax, preload=True, baseline=None, 
                                detrend=detrend, metadata=metadata)
        # or stimulus onset
        else:
            epochs = mne.Epochs(raw, events, event_id=1, tmin=tmin, tmax=tmax,
                                preload=True, baseline=None, detrend=detrend,
                                metadata=metadata)
            
        # resample the data (resampling before epoching jitters the trigger)

        epochs.resample(resample_rate)

        # pick only EEG channels for Ransac bad channel detection 
        picks = mne.pick_types(epochs.info, eeg=True, eog=False, ecg=False)

        # use RANSAC to detect bad channels
        # (autoreject interpolates bad channels and detects bad
        # epochs, takes quite long and removes a lot of epochs due to
        # blink artifacts)

        if ransac:
            
            print(f'Run ransac for {subj}')

            ransac = Ransac(verbose='progressbar', picks=picks, n_jobs=3)

            # which channels have been marked bad by RANSAC

            epochs = ransac.fit_transform(epochs)

            print('\n'.join(ransac.bad_chs_))

            ch_interp.append(ransac.bad_chs_)

        # detect bad epochs

        # now feed the clean channels into Autoreject to detect bad trials

        if autoreject:

            ar = AutoReject()

            epochs, reject_log = ar.fit_transform(epochs)

            os.chdir(save_dir_stim)

            reject_log.save(f'P_{subj}_reject_log.npz')

        if trigger == 1:

            os.chdir(save_dir_cue)

            epochs.save(f'P{subj}_epochs_cue-epo.fif')

        else:

            epochs.save(f'{save_dir_stim}\\P{subj}_epochs_stim_0.1Hzfilter-epo.fif')

            print(f'saved epochs for participant {subj}')

    ch_df = pd.DataFrame(ch_interp)

    ch_df.to_csv(f'{save_dir_stim}\\interpolated_channels.csv')

    print('Done with preprocessing and creating clean epochs')

    return ch_interp, anot

def channels_interp(df = None):
    """
    This function calculates the amount of channels interpolated per participant
    and the mean, std, min and max of channels interpolated across participants
    
    Parameters
    ----------
    df : pandas dataframe
        dataframe containing the amount of channels interpolated per participant
    Returns
    -------
    None.
    """

    df = pd.read_csv(f'{save_dir_stim}\\interpolated_channels.csv')
    
    df = df.drop(['Unnamed: 0'], axis=1)
    df['count_ch'] = df.count(axis=1)

    print(f"mean channels interpolated {df['count_ch'].mean()}")
    print(f"std of channels interpolated: {df['count_ch'].std()}")
    print(f"min channels interpolated: {df['count_ch'].min()}")
    print(f"max channels interpolated: {df['count_ch'].max()}")

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