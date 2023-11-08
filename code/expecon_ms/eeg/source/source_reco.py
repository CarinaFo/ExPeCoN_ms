#!/usr/bin/python3
"""
The script contains functions that source reconstruct 62 channel EEG data.

It is based on MNE methods (e.g., eLORETA) or beamforming for time-frequency.

Moreover, the script includes functions for statistical analysis in source space:
    permutation t-test or cluster permutation test in source space

Also, it includes a function to plot contrasts in source space.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023
"""
# %% Import
from __future__ import annotations

import subprocess
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from mne.datasets import fetch_fsaverage
import random

from expecon_ms.configs import PROJECT_ROOT, config, params, path_to

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/source/source_reco.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)

print("Last Commit Date for", __file__path, ":", last_commit_date)

#
save_dir_fsaverage_source_files = Path("E:/expecon_ms/data/templates")

# fetch fsaverage files and save path
subjects_dir = fetch_fsaverage()

# set root path to fsaverag files
fs_average_root_path = f'{subjects_dir}{Path("/")}bem{Path("/")}'

# load bem solution, source space and transformation matrix
bem = f'{fs_average_root_path}fsaverage-5120-5120-5120-bem-sol.fif'
src_fname = f'{fs_average_root_path}fsaverage-ico-5-src.fif'
trans_dir = f'{fs_average_root_path}fsaverage-trans.fif'

# Read the source space
src = mne.read_source_spaces(src_fname)

# clean epochs path
dir_clean_epochs = Path(path_to.data.eeg.preprocessed.ica.ICA)
dir_clean_epochs_expecon2 = Path(path_to.data.eeg.preprocessed.ica.clean_epochs_expecon2)

# load example epoch
epochs = mne.read_epochs(Path(dir_clean_epochs / f"P015_icacorr_0.1Hz-epo.fif"))

# set up forward solution
fwd = mne.make_forward_solution(
    epochs.info, trans=trans_dir, src=src, bem=bem, eeg=True, 
    mindist=5.0,
    n_jobs=None
)

# behavioral data
behav_path = Path(path_to.data.behavior)

# save paths for beamforming
beamformer_path = Path(path_to.data.eeg.source.beamformer)

# save paths for mne
mne_path = Path(path_to.data.eeg.source.mne)

# save source space figures
save_path_source_figs = Path(path_to.figures.manuscript.figure4_source)

# participant IDs
id_list_expecon1 = config.participants.ID_list_expecon1
id_list_expecon2 = config.participants.ID_list_expecon2

# data_cleaning parameters defined in config.toml
rt_max = config.behavioral_cleaning.rt_max
rt_min = config.behavioral_cleaning.rt_min
hitrate_max = config.behavioral_cleaning.hitrate_max
hitrate_min = config.behavioral_cleaning.hitrate_min
farate_max = config.behavioral_cleaning.farate_max
hit_fa_diff = config.behavioral_cleaning.hit_fa_diff

# pilot data counter for expecon 1
pilot_counter = config.participants.pilot_counter
# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def run_source_reco(study: int, 
                    cond: str,
                    dics: int, 
                    fmin: int, 
                    fmax: int,
                    tmin: int, 
                    tmax: int, 
                    save_path: str,
                    drop_bads: bool,
                    plot_alignment: bool) -> None:
    """
    Run source reconstruction on epoched EEG data using 
    eLoreta or DICS beamforming for oscillatory source analysis.

    Args:
    ----
    study : int, info: which study to analyze: 1 (block, stable environment) or 2 (trial,
    volatile environment)
    cond : str, info: which condition to analyze: "probability" or "prev_resp"
    fmin: int, info: lower frequency bound for DICS beamforming
    fmax: int, info: upper frequency bound for DICS beamforming
    tmin: int, info: lower time bound for DICS beamforming
    tmax: int, info: upper time bound for DICS beamforming
    dics: 1 for DICS beamforming, 0 for eLoreta
    save_path: path to save source estimates
    drop_bads: if True, bad epochs are dropped
    plot_alignment: if True, plot alignment of electrodes with source space

    Returns:
    -------
    .stc files for each hemisphere containing source reconstructions for each participant, shape: vertices-x-timepoints
    """

    if study == 1:
        id_list = id_list_expecon1
        # load behavioral data
        data = pd.read_csv(Path(path_to.data.behavior, "prepro_behav_data.csv"))

    elif study == 2:
        id_list = id_list_expecon2
        # load behavioral data
        data = pd.read_csv(Path(path_to.data.behavior,
                                "prepro_behav_data_expecon2.csv"))
    else:
        raise ValueError("input should be 1 or 2 for the respective study")

    if plot_alignment:
        plot_source_space_electrodes_alignment()

    # now loop over participants
    for idx, subj in enumerate(id_list):

        # print participant ID
        print("Analyzing " + subj)

        # load clean epochs (after ica component rejection)
        if study == 1:
            epochs = mne.read_epochs(
                Path(dir_clean_epochs / f"P{subj}_icacorr_0.1Hz-epo.fif"))
        elif study == 2:
            # skip ID 13
            if subj == '013':
                continue
            epochs = mne.read_epochs(dir_clean_epochs_expecon2 / f"P{subj}_icacorr_0.1Hz-epo.fif")
            # rename columns
            epochs.metadata = epochs.metadata.rename(columns ={"resp1_t": "respt1",
                                                               "stim_type": "isyes",
                                                               "resp1": "sayyes"})
        else:
            raise ValueError("input should be 1 or 2 for the respective study")

        # clean epochs (remove blocks based on hit and false alarm rates, reaction times, etc.)
        epochs = drop_trials(data=epochs)

        # get behavioral data for current participant
        if study == 1:
            subj_data = data[idx + pilot_counter == data.ID]
        elif study == 2:
            subj_data = data[idx + 1 == data.ID]
        else:
            raise ValueError("input should be 1 or 2 for the respective study")

        # get drop log from epochs
        drop_log = epochs.drop_log

        # Ignored bad epochs are those defined by the user as bad epochs
        search_string = "IGNORED"
        # remove trials where epochs are labelled as too short
        indices = [index for index, tpl in enumerate(drop_log) if tpl and search_string not in tpl]

        # drop trials without corresponding epoch
        if indices:
            epochs.metadata = subj_data.reset_index().drop(indices)
        else:
            epochs.metadata = subj_data

        if drop_bads:
            # drop epochs with abnormal strong signal (> 200 micro-volts)
            epochs.drop_bad(reject=dict(eeg=200e-6))

        if cond == "probability":
            epochs_a = epochs[
                (epochs.metadata.cue == 0.75)
            ]
            epochs_b = epochs[(epochs.metadata.cue == 0.25)]
            cond_a_name = "high"
            cond_b_name = "low"
        elif cond == "prev_resp":
            epochs_a = epochs[((epochs.metadata.prevresp == 1) &
                                (epochs.metadata.cue == 0.75))]
            epochs_b = epochs[((epochs.metadata.prevresp == 0) &
                                (epochs.metadata.cue == 0.75))]
            cond_a_name = "prevyesresp_highprob"
            cond_b_name = "prevnoresp_highprob"
        else:
            raise ValueError("input should be 'probability' or 'prev_resp'")

        # make sure we have an equal amount of trials in both conditions
        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        if dics == 1:
            
            # save path for source estimates
            save_path = beamformer_path

            # We are interested in the beta band
            freqs = np.arange(fmin, fmax, 1)

            # Computing the cross-spectral density matrix for the beta frequency band, for
            # different time intervals.
            # csd for all epochs
            csd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=tmin, tmax=tmax)
            # csd for high-prob trials only
            csd_a = mne.time_frequency.csd_morlet(epochs_a, freqs, tmin=tmin, tmax=tmax)
            # csd for low-prob trials only
            csd_b = mne.time_frequency.csd_morlet(epochs_b, freqs, tmin=tmin, tmax=tmax)

            info = epochs.info

            # To compute the source power for a frequency band, rather than each frequency
            # separately, we average the CSD objects across frequencies.
            csd_a = csd_a.mean()
            csd_b = csd_b.mean()

            # Computing DICS spatial filters using the CSD that was computed for all epochs
            filters = mne.beamformer.make_dics(
                info, fwd, csd, noise_csd=None, 
                pick_ori="max-power", reduce_rank=True, real_filter=True
            )

            # Applying DICS spatial filters separately to each condition
            source_power_a, freqs = mne.beamformer.apply_dics_csd(csd_a, filters)
            source_power_b, freqs = mne.beamformer.apply_dics_csd(csd_b, filters)

            source_power_a.save(Path(save_path, f"{cond_a_name}_{subj}_{study}"))
            source_power_b.save(Path(save_path, f"{cond_b_name}_{subj}_{study}"))

        else:

            save_path = mne_path

            # average epochs for MNE
            evokeds_a = epochs_a.average()
            evokeds_b = epochs_b.average()

            evoked_contrast = mne.combine_evoked(all_evoked=[evokeds_a, evokeds_b], 
                                                 weights=[0.5, -0.5])
            evoked_contrast.crop(tmin, tmax)
            # filter in beta band
            evoked_contrast.filter(l_freq=fmin, h_freq=fmax)

            # create noise covariance with a bias of data length
            # noise_cov = create_noise_cov(evokeds_high.data.shape, evokeds_high.info)

            # all epochs for noise covariance computation
            noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax,
                                                method=["shrunk", "empirical"],
                                                  rank="info")

            # save covariance matrix
            mne.write_cov(fname="covariance_prestim.cov", cov=noise_cov)

            # fixed forward solution for MNE methods
            fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True)

            info = evoked_contrast.info

            inv_op = mne.minimum_norm.make_inverse_operator(info, fwd_fixed, 
                                                            noise_cov, loose=0.2, 
                                                            depth=0.8)

            evoked_contrast.set_eeg_reference(projection=True)  # needed for inverse modeling

            method = "dSPM"
            snr = 3.0
            lambda2 = 1.0 / snr**2  # regularization

            stc = mne.minimum_norm.apply_inverse(evoked_contrast, inv_op, lambda2, 
                                                 method=method, pick_ori=None)

            stc.save(Path(save_path, f"contrast_{cond}_{subj}_{study}"))


def create_source_contrast_array(study: int,
                                 cond_a: str,
                                 cond_b: str,
                                 path_to_source: str):
    """
    Load source estimates per participant and contrasts them, before storing the 
    contrast in a numpy array.
    Args:
    ----
    study: int, info: which study to analyze: 1 (block, stable environment) or 2 (trial)
    cond: str, info: which condition to analyze: "probability" or "prev_resp"
    cond_a: str, info: name of condition a
    cond_b: str, info: name of condition b
    path_to_source: path to source estimates: e.g. | Path = beamformer_path

    Returns:
    -------
    shape of the numpy array: participants-x-vertices-x-timepoints
    """

    # load id list for respective study
    if study == 1:
        id_list = id_list_expecon1
    elif study == 2:
        id_list = id_list_expecon2
    else:
        raise ValueError("input should be 1 or 2 for the respective study")

    stc_all = []
    # loop over participants
    for subj in id_list:
        if study == 2:
            if subj == '013':
                continue
        # load source estimates
        stc_high = mne.read_source_estimate(path_to_source / f"{cond_a}_{subj}_{study}")
        stc_low = mne.read_source_estimate(path_to_source / f"{cond_b}_{subj}_{study}")
        # compute difference between conditions
        stc_diff = stc_high.data - stc_low.data
        # append to list
        stc_all.append(stc_diff)
    # convert list to numpy array
    stc_array = np.array(stc_all)

    return stc_array


def spatio_temporal_source_test(
    stc_array: np.ndarray,
    n_perm: int,
    jobs: int,
    save_path_source_figs: str,
    cond: str,
    method: str,
    study: int
):
    """
    Run a cluster-based permutation test over space and time.

    Args:
    ----
    stc_array: 3D numpy source array: participants x space x time
    n_perm: how many permutations for cluster test
    jobs: how many parallel GPUs should be used
    save_path_source_figs: path to save figures, e.g. | Path = beamformer_path
    cond: str, info: which condition to analyze: "probability" or "prev_resp"
    method: str, info: which method to analyze: "beamformer" or "mne"
    study: int, info: which study to analyze: 1 (block, stable environment) or 2 (trial)

    Returns:
    -------
    cluster output
    """

    print("Computing adjacency.")
    # get adjacency matrix for source space
    adjacency = mne.spatial_src_adjacency(src)

    # Note that X needs to be a multidimensional array of shape
    # observations (subjects) × time × space, so we permute dimensions
    x = np.transpose(stc_array, [0, 2, 1])

    # get rid of single-dimensional entries
    x_mean = np.squeeze(x)

    # permutation test to get sign. vertices
    t, p, h = mne.stats.permutation_t_test(x_mean)

    print(f"% of significant vertices: {np.sum(p < params.alpha) / len(p)}")

    # mean over participants
    x_avg = np.squeeze(np.mean(x, axis=0))

    # put contrast or p values in source space
    fsave_vertices = [s["vertno"] for s in src]
    stc = mne.SourceEstimate(x_avg, tmin=-0.4, tstep=0.0001, vertices=fsave_vertices,
     subject="fsaverage")
    
    # which hemisphere to plot
    hemisphere = "rh"

    view = 'lateral'

    # plot average source or t values
    brain = stc.plot(
        hemi=hemisphere, views=view, subjects_dir=subjects_dir, subject="fsaverage",
         time_viewer=True, background="white", colorbar=False
    )

    brain.save_image(f'{save_path_source_figs}{Path("/")}grand_average_{cond}_{method}_{study}_' +
                     f'_{view}_{hemisphere}.png')

    # spatio-temporal cluster permutation test
    clu = mne.stats.spatio_temporal_cluster_1samp_test(
        x, adjacency=adjacency, n_jobs=jobs, n_permutations=n_perm)

    return clu


def plot_cluster_output(clu: list, 
                        cond: str, 
                        freq_band: str, 
                        method: str):
    """
    Plot significant clusters of a spatio-temporal cluster permutation test.
    Args:
    ----
    clu: list, output of spatio-temporal cluster permutation test
    cond: str, info: which condition to analyze: "probability" or "prev_resp"
    freq_band: str, info: which frequency band to analyze: "alpha" or "beta"
    method: str, info: which method to analyze: "beamformer" or "mne"

    Returns:
    -------
    plot of cluster output
    """
    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_idx = np.where(clu[2] < params.alpha)[0]

    # check if there are sign. cluster to plot, otherwise break function
    if len(good_clusters_idx) == 0:
        return "No significant clusters."
    else:

        fsave_vertices = [s["vertno"] for s in src]

        # summarize cluster perm test output and prepare for visualization
        stc_all_cluster_vis = mne.stats.summarize_clusters_stc(
            clu, vertices=fsave_vertices, subject="fsaverage", p_thresh=params.alpha
        )

        # Let's actually plot the first "time point" in the SourceEstimate, which
        # shows all the clusters, weighted by duration.
        # blue blobs are for condition A < condition B, red for A > B
        brain = stc_all_cluster_vis.plot(
            hemi="rh",
            views="lateral",
            subjects_dir=subjects_dir,
            time_label="temporal extent (ms)",
            size=(800, 800),
            smoothing_steps=5,
            time_viewer=True,
            background="white",
            transparent=True,
            colorbar=True,
        )

        brain.save_image(Path(save_path_source_figs / "cluster_{cond}_{freq_band}_{method}.png"))


def create_noise_cov(data_size: tuple[int, int], data_info: mne.Info) -> mne.Covariance:
    """
    Compute identity noise covariance with a bias of data length.

    This method has been developed by Mina Jamshidi Idaji (minajamshidi91@gmail.com)

    Args:
    ----
    data_size: size of original data (dimensions - 1D)
    data_info: info that corresponds to the original data

    Returns:
    -------
    noise covariance for further source reconstruction
    """
    data1 = np.random.normal(loc=0.0, scale=1.0, size=data_size)
    raw1 = mne.io.RawArray(data1, data_info)
    return mne.compute_raw_covariance(raw1, tmin=0, tmax=None)

# Helper functions

def plot_source_space_electrodes_alignment():

    """ Plot the alignment of the electrodes with the source space."""
    
    # select a random subject
    random_subj = random.choice(config.participants.ID_list_expecon2)

    # load cleaned epochs to extract info for plotting alignment
    epochs = mne.read_epochs(dir_clean_epochs_expecon2 / f"P{random_subj}_icacorr_0.1Hz-epo.fif")

    mne.viz.plot_alignment(
        epochs.info,
        trans_dir,
        subject=subject,
        dig=False,
        src=src,
        subjects_dir=subjects_dir,
        verbose=True,
        meg=False,
        eeg=True,
    )

def drop_trials(data=None):
    """
    Drop trials based on behavioral data.

    Args:
    ----
    data: mne.Epochs, epoched data
    
    Returns:
    -------
    data: mne.Epochs, epoched data
    """

    # store number of trials before rt cleaning
    before_rt_cleaning = len(data.metadata)

    # remove no response trials or super fast responses
    data = data[data.metadata.respt1 != rt_max]
    data = data[data.metadata.respt1 > rt_min]

    # print rt trials dropped
    rt_trials_removed = before_rt_cleaning - len(data.metadata)

    print("Removed trials based on reaction time: ", rt_trials_removed)
    # Calculate hit rates per participant
    signal = data[data.metadata.isyes == 1]
    hitrate_per_subject = signal.metadata.groupby(['ID'])['sayyes'].mean()

    print(f"Mean hit rate: {np.mean(hitrate_per_subject):.2f}")

    # Calculate hit rates by participant and block
    hitrate_per_block = signal.metadata.groupby(['ID', 'block'])['sayyes'].mean()

    # remove blocks with hitrates > 90 % or < 20 %
    filtered_groups = hitrate_per_block[(hitrate_per_block > hitrate_max) | (hitrate_per_block < hitrate_min)]
    print('Blocks with hit rates > 0.9 or < 0.2: ', len(filtered_groups))

    # Extract the ID and block information from the filtered groups
    remove_hitrates = filtered_groups.reset_index()

    # Calculate false alarm rates by participant and block
    noise = data[data.metadata.isyes == 0]
    farate_per_block = noise.metadata.groupby(['ID', 'block'])['sayyes'].mean()

    # remove blocks with false alarm rates > 0.4
    filtered_groups = farate_per_block[farate_per_block > farate_max]
    print('Blocks with false alarm rates > 0.4: ', len(filtered_groups))

    # Extract the ID and block information from the filtered groups
    remove_farates = filtered_groups.reset_index()

    # Hitrate should be > false alarm rate
    filtered_groups = hitrate_per_block[hitrate_per_block - farate_per_block < hit_fa_diff]
    print('Blocks with hit rates < false alarm rates: ', len(filtered_groups))
          
    # Extract the ID and block information from the filtered groups
    hitvsfarate = filtered_groups.reset_index()

    # Concatenate the dataframes
    combined_df = pd.concat([remove_hitrates, remove_farates, hitvsfarate])

    # Remove duplicate rows based on 'ID' and 'block' columns
    unique_df = combined_df.drop_duplicates(subset=['ID', 'block'])

    # Merge the big dataframe with unique_df to retain only the non-matching rows
    data.metadata = data.metadata.merge(unique_df, on=['ID', 'block'], how='left',
                             indicator=True,
                             suffixes=('', '_y'))
    
    data = data[data.metadata['_merge'] == 'left_only']

    # Remove the '_merge' column
    data.metadata = data.metadata.drop('_merge', axis=1)

    return data

# Unused functions

def extract_time_course_from_label(data=None):

    """Extract the time course from a label in source space.
    
    Args:
    ----
    data: mne.SourceEstimate, source estimates

    Returns:
    -------
    time course for each label
    """
    
    # this extracts a certain brain area
    label_s1 = "rh.BA3a"
    fname_labels1 = subjects_dir / f"fsaverage/label/{label_s1}.label"
    labels1 = mne.read_label(str(fname_labels1))
    label_s2 = "rh.BA3b"
    fname_labels2 = subjects_dir / f"fsaverage/label/{label_s2}.label"
    labels2 = mne.read_label(str(fname_labels2))  
    label_aparc = "rh.aparc"
    fname_label_aparc = subjects_dir / f"fsaverage/label/{label_aparc}.label"
    label_ap = mne.read_label(str(fname_label_aparc))  

    # Get labels for FreeSurfer 'aparc' cortical parcellation with 75 labels/hemi
    labels_parc = mne.read_labels_from_annot("fsaverage", parc="aparc.a2009s", subjects_dir=subjects_dir)

    # extract activity in from source label
    # S1
    post_central_gyrus = mne.extract_label_time_course(
        [stc], labels_parc[55], src, allow_empty=True
    )
    # S2
    g_front_inf_opercular_rh = mne.extract_label_time_course(
        [stc], labels_parc[25], src, allow_empty=True
    )
    # ACC
    g_cingul_post_dorsal_rh = mne.extract_label_time_course(
        [stc], labels_parc[19], src, allow_empty=True
    )

    return post_central_gyrus, g_front_inf_opercular_rh, g_cingul_post_dorsal_rh
# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
