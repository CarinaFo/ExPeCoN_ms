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
import scipy
from mne.datasets import fetch_fsaverage

from expecon_ms.configs import PROJECT_ROOT, config, params, path_to

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/source/source_reco.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)

print("Last Commit Date for", __file__path, ":", last_commit_date)

PLOT_ALIGNMENT = False

# load source space files
fs_dir = Path(fetch_fsaverage(verbose=True))
subjects_dir = fs_dir.parent
subject = "fsaverage"

_oct = "6"

fwd_dir = subjects_dir / f"{subject}/bem/{subject}-oct{_oct}-fwd.fif"
src_fname = subjects_dir / f"{subject}/bem/{subject}-oct{_oct}-src.fif"
trans_dir = subjects_dir / f"{subject}/bem/{subject}-trans.fif"

# Read the source space and the forward solution
src = mne.read_source_spaces(src_fname)
fwd = mne.read_forward_solution(fwd_dir)

# clean epochs
dir_clean_epochs = Path(path_to.data.eeg.preprocessed.ica.ICA)
dir_clean_epochs_expecon2 = Path(path_to.data.eeg.preprocessed.ica.clean_epochs_expecon2)

subj = "008"

# load cleaned epochs (after ica component rejection)
epochs = mne.read_epochs(dir_clean_epochs_expecon2 / f"P{subj}_icacorr_0.1Hz-epo.fif")

# check alignment
if PLOT_ALIGNMENT:
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

behav_path = Path(path_to.data.behavior)

# save paths for beamforming
beamformer_path = Path("./data/eeg/source/high_low_pre_beamformer")
figs_dics_path = Path("./figs/manuscript_figures/figure6_tfr_contrasts/source")

# save paths for mne
mne_path = Path("./data/eeg/source/high_low_pre_eLORETA")
save_dir_cluster_output = Path("./figs/eeg/sensor/cluster_permutation_output")

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
# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def run_source_reco(study: int = 1, cond: str = "probability",
                    dics: int = 0, fmin=15, fmax=25, tmin=-0.2, tmax=0,
                    save_path: str | Path = mne_path, 
                    drop_bads: bool = True) -> None:
    """
    Run source reconstruction on epoched EEG data using eLoreta or DICS beamforming for frequency source analysis.

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
            cond_a = "high"
            cond_b = "low"
        elif cond == "prev_resp":
            epochs_a = epochs[
                (epochs.metadata.prevresp == 1)
            ]
            epochs_b = epochs[(epochs.metadata.prevresp == 0)]
            cond_a = "prevyesresp"
            cond_b = "prevnoresp"
        else:
            raise ValueError("input should be 'probability' or 'prev_resp'")

        # make sure we have an equal amount of trials in both conditions
        mne.epochs.equalize_epoch_counts([epochs_a, epochs_b])

        if dics == 1:
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
                info, fwd, csd, noise_csd=None, pick_ori="max-power", reduce_rank=True, real_filter=True
            )

            # Applying DICS spatial filters separately to each condition
            source_power_a, freqs = mne.beamformer.apply_dics_csd(csd_a, filters)
            source_power_b, freqs = mne.beamformer.apply_dics_csd(csd_b, filters)

            source_power_a.save(Path(save_path, f"{cond_a}_{subj}"))
            source_power_b.save(Path(save_path, f"{cond_b}_{subj}"))

        else:
            # average epochs for MNE
            evokeds_a = epochs_a.average()
            evokeds_b = epochs_b.average()

            evoked_contrast = mne.combine_evoked(all_evoked=[evokeds_a, evokeds_b], weights=[0.5, -0.5])
            evoked_contrast.crop(-0.4, tmax=0)
            # filter in beta band
            evoked_contrast.filter(l_freq=fmin, h_freq=fmax)

            # create noise covariance with a bias of data length
            # noise_cov = create_noise_cov(evokeds_high.data.shape, evokeds_high.info)

            # all epochs for noise covariance computation
            noise_cov = mne.compute_covariance(epochs, tmin=tmin, tmax=tmax, method=["shrunk", "empirical"], rank="info")

            # save covariance matrix
            mne.write_cov(fname="covariance_prestim.cov", cov=noise_cov)

            # fixed forward solution for MNE methods
            fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True)

            info = evoked_contrast.info

            inv_op = mne.minimum_norm.make_inverse_operator(info, fwd_fixed, noise_cov, loose=0.2, depth=0.8)

            evoked_contrast.set_eeg_reference(projection=True)  # needed for inverse modeling

            method = "dSPM"
            snr = 3.0
            lambda2 = 1.0 / snr**2  # regularization

            stc = mne.minimum_norm.apply_inverse(evoked_contrast, inv_op, lambda2, method=method, pick_ori=None)

            stc.save(Path(save_path, f"contrast_highlow_{subj}"))


def create_source_contrast_array(path_to_source: str | Path = mne_path):
    """
    Load source estimates per participant and contrasts them, before storing the contrast in a numpy array.

    Args:
    ----
    path_to_source: path to source estimates

    Returns:
    -------
    shape of the numpy array: participants-x-vertices-x-timepoints
    """
    path_to_source = Path(path_to_source)
    if path_to_source == mne_path:
        stc_all = []

        for subj in id_list:
            stc = mne.read_source_estimate(path_to_source / f"contrast_highlow_{subj}")

            stc_all.append(stc.data)

        stc_array = np.array(stc_all)

    else:
        stc_all, stc_high_all, stc_low_all = [], [], []

        for subj in id_list:
            stc_high = mne.read_source_estimate(path_to_source / f"high_beta_{subj}")
            stc_low = mne.read_source_estimate(path_to_source / f"low_beta_{subj}")

            stc_diff = stc_high.data - stc_low.data

            stc_high_all.append(stc_high.data)
            stc_low_all.append(stc_low.data)
            stc_all.append(stc_diff)

        stc_low = np.array(stc_low_all)  # TODO(simon): not used, print(...) ?!
        stc_high = np.array(stc_high_all)  # TODO(simon): not used, print(...) ?!
        stc_array = np.array(stc_all)

    return stc_array


def extract_time_course_from_label():
    """Extract the time course from a label in source space."""
    # Get labels for FreeSurfer 'aparc' cortical parcellation with 75 labels/hemi
    labels_parc = mne.read_labels_from_annot("fsaverage", parc="aparc.a2009s", subjects_dir=subjects_dir)

    stc_all, stc_high_all, stc_low_all = [], [], []  # TODO(simon): not used, print(...) ?!

    for idx, subj in enumerate(id_list):
        stc_high = mne.read_source_estimate(f"{path}high_{subj}")  # TODO(simon): path is not defined
        stc_low = mne.read_source_estimate(f"{path}low_{subj}")

        for stc in [stc_high, stc_low]:
            # extract activity in from source label
            # S1
            post_central_gyrus = mne.extract_label_time_course(  # TODO(simon): not used, print(...) ?!
                [stc], labels_parc[55], src, allow_empty=True
            )
            # S2
            g_front_inf_opercular_rh = mne.extract_label_time_course(  # TODO(simon): not used, print(...) ?!
                [stc], labels_parc[25], src, allow_empty=True
            )
            # ACC
            g_cingul_post_dorsal_rh = mne.extract_label_time_course(  # TODO(simon): not used, print(...) ?!
                [stc], labels_parc[19], src, allow_empty=True
            )


def spatio_temporal_source_test(
    data: np.ndarray | None = None,  # TODO(simon): not used?!
    n_perm: int = 10000,
    jobs: int = -1,
    save_path_source_figs: str | Path = figs_dics_path,
):
    """
    Run a cluster-based permutation test over space and time.

    Args:
    ----
    data: 3D numpy array: participants x space x time
    n_perm: how many permutations for cluster test
    jobs: how many parallel GPUs should be used
    save_path_source_figs: path to save figures

    Returns:
    -------
    cluster output
    """
    print("Computing adjacency.")

    adjacency = mne.spatial_src_adjacency(src)

    # Note that X needs to be a multidimensional array of shape
    # observations (subjects) × time × space, so we permute dimensions
    # read data in for expecon 2
    data = np.load(Path("./data/eeg/source/high_low_pre_beamformer/expecon2/source_beta_highlow.npy"))
    # TODO(simon): move to config

    x = np.transpose(data, [0, 2, 1])

    x_mean = np.mean(x[:, :, :], axis=1)

    # mean over time and permutation test to get sign. vertices
    t, p, h = mne.stats.permutation_t_test(x_mean)  # TODO(simon): not used, print(...) ?!

    # mean over time and participants and plot contrast in source space
    x_avg = np.mean(x[:, :, :], axis=(0, 1))

    # put contrast or p values in source space
    fsave_vertices = [s["vertno"] for s in src]
    stc = mne.SourceEstimate(x_avg, tmin=-0.4, tstep=0.0001, vertices=fsave_vertices, subject="fsaverage")

    brain = stc.plot(
        hemi="rh", views="medial", subjects_dir=subjects_dir, subject="fsaverage", time_viewer=True, background="white"
    )

    brain.save_image(Path(save_path_source_figs, "t_values_high_low_rh_beamformer_dics.png"))

    # Here we set a cluster-forming threshold based on a p-value for the cluster-based permutation test.
    # We use a two-tailed threshold, the "1 - p_threshold" is needed, because for two-tailed tests
    # we must specify a positive threshold.

    p_threshold = params.alpha
    t_threshold = scipy.stats.distributions.t.ppf(
        1 - p_threshold / 2, df=(len(id_list) - 2) - 1  # degrees of freedom for the test
    )

    # Now let's actually do the clustering. This can take a long time...

    print("Clustering.")

    t_obs, clusters, cluster_p_values, h0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(
        x[:, :, :], adjacency=adjacency, threshold=t_threshold, n_jobs=jobs, n_permutations=n_perm
    )  # TODO(simon): not used, print(...) ?!

    return clu


def plot_cluster_output(clu=None):
    """
    Plot the cluster output of the cluster permutation test.

    Args:
    ----
    clu: cluster output

    Returns:
    -------
    plot of cluster output
    """
    # Select the clusters that are statistically significant at p < 0.05
    good_clusters_idx = np.where(clu[2] < params.alpha)[0]
    good_clusters = [clu[1][idx] for idx in good_clusters_idx]  # TODO(simon): not used, print(...) ?!
    print(min(clu[2]))

    print("Visualizing clusters.")

    # Now let's build a convenient representation of our results, where consecutive cluster spatial maps are stacked
    # in the time dimension of a SourceEstimate object.
    # This way by moving through the time dimension, we will be able to see subsequent cluster maps.
    fsave_vertices = [s["vertno"] for s in src]

    stc_all_cluster_vis = mne.stats.summarize_clusters_stc(
        clu, vertices=fsave_vertices, subject="fsaverage", p_thresh=0.1
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
        time_viewer=False,
        background="white",
        transparent=True,
        colorbar=False,
    )

    brain.save_image(Path("./data/eeg/source/cluster_rh_lateral.png"))  # TODO(simon): move to config.toml


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
# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
