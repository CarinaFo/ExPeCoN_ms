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
from __future__ import annotations

# %% Import
import subprocess
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import scipy
from mne.datasets import fetch_fsaverage

from expecon_ms.configs import PROJECT_ROOT, config, path_to

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/source/source_reco.py")  # == __file__

last_commit_date = subprocess.check_output(
    ["git", "log", "-1", "--format=%cd", "--follow", __file__path]
).decode("utf-8").strip()

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

# data paths
dir_clean_epochs = Path(path_to.data.eeg.preprocessed.ica.clean_epochs)

subj = "008"

# load cleaned epochs (after ica component rejection)
epochs = mne.read_epochs(dir_clean_epochs / f"P{subj}_epochs_after_ica_0.1Hzfilter-epo.fif")

# check alignment
if PLOT_ALIGNMENT:
    mne.viz.plot_alignment(epochs.info, trans_dir, subject=subject, dig=False, src=src,
                           subjects_dir=subjects_dir, verbose=True, meg=False,
                           eeg=True)

behav_path = Path(path_to.data.behavior.behavior_df)

# save paths for beamforming
beamformer_path = Path("./data/eeg/source/high_low_pre_beamformer")
figs_dics_path = Path("./figs/manuscript_figures/figure6_tfr_contrasts/source")
# save paths for mne
mne_path = Path("./data/eeg/source/high_low_pre_eLORETA")
save_dir_cluster_output = Path("./figs/eeg/sensor/cluster_permutation_output")

id_list = config.participants.ID_list


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

def run_source_reco(dics: int = 0, save_path: str | Path = mne_path, drop_bads: bool = True) -> None:
    """
    Run source reconstruction on epoched EEG data using eLoreta or DICS beamforming for frequency source analysis.

    Args:
    ----
    dics: 1 for DICS beamforming, 0 for eLoreta
    save_path: path to save source estimates
    drop_bads: if True, bad epochs are dropped

    Returns:
    -------
    .stc files for each hemisphere containing source reconstructions for each participant, shape: vertices-x-timepoints
    """
    for idx, subj in enumerate(id_list):

        # print participant ID
        print("Analyzing " + subj)

        # load cleaned epochs (after ica component rejection)
        epochs = mne.read_epochs(dir_clean_epochs / f"P{subj}_epochs_after_ica_0.1Hzfilter-epo.fif")

        ids_to_delete = [10, 12, 13, 18, 26, 30, 32, 32, 39, 40, 40, 30]  # TODO(simon): move to config.toml ?!
        blocks_to_delete = [6, 6, 4, 3, 4, 3, 2, 3, 3, 2, 5, 6]

        # Check if the participant ID is in the list of IDs to delete
        if pd.unique(epochs.metadata.ID) in ids_to_delete:

            # Get the corresponding blocks to delete for the current participant
            participant_blocks_to_delete = [
                block for id_, block in zip(ids_to_delete, blocks_to_delete) if id_ == pd.unique(epochs.metadata.ID)
            ]

            # Drop the rows with the specified blocks from the dataframe
            epochs = epochs[~epochs.metadata.block.isin(participant_blocks_to_delete)]

        # remove trials with rts >= 2.5 (no response trials) and trials with rts < 0.1
        epochs = epochs[epochs.metadata.respt1 >= 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # load behavioral data
        data = pd.read_csv(behav_path / "prepro_behav_data.csv")

        subj_data = data[idx + 7 == data.ID]

        # get drop log from epochs
        drop_log = epochs.drop_log

        search_string = "IGNORED"

        indices = [index for index, tpl in enumerate(drop_log) if tpl and search_string not in tpl]

        # drop bad epochs (too late recordings)
        if indices:
            epochs.metadata = subj_data.reset_index().drop(indices)
        else:
            epochs.metadata = subj_data

        # drop bad epochs
        if drop_bads:
            # drop epochs with abnormal strong signal (> 200 micro-volts)
            epochs.drop_bad(reject=dict(eeg=200e-6))

        # epochs for high and low probability condition
        epochs_high = epochs[epochs.metadata.cue == 0.75]
        epochs_low = epochs[epochs.metadata.cue == 0.25]

        if dics == 1:

            # We are interested in the beta band
            freqs = np.arange(15, 25, 1)

            # Computing the cross-spectral density matrix for the beta frequency band, for
            # different time intervals.
            # csd for all epochs
            csd = mne.time_frequency.csd_morlet(epochs, freqs, tmin=-0.4, tmax=0)
            # csd for high-prob trials only
            csd_a = mne.time_frequency.csd_morlet(epochs_high, freqs, tmin=-0.4, tmax=0)
            # csd for low-prob trials only
            csd_b = mne.time_frequency.csd_morlet(epochs_low, freqs, tmin=-0.4, tmax=0)

            info = epochs.info

            # To compute the source power for a frequency band, rather than each frequency
            # separately, we average the CSD objects across frequencies.
            csd_a = csd_a.mean()
            csd_b = csd_b.mean()

            # Computing DICS spatial filters using the CSD that was computed for all epochs
            filters = mne.beamformer.make_dics(info, fwd, csd, noise_csd=None,
                                               pick_ori="max-power", reduce_rank=True, real_filter=True)

            # Applying DICS spatial filters separately to each condition
            source_power_a, freqs = mne.beamformer.apply_dics_csd(csd_a, filters)
            source_power_b, freqs = mne.beamformer.apply_dics_csd(csd_b, filters)

            source_power_a.save(Path(save_path, f"high_beta_{subj}"))
            source_power_b.save(Path(save_path, f"low_beta_{subj}"))

        else:

            # average epochs for MNE
            evokeds_high = epochs_high.average()
            evokeds_low = epochs_low.average()

            evoked_contrast = mne.combine_evoked(all_evoked=[evokeds_high, evokeds_low], weights=[0.5, -0.5])
            evoked_contrast.crop(-0.4, tmax=0)
            # filter in beta band
            evoked_contrast.filter(l_freq=15, h_freq=25)

            # create noise covariance with a bias of data length
            # noise_cov = create_noise_cov(evokeds_high.data.shape, evokeds_high.info)

            # all epochs for noise covariance computation
            noise_cov = mne.compute_covariance(epochs, tmin=-0.4, tmax=0,
                                   method=["shrunk", "empirical"],
                                   rank="info")

            # save covariance matrix
            mne.write_cov(fname="covariance_prestim.cov", cov=noise_cov)

            # fixed forward solution for MNE methods
            fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True)

            info = evoked_contrast.info

            inv_op = mne.minimum_norm.make_inverse_operator(
                info,
                fwd_fixed,
                noise_cov,
                loose=0.2,
                depth=0.8
            )

            evoked_contrast.set_eeg_reference(projection=True)  # needed for inverse modeling

            method = "dSPM"
            snr = 3.
            lambda2 = 1. / snr ** 2  # regularization

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

        stc_low = np.array(stc_low_all)  # TODO(simon): now used, print(...) ?!
        stc_high = np.array(stc_high_all)  # TODO(simon): now used, print(...) ?!
        stc_array = np.array(stc_all)

    return stc_array


def extract_time_course_from_label():
    """Extract the time course from a label in source space."""
    # Get labels for FreeSurfer 'aparc' cortical parcellation with 75 labels/hemi
    labels_parc = mne.read_labels_from_annot("fsaverage", parc="aparc.a2009s", subjects_dir=subjects_dir)

    stc_all, stc_high_all, stc_low_all = [], [], []

    for idx, subj in enumerate(id_list):

        stc_high = mne.read_source_estimate(f"{path}high_{subj}")  # TODO(simon): path is not defined
        stc_low = mne.read_source_estimate(f"{path}low_{subj}")

        for stc in [stc_high, stc_low]:

            # extract activity in from source label
            # S1
            post_central_gyrus = mne.extract_label_time_course(  # TODO(simon): now used, print(...) ?!
                [stc], labels_parc[55], src, allow_empty=True)
            # S2
            g_front_inf_opercular_rh = mne.extract_label_time_course(  # TODO(simon): now used, print(...) ?!
                [stc], labels_parc[25], src, allow_empty=True)
            # ACC
            g_cingul_post_dorsal_rh = mne.extract_label_time_course(  # TODO(simon): now used, print(...) ?!
                [stc], labels_parc[19], src, allow_empty=True)


def spatio_temporal_source_test(
    data: np.ndarray | None = None,  # TODO(simon): now used?!
    n_perm: int = 10000,
    jobs: int = -1,
    save_path_source_figs: str | Path = figs_dics_path
):
    """
    Run a cluster-based permutation test over space and time.

    Args:
    ----
    data: 3D numpy array: participants x space x time
    n_perm: how many permutations for cluster test
    jobs: how many parallel GPUs should be used

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
    t, p, h = mne.stats.permutation_t_test(x_mean)  # TODO(simon): now used, print(...) ?!

    # mean over time and participants and plot contrast in source space
    x_avg = np.mean(x[:, :, :], axis=(0, 1))

    # put contrast or p values in source space
    fsave_vertices = [s["vertno"] for s in src]
    stc = mne.SourceEstimate(x_avg, tmin=-0.4, tstep=0.0001, vertices=fsave_vertices, subject="fsaverage")

    brain = stc.plot(
        hemi="rh",
        views="medial",
        subjects_dir=subjects_dir,
        subject="fsaverage",
        time_viewer=True,
        background="white"
    )

    brain.save_image(Path(save_path_source_figs, "t_values_high_low_rh_beamformer_dics.png"))

    # Here we set a cluster-forming threshold based on a p-value for the cluster-based permutation test.
    # We use a two-tailed threshold, the "1 - p_threshold" is needed, because for two-tailed tests
    # we must specify a positive threshold.

    p_threshold = 0.05
    t_threshold = scipy.stats.distributions.t.ppf(
        1 - p_threshold / 2,
        df=(len(id_list) - 2) - 1  # degrees of freedom for the test
    )

    # Now let's actually do the clustering. This can take a long time...

    print("Clustering.")

    T_obs, clusters, cluster_p_values, H0 = clu = mne.stats.spatio_temporal_cluster_1samp_test(
        x[:, :, :],
        adjacency=adjacency,
        threshold=t_threshold,
        n_jobs=jobs,
        n_permutations=n_perm
    )  # TODO(simon): now used, print(...) ?!

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
    good_clusters_idx = np.where(clu[2] < 0.05)[0]
    good_clusters = [clu[1][idx] for idx in good_clusters_idx]  # TODO(simon): now used, print(...) ?!
    print(min(clu[2]))

    print("Visualizing clusters.")

    # Now let's build a convenient representation of our results, where consecutive cluster spatial maps are stacked
    # in the time dimension of a SourceEstimate object.
    # This way by moving through the time dimension, we will be able to see subsequent cluster maps.
    fsave_vertices = [s["vertno"] for s in src]

    stc_all_cluster_vis = mne.stats.summarize_clusters_stc(
        clu,
        vertices=fsave_vertices,
        subject="fsaverage",
        p_thresh=0.1
    )

    # Let's actually plot the first "time point" in the SourceEstimate, which
    # shows all the clusters, weighted by duration.

    # blue blobs are for condition A < condition B, red for A > B

    brain = stc_all_cluster_vis.plot(
        hemi="rh", views="lateral", subjects_dir=subjects_dir,
        time_label="temporal extent (ms)", size=(800, 800),
        smoothing_steps=5, time_viewer=False,
        background="white", transparent=True, colorbar=False)

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


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
