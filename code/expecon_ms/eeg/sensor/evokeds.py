#!/usr/bin/python3
"""
The script provides functions to analyze cleaned epochs and compute evoked signals from different conditions.

This includes permutation cluster tests.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023
"""
# %% Import
import pickle
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mne
import numpy as np
import pandas as pd

from expecon_ms.behav import figure1
from expecon_ms.configs import PROJECT_ROOT, config, params, path_to

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/sensor/evokeds.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# for plotting in new window (copy to interpreter)
# %matplotlib qt
# for inline plotting
# %matplotlib inline

# set font to Arial and font size to 14
plt.rcParams.update({"font.size": 14, "font.family": "sans-serif", "font.sans-serif": "Arial"})

# directory that contains the cleaned epochs
dir_clean_epochs = Path(path_to.data.eeg.preprocessed.ica.ICA)

# behavioral data for each epoch
behav_path = Path(path_to.data.behavior)

# participant IDs
id_list = config.participants.ID_list

# pilot data counter
pilot_counter = config.participants.pilot_counter

# blocks to delete
ids_to_delete = config.behavioral_cleaning.ids_to_delete
blocks_to_delete = config.behavioral_cleaning.blocks_to_delete
# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def create_contrast(drop_bads: bool = True, laplace: bool = False, 
                    subtract_evoked: bool = False,
                    save_data_to_disk: bool = False):
    """
    Load cleaned epoched data and create contrasts.

    Args:
    ----
    drop_bads: boolean, drop bad epochs if True
    laplace: apply CSD to data if boolean is True
    subtract_evoked: boolean, subtract evoked signal from each epoch
    save_data_to_disk: boolean, save data to disk if True

    Returns:
    -------
    list of condition evokeds
    """
    # store trials remaining and removed per participant
    all_trials, trials_removed = [], []

    # save  evoked data per participant
    evokeds_signal_all, evokeds_noise_all, evokeds_hit_all, evokeds_miss_all = [], [], [], []

    # metadata after epoch cleaning
    metadata_allsubs = []

    for idx, subj in enumerate(id_list):
        # print participant idx
        print(f"Participant {idx!s}")

        # load cleaned epochs
        epochs = mne.read_epochs(dir_clean_epochs / f"P{subj}_icacorr_0.1Hz-epo.fif")

        # Check if the participant ID is in the list of IDs to delete
        if pd.unique(epochs.metadata.ID) in ids_to_delete:
            # Get the corresponding blocks to delete for the current participant
            participant_blocks_to_delete = [
                block for id_, block in zip(ids_to_delete, blocks_to_delete) if id_ == pd.unique(epochs.metadata.ID)
            ]

            # Drop the rows with the specified blocks from the dataframe
            epochs = epochs[~epochs.metadata.block.isin(participant_blocks_to_delete)]

        # remove trials with rts >= 2.5 (no response trials) and trials with rts < 0.1
        before_rt_removal = len(epochs.metadata)
        epochs = epochs[epochs.metadata.respt1 >= 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # save n_trials per participant
        all_trials.append(len(epochs.metadata))

        trials_removed.append(before_rt_removal - len(epochs.metadata))

        # load behavioral data
        data = pd.read_csv(f'{behav_path}{Path("/")}prepro_behav_data.csv')

        subj_data = data[idx + pilot_counter == data.ID]

        # get drop log from epochs
        drop_log = epochs.drop_log

        search_string = "IGNORED"

        indices = [index for index, tpl in enumerate(drop_log) if tpl and search_string not in tpl]

        # drop bad epochs (too late recordings)
        if indices:
            epochs.metadata = subj_data.reset_index().drop(indices)
        else:
            epochs.metadata = subj_data

        if subtract_evoked:
            epochs.subtract_evoked()

        # reject epochs that exceed a certain threshold (amplitude)
        if drop_bads:
            epochs.drop_bad(reject=dict(eeg=200e-6))
            droplog_fig = epochs.plot_drop_log(show=False)
            droplog_fig.savefig(dir_clean_epochs / f"drop_log_{subj}.png")

        metadata_allsubs.append(epochs.metadata)

        if laplace:
            epochs = mne.preprocessing.compute_current_source_density(epochs)

        # signal vs. noise trials
        epochs_signal = epochs[(epochs.metadata.isyes == 1)]
        epochs_noise = epochs[(epochs.metadata.isyes == 0)]

        # hit vs. miss trials
        epochs_hit = epochs[((epochs.metadata.isyes == 1) & 
                             (epochs.metadata.sayyes == 1))]
        epochs_miss = epochs[((epochs.metadata.isyes == 1) & 
                              (epochs.metadata.sayyes == 0))]

        mne.epochs.equalize_epoch_counts([epochs_hit, epochs_miss])

        # average over 
        evokeds_hit_all.append(epochs_hit.average())
        evokeds_miss_all.append(epochs_miss.average())
        evokeds_signal_all.append(epochs_signal.average())
        evokeds_noise_all.append(epochs_noise.average())

        if save_data_to_disk:

            with Path(path_to.data.eeg.sensor.erp.hits).open("wb") as fp:  # pickling
                pickle.dump(evokeds_hit_all, fp)  

            with Path(path_to.data.eeg.sensor.erp.misses).open("wb") as fp:  # pickling
                pickle.dump(evokeds_miss_all, fp)

        # save trial number and trials removed to csv file
        pd.DataFrame(trials_removed).to_csv(f'{dir_clean_epochs}{Path("/")}trials_removed.csv')
        pd.DataFrame(all_trials).to_csv(f'{dir_clean_epochs}{Path("/")}trials_per_subject.csv')

    return [evokeds_signal_all, evokeds_noise_all, evokeds_hit_all, evokeds_miss_all]


def plot_ssroi(tmin: float = -0.2, tmax: float = 0.5,
               tmin_base: float = -0.1, tmax_base: float = 0):
    """
    Plot topography of P50 for the contrast of signal - noise trials 
    and the contrast over time for the channel with the strongest effect.
    
    ----
    tmin: start time of time window
    tmax: end time of time window
    tmin_base: start time of baseline window
    tmax_base: end time of baseline window

    Returns:
    -------
    None
    """

    conds = create_contrast()

    # baseline correct and crop the data for each participant
    signal = [s.apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax) for s in conds[0]]
    noise = [n.apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax) for n in conds[1]]

    # combine evokeds (subtract noise from signal)
    x = [mne.combine_evoked([s, n], weights=[1, -1]) for s, n in zip(signal, noise)]

    # mean of contrast over participants
    x_gra = mne.grand_average(x)

    # Figure 3: Sensory region of interest definition
    # Create a 1x2 grid of subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    
    # plot topoplot for contrast around P50
    x_gra.plot_topomap(times=[.05], average=[0.02], axes=axs[0], show=False, colorbar=False,
                          show_names=False)

    # plot contrast over time for CP4 (strongest effect)
    x_gra.plot(axes=axs[1], show=False, picks=['CP4'])

    # save image
    plt.tight_layout()
    plt.savefig(Path(path_to.figures.manuscript.figure3, "fig3_ssroi_expecon1.svg"), dpi=300)
    plt.show()


def get_significant_channel(tmin=.04, tmax=0.06, 
                            tmin_base=-0.1, tmax_base=0):

    conds = create_contrast()

    # get data from each participant
    x = np.array(
        [
            ax.copy().apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax).data
            - bx.copy().apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax).data
            for ax, bx in zip(conds[2], conds[3])
        ]
    )

    # significant channels (average over time)
    t, p, h = mne.stats.permutation_t_test(np.squeeze(np.mean(x, axis=2)),
                                            n_permutations=10000, n_jobs=-1)
    # get significant channels
    np.where(p < params.alpha)

    # where is the strongest difference?
    np.argmax(t)

    # change order of dimensions for cluster test
    x = np.transpose(x, [0, 2, 1])  # channels should be last dimension

    # load example epoch to extract channel adjacency matrix
    subj = "007"
    epochs = mne.read_epochs(dir_clean_epochs / f"P{subj}_icacorr_0.1Hz-epo.fif")

    ch_adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, ch_type="eeg")

    # run 2D cluster test ()
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        x, n_permutations=10000, adjacency=ch_adjacency, tail=0, n_jobs=-1
    )

    # get significant clusters
    good_cluster_inds = np.where(cluster_p_values < params.alpha)[0]

    return good_cluster_inds, t_obs, clusters, h0, cluster_p_values

def plot_cluster_output_2D(tmin=0, tmax=0.3, 
                           tmin_base=-0.1, tmax_base=0,
                           good_cluster_inds=None,
                           t_obs=None, 
                           clusters=None):

    # extract data
    signal = [s.apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax) for s in conds[2]]
    noise = [n.apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax) for n in conds[3]]

    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        t_map = t_obs[time_inds, ...].mean(axis=0)

        # get signals at the sensors contributing to the cluster
        sig_times = signal[0].times[time_inds]

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))

        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(t_map[:, np.newaxis], signal[0].info, tmin=0)
        f_evoked.plot_topomap(
            times=0,
            mask=mask,
            axes=ax_topo,
            show=False,
            colorbar=False,
            mask_params=dict(markersize=10),
        )
        image = ax_topo.images[0]

        # remove the title that would otherwise say "0.000 s"
        ax_topo.set_title("")

        # create additional axes (for ERF and colorbar)
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel("Averaged t-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]]))

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes("right", size="300%", pad=1.2)
        title = f"Cluster #{i_clu + 1}, {len(ch_inds)} sensor"
        if len(ch_inds) > 1:
            title += "s (mean)"

        mne.viz.plot_compare_evokeds(
            [signal, noise],
            title=title,
            picks=ch_inds,
            combine="mean",
            axes=ax_signals,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
        )

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1], color="grey", alpha=0.3)

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=0.05)

        # save the figure before showing it
        plt.savefig(
            Path(path_to.figures.manuscript.figure3, f"{clu_idx!s}.svg")
        )
        plt.show()


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
