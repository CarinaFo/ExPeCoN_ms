#!/usr/bin/python3
"""
The script provides functions to analyze cleaned epochs and compute evoked signals from different conditions.

This includes permutation cluster tests.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023
"""
# %% Import
import os
import pickle
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.api as sm

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

# set paths (Path works both on Windows and Linux)
dir_clean_epochs = Path(path_to.data.eeg.preprocessed.ica.clean_epochs)
behav_path = Path(path_to.data.behavior.behavior_df)

# participant IDs
id_list = config.participants.ID_list


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def create_contrast(drop_bads: bool = False, laplace: bool = False, subtract_evoked: bool = False):
    """
    Load cleaned epoched data and create contrasts.

    Args:
    ----
    laplace: apply CSD to data if boolean is True
    subtract_evoked: boolean, subtract evoked signal from each epoch

    Returns:
    -------
    list of condition evokeds
    """
    all_trials, trials_removed = [], []

    # TODO(simon): all not used?!
    evokeds_high_hits_all, evokeds_low_hits_all, evokeds_high_all, evokeds_low_all, evokeds_prevyes_all = (
        [],
        [],
        [],
        [],
        [],
    )
    evokeds_prevno_all, evokeds_signal_all, evokeds_noise_all, evokeds_hit_all, evokeds_miss_all = [], [], [], [], []

    # metadata after epoch cleaning
    metadata_allsubs = []

    for idx, subj in enumerate(id_list):
        # print participant idx
        print(f"Participant {idx!s}")

        # load cleaned epochs
        epochs = mne.read_epochs(dir_clean_epochs / f"P{subj}_epochs_after_ica_0.1Hzfilter-epo.fif")

        ids_to_delete = [10, 12, 13, 18, 26, 30, 32, 32, 39, 40, 40, 30]  # TODO(simon): move to config.toml
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
        before_rt_removal = len(epochs.metadata)
        epochs = epochs[epochs.metadata.respt1 >= 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        # save n_trials per participant
        all_trials.append(len(epochs.metadata))

        trials_removed.append(before_rt_removal - len(epochs.metadata))

        # load behavioral data
        data = pd.read_csv(f'{behav_path}{Path("/")}prepro_behav_data.csv')

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

        if subtract_evoked:
            epochs.subtract_evoked()

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
        epochs_hit = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1))]
        epochs_miss = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 0))]

        mne.epochs.equalize_epoch_counts([epochs_hit, epochs_miss])

        evokeds_hit_all.append(epochs_hit.average())
        evokeds_miss_all.append(epochs_miss.average())
        evokeds_signal_all.append(epochs_signal.average())
        evokeds_noise_all.append(epochs_noise.average())

        droppath = Path("D:/expecon_ms/data/eeg/prepro_ica/droplog")

        # save trial number and trials removed to csv file
        pd.DataFrame(trials_removed).to_csv(f'{droppath}{Path("/")}trials_removed.csv')
        pd.DataFrame(all_trials).to_csv(f'{droppath}{Path("/")}trials_per_subject.csv')

    return [evokeds_signal_all, evokeds_noise_all, evokeds_hit_all, evokeds_miss_all]  # conds


def cluster_perm_space_time_plot(tmin: float = -0.1, tmax: float = 0.5, channel=["CP4"]):
    """
    Plot cluster permutation results in space and time.

    This function prepares the data for statistical tests in
    1D (permutation over timepoints) or
    2D (permutation over timepoints and channels).

    Significant clusters are plotted.

    Cluster output is correlated with the behavioral outcome.

    Args:
    ----
    tmin: start time of time window
    tmax: end time of time window
    channel: channel to plot

    Returns:
    -------
    None
    """
    conds = create_contrast()

    # TODO(simon): why do we write at the beginning of the function?
    with Path(path_to.data.eeg.sensor.erp.hits).open("wb") as fp:  # pickling
        pickle.dump(conds[8], fp)  # TODO(simon): output of create_contrast() has len==4 ?!

    with Path(path_to.data.eeg.sensor.erp.misses).open("wb") as fp:  # pickling
        pickle.dump(conds[9], fp)

    # get grand average over all subjects for plotting the results later
    a = [ax.copy().crop(tmin, tmax).filter(15, 25) for ax in conds[2]]  # TODO(simon): not used?!
    b = [bx.copy().crop(tmin, tmax).filter(15, 25) for bx in conds[3]]  # TODO(simon): not used?!

    if tmax <= 0:
        a = [
            ax.copy().pick_channels(channel).crop(tmin, tmax) for ax in evokeds_a_all
        ]  # TODO(simon): not defined & used!
        b = [
            bx.copy().pick_channels(channel).crop(tmin, tmax) for bx in evokeds_b_all
        ]  # TODO(simon): not defined & used!
    else:
        hit = [ax.copy().apply_baseline((-0.1, 0)).crop(tmin, tmax) for ax in conds[8]]
        miss = [bx.copy().apply_baseline((-0.1, 0)).crop(tmin, tmax) for bx in conds[9]]

    a_gra = mne.grand_average(hit)
    b_gra = mne.grand_average(miss)

    # colors from colorbrewer2.org
    colors_prevchoice = ["#e66101", "#5e3c99"]  # brown #d8b365 and green #5ab4ac  # TODO(simon): not used?!
    colors_highlow = ["#ca0020", "#0571b0"]  # red and blue
    colors_hitmiss = ["#d01c8b", "#018571"]  # pink, darkgreen
    linestyles = ["--", "--"]  # TODO(simon): not used?!

    # Create a 3x1 grid of subplots
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))

    # plot topo for hits only and the first 100 ms after stimulation onset
    a_gra.copy().crop(0, 0.1).plot_topo(axes=axs[0], show=False)

    # plot single channel signal for hits
    a_gra.copy().crop(0, 0.1).plot(picks=channel, axes=axs[1])

    # plot contrast between hits and misses for C4
    mne.viz.plot_compare_evokeds(
        evokeds={"hit": hit, "miss": miss},
        picks=channel,
        show_sensors=False,
        colors=colors_hitmiss,
        axes=axs[2],
        show=False,
        legend="lower right",
    )
    # save image
    plt.tight_layout()
    plt.savefig(Path(path_to.figures.manuscript.figure5, "fig5_expecon1.svg"), dpi=300)
    plt.show()

    diff = mne.combine_evoked([a_gra, b_gra], weights=[1, -1])
    topo = diff.plot_topo()
    figpath = Path("./figs/manuscript_figures/Figure3/topo.svg")  # TODO(simon): move to configs.toml
    # TODO(simon): note that path_to.figures.manuscript.figure3 is not the same as here.
    #  It is used in figure1.py ?!

    topo.savefig(figpath)

    x = np.array(
        [
            ax.copy().apply_baseline((-0.5, -0.4)).crop(tmin, tmax).data
            - bx.copy().apply_baseline((-0.5, -0.4)).crop(tmin, tmax).data
            for ax, bx in zip(conds[2], conds[3])
        ]
    )

    t, p, h = mne.stats.permutation_t_test(np.squeeze(x))

    times = conds[2][0].copy().crop(-0.5, 0).times

    sig_times = np.where(p < params.alpha)

    print(times[sig_times])

    x = np.transpose(x, [0, 2, 1])  # channels should be last dimension

    # load example epoch to extract channel adjacency matrix
    subj = "007"
    epochs = mne.read_epochs(dir_clean_epochs / f"P{subj}_epochs_after_ica-epo.fif")

    ch_adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, ch_type="eeg")

    # threshold-free cluster enhancement
    threshold_tfce = dict(start=0, step=0.1)  # TODO(simon): not used?!

    # 2D cluster test
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        x[:, :, :], n_permutations=10000, adjacency=ch_adjacency, tail=0, n_jobs=-1
    )

    good_cluster_inds = np.where(cluster_p_values < params.alpha)[0]
    print(good_cluster_inds)  # times when something significant happened

    min(cluster_p_values)
    print(cluster_p_values)

    cluster_channel = np.unique(clusters[good_cluster_inds[0]][1])
    cluster1_channel = np.unique(clusters[good_cluster_inds[1]][1])

    ch_names_cluster0 = [evokeds_a_all[0].ch_names[c] for c in cluster_channel]  # TODO(simon): not defined & unused?!
    ch_names_cluster1 = [evokeds_a_all[0].ch_names[c] for c in cluster1_channel]  # TODO(simon): not defined & unused?!

    # cluster previous yes
    cluster_channel = np.unique(clusters[good_cluster_inds[0]][1])
    ch_names_cluster_prevyes = [
        evokeds_a_all[0].ch_names[c] for c in cluster_channel
    ]  # TODO(simon): not defined & unused?!

    # 1D cluster test
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        np.squeeze(x[:, :, :]), n_permutations=10000, tail=0, n_jobs=-1
    )

    good_cluster_inds = np.where(cluster_p_values < params.alpha)[0]  # times when something significant happened
    print(good_cluster_inds)

    min(cluster_p_values)
    print(cluster_p_values)

    for g in good_cluster_inds:
        print(high[0].times[clusters[g]])  # TODO(simon): not defined?!
        print(cluster_p_values[g])

    # load signal detection dataframe
    out = figure1.prepare_for_plotting()

    sdt = out[0][0]

    # TODO(simon): consider to move cue params into config.toml
    crit_diff = np.array(sdt.criterion[sdt.cue == 0.75]) - np.array(
        sdt.criterion[sdt.cue == 0.25]
    )  # TODO(simon): unused?!
    d_diff = np.array(sdt.dprime[sdt.cue == 0.75]) - np.array(sdt.dprime[sdt.cue == 0.25])  # TODO(simon): unused?!

    # load brms beta weights
    reg_path = Path(path_to.data.behavior.behavior_df, "brms_betaweights.csv")
    df_beta_weights = pd.read_csv(reg_path)

    # 1D test: average over timepoints
    x_time_avg = np.mean(x[:, :, 91:], axis=(1, 2))

    # average over significant timepoints
    x_time = np.mean(x[:, np.unique(clusters[4][0]), :], axis=1)

    # average over significant channels
    x_time_channel = np.mean(x_time[:, np.unique(clusters[4][1])], axis=1)  # TODO(simon): unused?!

    # correlate with cluster
    scipy.stats.pearsonr(df_beta_weights.cue_prev, x)

    x = x_time_avg
    x = x * 10**5
    y = df_beta_weights.prev_choice

    sns.regplot(x, y, fit_reg=True)

    # fit the linear regression model using statsmodels
    model = sm.OLS(y, sm.add_constant(x))
    results = model.fit()

    # extract the regression weights
    intercept = results.params[0]
    slope = results.params[1]

    # plot the regression line
    plt.plot(x, intercept + slope * x, color="blue")
    reg_savepath = Path("./figs/manuscript_figures/Figure3/regplot.svg")  # TODO(simon): see other comment
    plt.savefig(reg_savepath)
    # show the plot
    plt.show()

    # now plot the significant cluster(s)

    # configure variables for visualization
    colors = {cond_a: colors_highlow[0], cond_b: colors_highlow[1]}  # TODO(simon): not defined?!

    a = [a.copy().apply_baseline((-0.5, -0.4)).crop(tmin, tmax) for a in conds[2]]
    b = [a.copy().apply_baseline((-0.5, -0.4)).crop(tmin, tmax) for a in conds[3]]

    a_gra = mne.grand_average(a)
    b_gra = mne.grand_average(b)  # TODO(simon): not used?!

    # organize data for plotting
    # instead of grand average we could use the evoked data per subject so that we can plot CIs
    grand_average = {cond_a: a, cond_b: b}  # TODO(simon): not defined?!

    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        t_map = t_obs[time_inds, ...].mean(axis=0)

        # get signals at the sensors contributing to the cluster
        sig_times = a_gra.times[time_inds]

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(nrows=1, ncols=1, figsize=(10, 3))

        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(t_map[:, np.newaxis], a[0].info, tmin=0)
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
        divider = make_axes_locatable(ax_topo)  # TODO(simon): not defined?!

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
            grand_average,
            title=title,
            picks=ch_inds,
            combine="mean",
            axes=ax_signals,
            colors=colors,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
        )

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx((ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3)

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=0.05)

        # save the figure before showing it
        plt.savefig(
            Path(path_to.figures.manuscript.figure5, f"{cond}_{clu_idx!s}_prestim_.svg")
        )  # TODO(simon): not defined?! & '_' at the end of filename?!
        # TODO(simon): cond not defined
        plt.show()


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
