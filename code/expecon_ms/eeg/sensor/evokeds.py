#!/usr/bin/python3
"""
The script provides functions to analyze cleaned epochs and compute evoked signals from different conditions.

This includes cluster-based permutation tests.

Plotting functions  are also provided.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023-2024
"""

# %% Import
import pickle
import subprocess
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable

from expecon_ms.configs import PROJECT_ROOT, config, params, paths

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/eeg/sensor/evokeds.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# for plotting in a new window (copy to interpreter)
# %matplotlib qt
# for inline plotting
# %matplotlib inline

# set font to Arial and font size to 14
plt.rcParams.update({
    "font.size": params.plot.font.size,
    "font.family": params.plot.font.family,
    "font.sans-serif": params.plot.font.sans_serif,
})

# participant IDs
participants = config.participants

# pilot data counter
pilot_counter = config.participants.pilot_counter

# data_cleaning parameters defined in config.toml
# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def create_contrast(
    study: int, drop_bads: bool = True, laplace: bool = True, subtract_evoked: bool = False,
     save_data_to_disk: bool  = False, save_drop_log: bool = False
):
    """
    Load cleaned epoched data and create contrasts.

    Args:
    ----
    study: int, study number (1 or 2), 1 == expecon1, 2 == expecon2
    drop_bads: boolean, drop bad epochs if True
    laplace: apply CSD to data if boolean is True
    subtract_evoked: boolean, subtract evoked signal from each epoch
    save_data_to_disk: boolean, save data to disk if True
    save_drop_log: boolean, save drop log plots to disk if True

    Returns:
    -------
    list of condition evokeds

    """
    # store trials remaining and removed per participant
    all_trials, trials_removed = [], []
    evokeds_hit_high_all, evokeds_hit_low_all = [], []

    # save  evoked data per participant
    evokeds_signal_all, evokeds_noise_all = [], []
    evokeds_hit_all, evokeds_miss_all = [], []
    evokeds_all = []

    # append epochs per participant
    signal_epochs, noise_epochs, epochs_all = [], [], []

    # metadata after epoch cleaning
    metadata_all_subs = []

    if study == 1:
        # load the cleaned behavioral data for EEG preprocessing
        # includes additional variables, e.g., previous trial history
        id_list = participants.ID_list_expecon1
        df_cleaned = pd.read_csv(Path(paths.data.behavior, "prepro_behav_data_1.csv"))
    else:
        id_list = participants.ID_list_expecon2
        df_cleaned = pd.read_csv(Path(paths.data.behavior, "prepro_behav_data_2.csv"))

    for idx, subj in enumerate(id_list):
        # print participant idx
        print(f"Participant {idx!s}")

        if study == 1:
            dir_clean_epochs = Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon1)
            # load cleaned epochs
            epochs = mne.read_epochs(dir_clean_epochs / f"P{subj}_icacorr_0.1Hz-epo.fif")
        else:
            # skip ID 13
            if subj == "013":
                continue
            dir_clean_epochs = Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon2)
            epochs = mne.read_epochs(dir_clean_epochs / f"P{subj}_icacorr_0.1Hz-epo.fif")
            epochs.metadata = epochs.metadata.rename(
                columns={"resp1_t": "respt1", "stim_type": "isyes", "resp1": "sayyes"}
            )

        # save for descriptives
        before_cleaning = len(epochs.metadata)

        # clean epochs (remove blocks based on hit and false alarm rates, reaction times, etc.)
        epochs = drop_trials(data=epochs)

        # save n_trials per participant
        all_trials.append(len(epochs.metadata))

        # save number of trials removed per participant
        trials_removed.append(before_cleaning - len(epochs.metadata))

        # get behavioral data for current participant
        if study == 1:
            subj_data = df_cleaned[idx + pilot_counter == df_cleaned.ID]
        else:
            subj_data = df_cleaned[idx + 1 == df_cleaned.ID]

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
            epochs = epochs.subtract_evoked()

        # reject epochs that exceed a certain threshold (amplitude)
        if drop_bads:
            epochs.drop_bad(reject=dict(eeg=200e-6))

        # save drop log plots to disk
        if save_drop_log:
            drop_log_fig = epochs.plot_drop_log(show=False)
            drop_log_fig.savefig(dir_clean_epochs / f"drop_log_{subj}.png")

        metadata_all_subs.append(epochs.metadata)

        if laplace:
            epochs = mne.preprocessing.compute_current_source_density(epochs)
        
        # save epochs per participant
        epochs_all.append(epochs)

        # signal vs. noise trials
        epochs_signal = epochs[(epochs.metadata.isyes == 1)]
        epochs_noise = epochs[(epochs.metadata.isyes == 0)]

        # hit vs. miss trials
        epochs_hit = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1))]
        epochs_miss = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 0))]

        epochs_hit_high = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1) 
                            & (epochs.metadata.cue == 0.75) & (epochs.metadata.conf == 1))]
        epochs_hit_low = epochs[((epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1) 
                            & (epochs.metadata.cue == 0.25) & (epochs.metadata.conf == 1))]

        mne.epochs.equalize_epoch_counts([epochs_hit, epochs_miss])
        mne.epochs.equalize_epoch_counts([epochs_hit_high, epochs_hit_low])

        signal_epochs.append(epochs_signal)
        noise_epochs.append(epochs_noise)
   
        # average over
        evokeds_all.append(epochs.average())
        evokeds_hit_all.append(epochs_hit.average())
        evokeds_miss_all.append(epochs_miss.average())
        evokeds_signal_all.append(epochs_signal.average())
        evokeds_noise_all.append(epochs_noise.average())
        evokeds_hit_high_all.append(epochs_hit_high.average())
        evokeds_hit_low_all.append(epochs_hit_low.average())

        if save_data_to_disk:
            with Path(paths.data.eeg.sensor.erp.hit).open("wb") as fp:  # pickling
                pickle.dump(evokeds_hit_all, fp)
            with Path(paths.data.eeg.sensor.erp.miss).open("wb") as fp:
                pickle.dump(evokeds_miss_all, fp)
            with Path(paths.data.eeg.sensor.erp.signal).open("wb") as fp:
                pickle.dump(evokeds_signal_all, fp)
            with Path(paths.data.eeg.sensor.erp.noise).open("wb") as fp:
                pickle.dump(evokeds_noise_all, fp)

        # save trial number and trials removed to csv file
        pd.DataFrame(trials_removed).to_csv(dir_clean_epochs / f"trials_removed_{study!s}.csv")
        pd.DataFrame(all_trials).to_csv(dir_clean_epochs / f"trials_per_subject_{study!s}.csv")

    return [signal_epochs, noise_epochs, evokeds_all, evokeds_signal_all, evokeds_noise_all, evokeds_hit_all, evokeds_miss_all
            , evokeds_hit_high_all, evokeds_hit_low_all, epochs_all]


def plot_roi(study: int, data: np.ndarray, tmin: float, tmax: float, tmin_base: float, tmax_base: float):
    """
    Plot topography of P50 for the contrast of signal - noise trials.

    Args:
    ----
    study: int: study number (1 or 2), 1 == expecon1, 2 == expecon2
    data: list of evoked objects
    tmin: float: start time of time window
    tmax: float: end time of time window
    tmin_base: float: start time of baseline window
    tmax_base: float: end time of baseline window

    Returns:
    -------
    None

    """
    # baseline correct and crop the data for each participant
    signal = [s.copy().apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax) for s in data[1]]
    noise = [n.copy().apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax) for n in data[2]]

    # combine evokeds (subtract noise from signal)
    x = [mne.combine_evoked([s, n], weights=[1, -1]) for s, n in zip(signal, noise)]

    # mean of contrast over participants
    x_gra = mne.grand_average(x)

    # mean of epochs over participants
    e_gra = mne.grand_average(data[0])

    # Figure 3: Sensory region of interest definition.
    # Create a 1x2 grid of subplots
    _, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # plot topo plot for contrast around P50
    x_gra.plot_topomap(times=[0.05], average=[0.02], axes=axs[0], show=False, colorbar=False, show_names=False)

    # plot contrast over time for CP4 (the strongest effect)
    x_gra.plot(axes=axs[1], show=False, picks=["CP4"])

    # save image
    plt.tight_layout()
    plt.savefig(Path(paths.figures.manuscript.figure3, f"fig3_ssroi_{study!s}.svg"), dpi=300)
    plt.show()

    # plot all epochs in CP4
    e_gra.copy().crop(-1,0).plot(picks=['CP4'])
    plt.savefig(Path(paths.figures.manuscript.figure3_suppl_erps, f"prestim_allepochs_{study!s}_CP4.svg"), dpi=300)
    

def get_significant_channel(study: int, data_dict: dict, tmin: float, tmax: float, baseline: bool = False,
                            tmin_base: Union[float, bool] = None, 
                            tmax_base: Union[float, bool] = None):
    """
    Get significant channels for the contrast of signal - noise trials.

    Args:
    ----
    data: list of evoked objects
    tmin: float: start time of time window
    tmax: float: end time of time window
    tmin_base: float: start time of baseline window
    tmax_base: float: end time of the baseline window

    Returns:
    -------
    None

    """
    if baseline:
        # get data from each participant
        x = np.array([
            ax.copy().apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax).data
            - bx.copy().apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax).data
            for ax, bx in zip(list(data_dict.values())[0], list(data_dict.values())[1])
        ])
    else:
        x = np.array([ax.copy().crop(tmin, tmax).data - bx.copy().crop(tmin, tmax).data for ax,
         bx in zip(list(data_dict.values())[0], list(data_dict.values())[1])])

    # significant channels (average over time)
    t, p, _ = mne.stats.permutation_t_test(np.squeeze(np.mean(x, axis=2)), n_permutations=10000, n_jobs=-1)
    # get significant channels
    np.where(p < params.alpha)

    # where is the strongest difference?
    np.argmax(t)

    # change order of dimensions for cluster test
    x = np.transpose(x, [0, 2, 1])  # channels should be last dimension

    # load example epoch to extract channel adjacency matrix
    # doesn't matter which study, same channel layout for both studies
    subj = "007"
    epochs = mne.read_epochs(
        Path(paths.data.eeg.preprocessed.ica.clean_epochs_expecon1, f"P{subj}_icacorr_0.1Hz-epo.fif")
    )

    ch_adjacency, _ = mne.channels.find_ch_adjacency(epochs.info, ch_type="eeg")

    # run 2D cluster test ()
    t_obs, clusters, cluster_p_values, h0 = mne.stats.permutation_cluster_1samp_test(
        x, n_permutations=1000, adjacency=ch_adjacency, tail=0, n_jobs=-1
    )

    # get significant clusters
    good_cluster_inds = np.where(cluster_p_values < params.alpha)[0]

    print(min(cluster_p_values))

    plot_cluster_output_2d(study=study, data_dict=data_dict, tmin=tmin, tmax=tmax,
                           tmin_base=tmin_base, tmax_base=tmax_base,
                           good_cluster_inds=good_cluster_inds, t_obs=t_obs, clusters=clusters)

    return good_cluster_inds, t_obs, clusters, h0, cluster_p_values


def plot_cluster_output_2d(
    study: int,
    data_dict: dict,
    tmin: float,
    tmax: float,
    tmin_base: float,
    tmax_base: float,
    good_cluster_inds: int,
    t_obs: int,
    clusters: int,
):
    """
    Plot cluster output of 2D cluster permutation test.

    Args:
    ----
    study: int: study number (1 or 2), 1 == expecon1, 2 == expecon2
    data: list of evoked objects
    tmin: float: start time of time window
    tmax: float: end time of time window
    tmin_base: float: start time of baseline window
    tmax_base: float: end time of baseline window
    good_cluster_inds: int: indices of significant clusters
    t_obs: int: t-values
    clusters: int: cluster information

    """
    # extract data
    signal = [s.apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax) for s in list(data_dict.values())[0]]
    noise = [n.apply_baseline((tmin_base, tmax_base)).crop(tmin, tmax) for n in list(data_dict.values())[1]]

    # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # convert to numpy array
        ch_inds_np = np.array(ch_inds)
        time_inds_np = np.array(time_inds)

        if study == 1:
            savepath = Path(paths.data.eeg.sensor.erp1)
        else:
            savepath = Path(paths.data.eeg.sensor.erp2)
        # save to disk
        np.save(Path(savepath, f"cluster_{clu_idx!s}_{study}_ch_inds.npy"), ch_inds_np)
        np.save(Path(savepath, f"cluster_{clu_idx!s}_{study}_time_inds.npy"), time_inds_np)

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
            title += "s"

        mne.viz.plot_compare_evokeds(
            data_dict,
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
        fig.subplots_adjust(bottom=0.05)

        # save the figure before showing it
        plt.savefig(Path(paths.figures.manuscript.figure3_suppl_erps, f"{clu_idx!s}_{str(study)}.svg"))
        plt.show()


def extract_single_trial_erp_from_cluster(study: int):
    """
    Extract cluster information from single trial data and 
    save CPP slope and mean with behavioural dataframe.

    Args:
    ----
    study: int: study number (1 or 2), 1 == expecon1, 2 == expecon2

    Returns:
    -------
    None

    """
    from scipy.stats import linregress

    # extract epochs
    out = create_contrast(study=study)

    # extract epochs
    epochs_all = out[-1]

    # load cluster information
    if study == 1:
        savepath = Path(paths.data.eeg.sensor.erp1)
        # load time indices
        time_inds = np.load(Path(savepath, "cluster_93_1_time_inds.npy"))
        # load channel indices
        ch_inds = np.load(Path(savepath, "cluster_93_1_ch_inds.npy"))
    else:
        savepath = Path(paths.data.eeg.sensor.erp2)
        # load time indices
        time_inds = np.load(Path(savepath, "cluster_90_2_time_inds.npy"))
        # load channel indices
        ch_inds = np.load(Path(savepath, "cluster_90_2_ch_inds.npy"))

    # average signal over time indices and channel indices for each participant
    x_all, metadata_all, slopes_sub = [], [], []

    epochs_data = [e.get_data() for e in epochs_all]

    times = epochs_all[0].times

    # for all epochs for all participants average over cluster channels and time indices
    # loop over participants
    for sub_idx, epochs_sub in enumerate(epochs_data):

        slopes_epoch = []

        # calculate mean or slope with regression
        # Average the signal across the selected channels to get the CPP signal
        x_ch = epochs_sub[:, ch_inds, :].mean(axis=(1))

        # average over time indices and channel indices
        x = x_ch[:, time_inds].mean(axis=(1))

        # Select the time window of interest
        cpp_window = x_ch[:,time_inds]
        time_window_selected = times[time_inds]

        # run a regression per epoch
        for epoch_idx,e in enumerate(epochs_sub):
            # Compute the slope using linear regression
            slope, intercept, r_value, p_value, std_err = linregress(time_window_selected, cpp_window[epoch_idx,:])
            # Append the slope to the list of epochs
            slopes_epoch.append(slope)

        # append slopes to list
        slopes_sub.append(slopes_epoch)

        # save values to participant list
        x_all.append(x)

        metadata = epochs_all[sub_idx].metadata

        metadata_all.append(metadata)

    # concatenate all participants
    x_all = np.concatenate(x_all)

    # add to metadata for further analysis
    metadata_all = pd.concat(metadata_all)

    #drop nan rows if they only contain NaNs
    metadata_all = metadata_all.dropna(how='all')
    
    # unlist
    slopes_all = [sub for sublist in slopes_sub for sub in sublist]

    # append single trial CPP slope to the data
    metadata_all['CPP_mean'] = x_all
    metadata_all['CPP_slope'] = slopes_all

    # Group by 'ID' and standardize 'CPP' for each participant
    metadata_all['CPP_mean_zscore'] = metadata_all.groupby('ID')['CPP_mean'].transform(lambda x: (x - x.mean()) / x.std())
    metadata_all['CPP_slope_zscore'] = metadata_all.groupby('ID')['CPP_slope'].transform(lambda x: (x - x.mean()) / x.std())

    # save to disk
    metadata_all.to_csv(Path(savepath, f"metadata_CPP_slope_{study}.csv"))


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
    data = data[data.metadata.respt1 != params.behavioral_cleaning.rt_max]
    data = data[data.metadata.respt1 > params.behavioral_cleaning.rt_min]

    # print rt trials dropped
    rt_trials_removed = before_rt_cleaning - len(data.metadata)

    print("Removed trials based on reaction time: ", rt_trials_removed)
    # Calculate hit rates per participant
    signal = data[data.metadata.isyes == 1]
    hit_rate_per_subject = signal.metadata.groupby(["ID"])["sayyes"].mean()

    print(f"Mean hit rate: {np.mean(hit_rate_per_subject):.2f}")

    # Calculate hit rates by participant and block
    hit_rate_per_block = signal.metadata.groupby(["ID", "block"])["sayyes"].mean()

    # remove blocks with hit rates > 90 % or < 20 %
    filtered_groups = hit_rate_per_block[
        (hit_rate_per_block > params.behavioral_cleaning.hitrate_max)
        | (hit_rate_per_block < params.behavioral_cleaning.hitrate_min)
    ]
    print("Blocks with hit rates > 0.9 or < 0.2: ", len(filtered_groups))

    # Extract the ID and block information from the filtered groups
    remove_hit_rates = filtered_groups.reset_index()

    # Calculate false alarm rates by participant and block
    noise = data[data.metadata.isyes == 0]
    fa_rate_per_block = noise.metadata.groupby(["ID", "block"])["sayyes"].mean()

    # remove blocks with false alarm rates > 0.4
    filtered_groups = fa_rate_per_block[fa_rate_per_block > params.behavioral_cleaning.farate_max]
    print(f"Blocks with false alarm rates > {params.behavioral_cleaning.farate_max}:", len(filtered_groups))

    # Extract the ID and block information from the filtered groups
    remove_fa_rates = filtered_groups.reset_index()

    # Hit-rate should be > the false alarm rate
    filtered_groups = hit_rate_per_block[
        hit_rate_per_block - fa_rate_per_block < params.behavioral_cleaning.hit_fa_diff
    ]
    print("Blocks with hit rates < false alarm rates:", len(filtered_groups))

    # Extract the ID and block information from the filtered groups
    hit_vs_fa_rate = filtered_groups.reset_index()

    # Concatenate the dataframes
    combined_df = pd.concat([remove_hit_rates, remove_fa_rates, hit_vs_fa_rate])

    # Remove duplicate rows based on 'ID' and 'block' columns
    unique_df = combined_df.drop_duplicates(subset=["ID", "block"])

    # Merge the big dataframe with unique_df to retain only the non-matching rows
    data.metadata = data.metadata.merge(unique_df, on=["ID", "block"], how="left", indicator=True, suffixes=("", "_y"))

    data = data[data.metadata["_merge"] == "left_only"]

    # Remove the '_merge' column
    data.metadata = data.metadata.drop("_merge", axis=1)

    return data

# %% run stuff

# Load cleaned epochs and create contrasts
#output = create_contrast(study=2)
# create dictionary with data
#hit_low = output[-2]
#hit_high = output[-3]
#data_dict = {'high': hit_high, 'low': hit_low}
# run cluster test and plot sign. cluster (no baseline correction)
#output_cluster = get_significant_channel(study=2, data_dict=data_dict, tmin=-1, tmax=1)

# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
