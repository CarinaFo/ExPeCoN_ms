# this script provides functions to anaylze cleaned epochs and compute evoked signals from different conditions
# including permutation cluster tests

# author: Carina Forster
# email: forster@cbs.mgp.de

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import subprocess

# Specify the file path for which you want the last commit date
file_path = "D:\expecon_ms\\analysis_code\\eeg\\sensor\\evokeds.py"

last_commit_date = subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", file_path]).decode("utf-8").strip()
print("Last Commit Date for", file_path, ":", last_commit_date)

# own modules
modulepath = Path('D:/expecon_ms/analysis_code')
# add path to sys.path.append() if package isn't found
sys.path.append(modulepath)

os.chdir('D:/expecon_ms/analysis_code')
from behav import figure1

# for plotting in new window (copy to interpreter)
# %matplotlib qt

# for inline plotting
#%matplotlib inline

# set font to Arial and font size to 14
plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# set paths (Path works both on Windows and Linux)
dir_cleanepochs = Path('D:/expecon_ms/data/eeg/prepro_ica/clean_epochs_corr')
behavpath = Path('D:/expecon_ms/data/behav/behav_df')

# save path for figure
save_dir_cluster_output = Path('D:/expecon_ms/figs/manuscript_figures/figure5_hitmiss_roi')

# participant index list
IDlist = ['007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021','022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046',
          '047', '048', '049']


def create_contrast(drop_bads=False,
                    laplace=False,
                    subtract_evoked=False):

    """ this function loads cleaned epoched data and creates evoked contrasts for different conditions
    input: 
    drop_bads: boolean, drop bad epochs if True
    laplace: apply CSD to data if boolean is True
    subtract_evoked: boolean, subtract evoked signal from each epoch
    output:
    list of condition evokeds
    """

    all_trials, trials_removed = [], []

    evokeds_signal_all, evokeds_noise_all, evokeds_hit_all, evokeds_miss_all = [], [], [], []

    # metadata after epoch cleaning
    metadata_allsubs = []

    for idx, subj in enumerate(IDlist):

        # print participant idx
        print(f'Participant {str(idx)}')

        # load cleaned epochs
        epochs = mne.read_epochs(f'{dir_cleanepochs}{Path("/")}P{subj}_epochs_after_ica_0.1Hzfilter-epo.fif')
        
        ids_to_delete = [10, 12, 13, 18, 26, 30, 32, 32, 39, 40, 40, 30]
        blocks_to_delete = [6, 6, 4, 3, 4, 3, 2, 3, 3, 2, 5, 6]

        # Check if the participant ID is in the list of IDs to delete
        if pd.unique(epochs.metadata.ID) in ids_to_delete:

            # Get the corresponding blocks to delete for the current participant
            participant_blocks_to_delete = [block for id_, block in
                                            zip(ids_to_delete, blocks_to_delete)
                                            if id_ == pd.unique(epochs.metadata.ID)]
            
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
        data = pd.read_csv(f'{behavpath}{Path("/")}prepro_behav_data.csv')

        subj_data = data[data.ID == idx+7]

        # get drop log from epochs
        drop_log = epochs.drop_log

        search_string = 'IGNORED'

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
                droplog_fig.savefig(f'{dir_cleanepochs}{Path("/")}drop_log_{subj}.png')

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

        droppath = Path('D:/expecon_ms/data/eeg/prepro_ica/droplog')

        # save trial number and trials removed to csv file
        pd.DataFrame(trials_removed).to_csv(f'{droppath}{Path("/")}trials_removed.csv')
        pd.DataFrame(all_trials).to_csv(f'{droppath}{Path("/")}trials_per_subject.csv')

    conds = [evokeds_signal_all, evokeds_noise_all,
             evokeds_hit_all, evokeds_miss_all]

    return conds


def plot_evoked_contrast(tmin=-0.1, tmax=0.5, baseline_tmin=-0.1,
                                 baseline_tmax=0, channel=['CP4']):

    """Plot evoked  contrast for two conditions.
    input:
    tmin: start time of time window
    tmax: end time of time window
    channel: channel to plot
    output:
    None
    """

    conds = create_contrast()

    # crop and baseline correct the data 
    hit = [ax.copy().apply_baseline((baseline_tmin, baseline_tmax))
                    .crop(tmin, tmax) for ax in conds[2]]
    miss = [bx.copy().apply_baseline((-0.1, 0))
                .crop(tmin, tmax) for bx in conds[3]]

    # get grand average over all subjects for plotting the results later
    a_gra = mne.grand_average(hit)
    b_gra = mne.grand_average(miss)

    # colors from colorbrewer2.org
    colors_prevchoice = ['#e66101', '#5e3c99'] # brown #d8b365 and green #5ab4ac
    colors_highlow = ["#ca0020", '#0571b0'] # red and blue
    colors_hitmiss = ['#d01c8b', '#018571'] # pink, darkgreen

    # Create a 3x1 grid of subplots
    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(6, 12))

    # plot topo for hits only and the first 100 ms after stimulation onset
    a_gra.copy().crop(0, 0.1).plot_topo(axes=axs[0], show=False)

    # plot single channel signal for hits
    a_gra.copy().crop(0, 0.1).plot(picks=channel, axes=axs[1])

    # plot contrast between hits and misses for C4
    mne.viz.plot_compare_evokeds({'hit': hit, 'miss': miss}, picks=channel, show_sensors=False,
                                       colors = colors_hitmiss, axes=axs[2], show=False,
                                       legend='lower right')
    # save image
    plt.tight_layout()
    #figpath = Path('D:/expecon_ms/figs/manuscript_figures/figure5_hitmiss_roi/fig5_expecon1.svg')
    #plt.savefig(figpath, dpi=300)

    plt.show()
    
    diff = mne.combine_evoked([a_gra,b_gra], weights=[1,-1])
    topo = diff.plot_topo(title='hit-miss')
    figpath = Path('D:/expecon_ms/figs/manuscript_figures/figure5_hitmiss_roi/topo_hitmiss_diff.svg')
    topo.savefig(figpath)

    return conds

def run_cluster_perm_test():

    """Run cluster permutation test for two conditions.
    Either over 2 dimensions (channels and time) or over time only.)"""

    # create data array for permutation test over time only
    X = np.array([ax.copy().apply_baseline((baseline_tmin, baseline_tmax)).crop(tmin, tmax).pick_channels(['CP4'])
                  .data-bx.copy().apply_baseline((baseline_tmin, baseline_tmax))
                  .crop(tmin, tmax).pick_channels(['CP4']).data for ax, bx in zip(conds[2], conds[3])])

    X.shape

    t,p,h = mne.stats.permutation_t_test(np.squeeze(X))

        
    # 1D cluster test
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(np.squeeze(X), 
                                                                    n_permutations=10000,
                                                                    tail=0, 
                                                                    n_jobs=-1)

    times = conds[2][0].copy().crop(tmin, tmax).times

    sig_times = np.where(p < 0.05)

    print(times[sig_times])

    # create data array for cluster permutation test over time and channels
    X = np.array([ax.copy().apply_baseline((baseline_tmin, baseline_tmax)).crop(tmin, tmax)
                  .data-bx.copy().apply_baseline((baseline_tmin, baseline_tmax))
                  .crop(tmin, tmax).data for ax, bx in zip(conds[2], conds[3])])

    X.shape

    X = np.transpose(X, [0, 2, 1]) # channels should be last dimension

    X.shape

    # load example epoch to extract channel adjacency matrix
    subj='007'
    epochs = mne.read_epochs(f'{dir_cleanepochs}{Path("/")}P{subj}_epochs_after_ica-epo.fif')

    ch_adjacency,_ = mne.channels.find_ch_adjacency(epochs.info, ch_type='eeg')

    # threshold free cluster enhancement
    threshold_tfce = dict(start=0, step=0.1)

    # 2D cluster test over time and channels
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X, n_permutations=10000,
                                                                                    adjacency=ch_adjacency, 
                                                                                    tail=0, 
                                                                                    n_jobs=-1)

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print(good_cluster_inds)

    return T_obs, clusters, cluster_p_values, H0, good_cluster_inds
    

def plot_cluster_perm_ouput():

    """Plot cluster permutation test output."""

    T_obs, clusters, cluster_p_values, H0, good_cluster_inds = run_cluster_perm_test()

    colors_hitmiss = ['#d01c8b', '#018571'] # pink, darkgreen
    # configure variables for visualization
    colors = {"hit": colors_hitmiss[0], "miss": colors_hitmiss[1]}

    a = [a.copy().apply_baseline((baseline_tmin, baseline_tmax)).crop(tmin, tmax) for a in conds[2]]
    b = [a.copy().apply_baseline((baseline_tmin, baseline_tmax)).crop(tmin, tmax) for a in conds[3]]

    # organize data for plotting
    # instead of grand average we could use the evoked data per subject so that we can plot CIs
    grand_average = {"hit": a, "miss": b}

        # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        t_map = T_obs[time_inds, ...].mean(axis=0)

        # get signals at the sensors contributing to the cluster
        sig_times = a[0].times[time_inds]

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

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
        divider = make_axes_locatable(ax_topo)

        # add axes for colorbar
        ax_colorbar = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(image, cax=ax_colorbar)
        ax_topo.set_xlabel(
            "Averaged t-map ({:0.3f} - {:0.3f} s)".format(*sig_times[[0, -1]])
        )

        # add new axis for time courses and plot time courses
        ax_signals = divider.append_axes("right", size="300%", pad=1.2)
        title = "Cluster #{0}, {1} sensor".format(i_clu + 1, len(ch_inds))
        if len(ch_inds) > 1:
            title += "s (mean)"

        mne.viz.plot_compare_evokeds(
            grand_average,
            title=title,
            picks=ch_inds, 
            combine='mean',
            axes=ax_signals,
            colors=colors,
            show=False,
            split_legend=True,
            truncate_yaxis="auto",
        )

        # plot temporal cluster extent
        ymin, ymax = ax_signals.get_ylim()
        ax_signals.fill_betweenx(
            (ymin, ymax), sig_times[0], sig_times[-1], color="orange", alpha=0.3
        )

        # clean up viz
        mne.viz.tight_layout(fig=fig)
        fig.subplots_adjust(bottom=0.05)

        # save figure before showing the figure
        plt.savefig(f'{save_dir_cluster_output}{Path("/")}{"hitmiss"}_{str(clu_idx)}.svg')
        plt.show()