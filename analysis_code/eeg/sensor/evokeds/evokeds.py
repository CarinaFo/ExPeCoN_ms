########################################evoked analysis######################################


# Author: Carina Forster
# Date: 2023-07-17

# This script plots evoked responses for two conditions and runs permutation test statistics


import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.api as sm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

modulepath = Path('D:/expecon_ms/analysis_code')
# add path to sys.path.append() if package isn't found
sys.path.append(modulepath)

from behav import figure1

# set font to Arial and font size to 14
plt.rcParams.update({'font.size': 14, 'font.family': 'sans-serif', 'font.sans-serif': 'Arial'})

# set paths (Path works both on Windows and Linux)
dir_cleanepochs = Path('D:/expecon_ms/data/eeg/prepro_ica/clean_epochs_corr')
behavpath = Path('D:/expecon_ms/data/behav/behav_df')

# save cluster figures as svg and png files
save_dir_cluster_output = Path('D:/expecon_ms/figs/manuscript_figures/Figure3')

# participant index list

IDlist = ['007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021','022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046',
          '047', '048', '049']


def create_contrast(drop_bads=True,
                    laplace=True,
                    subtract_evoked=False):

    """ this function loads cleaned epoched data and creates contrasts
    input: 
    laplace: apply CSD to data if boolean is True
    subtract_evoked: boolean, subtract evoked signal from each epoch
    output:
    list of condition evokeds
       """

    all_trials, trials_removed = [], []

    evokeds_high_hits_all, evokeds_low_hits_all, evokeds_high_prevyes_all, evokeds_low_prevyes_all = [], [], [], []
    evokeds_high_prevno_all, evokeds_low_prevno_all, evokeds_high_all, evokeds_low_all, evokeds_prevyes_all = [], [], [], [], []
    evokeds_prevno_all, evokeds_signal_all, evokeds_noise_all, evokeds_hit_all, evokeds_miss_all = [], [], [], [], []

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

        epochs_high_hits = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1))]
        epochs_low_hits = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1))]

        epochs_high_prevyes = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.prevsayyes == 1))]
        epochs_low_prevyes = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.prevsayyes == 1))]

        epochs_high_prevno = epochs[((epochs.metadata.cue == 0.75) & (epochs.metadata.prevsayyes == 0))]
        epochs_low_prevno = epochs[((epochs.metadata.cue == 0.25) & (epochs.metadata.prevsayyes == 0))]

        epochs_high = epochs[(epochs.metadata.cue == 0.75)]
        epochs_low = epochs[(epochs.metadata.cue == 0.25)]

        epochs_prevyes = epochs[epochs.metadata.prevsayyes == 1]
        epochs_prevno = epochs[epochs.metadata.prevsayyes == 0]

        epochs_signal = epochs[(epochs.metadata.isyes == 1)]
        epochs_noise = epochs[(epochs.metadata.isyes == 0)]

        epochs_hit = epochs[(epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 1)]
        epochs_miss = epochs[(epochs.metadata.isyes == 1) & (epochs.metadata.sayyes == 0)]

        mne.epochs.equalize_epoch_counts([epochs_high_hits, epochs_low_hits])
        mne.epochs.equalize_epoch_counts([epochs_high_prevyes, epochs_low_prevyes])
        mne.epochs.equalize_epoch_counts([epochs_high_prevno, epochs_low_prevno])
        mne.epochs.equalize_epoch_counts([epochs_prevyes, epochs_prevno])
        mne.epochs.equalize_epoch_counts([epochs_hit, epochs_miss])

        evokeds_hit_all.append(epochs_hit.average())
        evokeds_miss_all.append(epochs_miss.average())
        evokeds_high_hits_all.append(epochs_high_hits.average())
        evokeds_low_hits_all.append(epochs_low_hits.average())
        evokeds_high_prevyes_all.append(epochs_high_prevyes.average())
        evokeds_low_prevyes_all.append(epochs_low_prevyes.average())
        evokeds_high_prevno_all.append(epochs_high_prevno.average())
        evokeds_low_prevno_all.append(epochs_low_prevno.average())
        evokeds_high_all.append(epochs_high.average())
        evokeds_low_all.append(epochs_low.average())
        evokeds_prevyes_all.append(epochs_prevyes.average())
        evokeds_prevno_all.append(epochs_prevno.average())
        evokeds_signal_all.append(epochs_signal.average())
        evokeds_noise_all.append(epochs_noise.average())

        droppath = Path('D:/expecon_ms/data/eeg/prepro_ica/droplog')

        # save trial number and trials removed to csv file
        pd.DataFrame(trials_removed).to_csv(f'{droppath}{Path("/")}trials_removed.csv')
        pd.DataFrame(all_trials).to_csv(f'{droppath}{Path("/")}trials_per_subject.csv')

    conds = [evokeds_high_hits_all, evokeds_low_hits_all,
             evokeds_high_prevyes_all, evokeds_low_prevyes_all,
             evokeds_high_prevno_all, evokeds_low_prevno_all,
             evokeds_high_all, evokeds_low_all, evokeds_prevyes_all,
             evokeds_prevno_all, evokeds_signal_all,
             evokeds_noise_all, evokeds_hit_all,
             evokeds_miss_all]

    return conds

def cluster_perm_space_time_plot(tmin=0, tmax=1, channel=['C4']):
    """Plot cluster permutation results in space and time. This function prepares the
    data for stats tests in 1D (permutation over timepoints) or 2D (permutation over
    timepoints and channels). 
    Significant cluster are plotted
    Cluster output is correlated with behavioral outcome.
    input:
    tmin: start time of time window
    tmax: end time of time window
    channel: channel to plot
    output:
    None
    """

    conds = create_contrast()

    # get grand average over all subjects for plotting the results later
    
    a = [ax.copy().crop(tmin, tmax) for ax in evokeds_a_all]
    b = [bx.copy().crop(tmin, tmax) for bx in evokeds_b_all]

    if tmax <= 0:
        a = [ax.copy().pick_channels(channel).crop(tmin, tmax) for ax in evokeds_a_all]
        b = [bx.copy().pick_channels(channel).crop(tmin, tmax) for bx in evokeds_b_all]
    else:
        high_no = [ax.copy().pick_channels(channel).apply_baseline((-0.5,-0.4)).crop(tmin, tmax) for ax in conds[4]]
        low_no = [bx.copy().pick_channels(channel).apply_baseline((-0.5,-0.4)).crop(tmin, tmax) for bx in conds[5]]

    a_gra = mne.grand_average(hit)
    b_gra = mne.grand_average(low)

    # colors from colorbrewer2.org
    colors_prevchoice = ['#e66101', '#5e3c99'] # brown #d8b365 and green #5ab4ac
    colors_highlow = ["#ca0020", '#0571b0'] # red and blue
    colors_hitmiss = ['#d01c8b', '#018571'] # pink, darkgreen
    linestyles=['--', '--']

    # Create a 1x2 grid of subplots
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    mne.viz.plot_compare_evokeds({'0.75': high_yes, '0.25': low_yes}, combine='mean', picks=channel, show_sensors=False,
                                       colors = colors_highlow, axes=axs[0], show=False, legend='lower right',
                                       truncate_xaxis=False, truncate_yaxis=False, title='previous yes trials')
    mne.viz.plot_compare_evokeds({'0.75': high_no, '0.25': low_no}, combine='mean', picks=channel, show_sensors=False,
                                       colors = colors_highlow, axes=axs[1], show=False, legend='lower right',
                                       truncate_xaxis=False, truncate_yaxis=False, linestyles=linestyles, title='previous no trials')
    
    # Adjust the layout to prevent overlapping
    plt.tight_layout()
    figpath = Path('D:/expecon_ms/figs/manuscript_figures/Figure3/fig3_prevchoice_cue.svg')
    plt.savefig(figpath)
    plt.show()
    
    diff = mne.combine_evoked([a_gra,b_gra], weights=[1,-1])
    topo = diff.plot_topo()
    figpath = Path('D:/expecon_ms/figs/manuscript_figures/Figure3/topo.svg')
    topo.savefig(figpath)
   
    X = np.array([ax.data-bx.data for ax,bx in zip(high_no, low_no)])

    X = np.transpose(X, [0, 2, 1]) # channels should be last dimension

    # load example epoch to extract channel adjacency matrix
    subj='007'
    epochs = mne.read_epochs(f'{dir_cleanepochs}{Path("/")}P{subj}_epochs_after_ica-epo.fif')

    ch_adjacency,_ = mne.channels.find_ch_adjacency(epochs.info, ch_type='eeg')

    # threshold free cluster enhancement
    threshold_tfce = dict(start=0, step=0.01)

    # 2D cluster test
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(X[:,:,:], n_permutations=10000,
                                                                                    adjacency=ch_adjacency, 
                                                                                    tail=0, 
                                                                                    n_jobs=-1)

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
    print(good_cluster_inds) # times where something significant happened
    
    min(cluster_p_values)
    print(cluster_p_values) 

    cluster_channel = np.unique(clusters[good_cluster_inds[0]][1])
    cluster1_channel = np.unique(clusters[good_cluster_inds[1]][1])

    ch_names_cluster0 = [evokeds_a_all[0].ch_names[c] for c in cluster_channel]
    ch_names_cluster1 = [evokeds_a_all[0].ch_names[c] for c in cluster1_channel]

    # cluster previous yes

    cluster_channel = np.unique(clusters[good_cluster_inds[0]][1])
    ch_names_cluster_prevyes = [evokeds_a_all[0].ch_names[c] for c in cluster_channel]
    
    # 1D cluster test
    T_obs, clusters, cluster_p_values, H0 = mne.stats.permutation_cluster_1samp_test(np.squeeze(X[:,:,:]), n_permutations=10000,
                                                                                    tail=0, n_jobs=-1
                                                                                    )

    good_cluster_inds = np.where(cluster_p_values < 0.05)[0] # times where something significant happened
    print(good_cluster_inds)

    min(cluster_p_values)
    print(cluster_p_values)

    for g in good_cluster_inds:
        print(a_gra.times[clusters[g]])
        print(cluster_p_values[g])

    # load signal detection dataframe
    out = figure1.prepare_for_plotting()
        
    sdt = out[0][0]

    crit_diff = np.array(sdt.criterion[sdt.cue == 0.75]) - np.array(sdt.criterion[sdt.cue == 0.25])

    d_diff = np.array(sdt.dprime[sdt.cue == 0.75]) - np.array(sdt.dprime[sdt.cue == 0.25])

    # load brms beta weights
    reg_path = Path('D:/expecon_ms/data/behav/behav_df/brms_betaweights.csv')
    df = pd.read_csv(reg_path)

    # 1D test: average over timepoints
    X_timeavg = np.mean(X[:,:,91:], axis=(1,2))

    # average over significant timepoints
    X_time = np.mean(X[:,np.unique(clusters[4][0]),:], axis=1)

    # average over significant channels
    X_time_channel = np.mean(X_time[:,np.unique(clusters[4][1])], axis=1)

    # correlate with cluster
    scipy.stats.pearsonr(df.cue_prev, x)

    x = X_timeavg
    x = x*10**5
    y = df.prev_choice

    sns.regplot(x, y, fit_reg=True)

    # Fit the linear regression model using statsmodels
    model = sm.OLS(y, sm.add_constant(x))
    results = model.fit()

    # Extract the regression weights
    intercept = results.params[0]
    slope = results.params[1]

    # Plot the regression line
    plt.plot(x, intercept + slope * x, color='blue')
    reg_savepath = Path('D:/expecon_ms/figs/manuscript_figures/Figure3/regplot.svg')
    plt.savefig(reg_savepath)
    # Show the plot
    plt.show()

    # now plot the significant cluster(s)

    # configure variables for visualization
    colors = {cond_a: colors_highlow[0], cond_b: colors_highlow[1]}

    # organize data for plotting
    # instead of grand average we could use the evoked data per subject so that we can plot CIs
    grand_average = {cond_a: a, cond_b: b}

        # loop over clusters
    for i_clu, clu_idx in enumerate(good_cluster_inds):
        # unpack cluster information, get unique indices
        time_inds, space_inds = np.squeeze(clusters[clu_idx])
        ch_inds = np.unique(space_inds)
        time_inds = np.unique(time_inds)

        # get topography for F stat
        t_map = T_obs[time_inds, ...].mean(axis=0)

        # get signals at the sensors contributing to the cluster
        sig_times = a_gra.times[time_inds]

        # create spatial mask
        mask = np.zeros((t_map.shape[0], 1), dtype=bool)
        mask[ch_inds, :] = True

        # initialize figure
        fig, ax_topo = plt.subplots(1, 1, figsize=(10, 3))

        # plot average test statistic and mark significant sensors
        f_evoked = mne.EvokedArray(t_map[:, np.newaxis], a_gra.info, tmin=0)
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
        plt.savefig(f'{save_dir_cluster_output}{Path("/")}{cond}_{str(clu_idx)}_prestim_.svg')
        plt.show()