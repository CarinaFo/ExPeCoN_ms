import os
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np


def prepro_behavioral_data():

    """ This function preprocesses the behavioral data.
    It removes participants/blocks with excessively high/low hitrates,
    removes trials with no response or super fast responses,
    and removes the first trial from each block (weird trigger behavior).
    It also adds columns for lagged variables.
    Returns:
    data: pandas dataframe containing the preprocessed behavioral data
    """

    behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'

    # Load the behavioral data from the specified path
    data = []
    for root, dirs, files in os.walk(behavpath):
        for name in files:
            if 'behav_cleaned_for_eeg.csv' in name:
                data = pd.read_csv(os.path.join(root, name))

    # Clean up the dataframe by dropping unneeded columns
    columns_to_drop = ["Unnamed: 0.2", 'Unnamed: 0.1', 'Unnamed: 0']
    data = data.drop(columns_to_drop, axis=1)

    # Change the block number for participant 7's block 3
    data.loc[(144*2):(144*3), 'block'] = 4

    # Drop participants because of excessively high/low hitrates
    drop_participants = [16, 32, 40, 45]
    data = data.drop(data[data.ID.isin(drop_participants)].index)

    # add a column that indicates correct responses & congruency
    data['correct'] = data.sayyes == data.isyes
    # Add a 'congruency' column
    data['congruency'] = ((data.cue == 0.25) & (data.sayyes == 0)) | ((
                          data.cue == 0.75) & (data.sayyes == 1))
    
    data = data.reset_index()

    # add a column that combines the confidence ratings and the detection response

    conf_resp = [4 if data.loc[i, 'sayyes'] == 1 and data.loc[i, 'conf'] == 1 else
                 3 if data.loc[i, 'sayyes'] == 0 and data.loc[i, 'conf'] == 1 else
                 2 if data.loc[i, 'sayyes'] == 1 and data.loc[i, 'conf'] == 0 else
                 1 for i in range(len(data))]
    
    data['conf_resp'] = conf_resp

    # add lagged variables
    data['prevsayyes'] = data['sayyes'].shift(1)
    data['prevcue'] = data['cue'].shift(1)
    data['previsyes'] = data['isyes'].shift(1)
    data['prevcorrect'] = data['correct'].shift(1)
    data['prevconf'] = data['conf'].shift(1)
    data['prevconf_resp'] = data['conf_resp'].shift(1)

    # Remove blocks with hitrates < 0.2 or > 0.8
    drop_blocks = [(10, 6), (12, 6), (26, 4), (30, 3), (39, 3)]
    for participant, block in drop_blocks:
        data = data.drop(data[((data.ID == participant) &
                               (data.block == block))].index)

    # remove no response trials or super fast responses
    data = data.drop(data[data.respt1 == 2.5].index)
    data = data.drop(data[data.respt1 < 0.1].index)
    # remove the first trial from each block (weird trigger behavior)
    data = data.drop(data[data.trial == 1].index)

    # save the preprocessing data
    os.chdir(behavpath)
    data.to_csv("prepro_behav_data.csv")


def prepare_behav_data(exclude_high_fa=False):
    """
    This function prepares the behavioral data for the figure 1 grid.
    It calculates hit rates, false alarm rates, d-prime, and criterion
    for each participant and cue condition.
    It also calculates mean confidence for each participant and congruency
    condition.
    Parameters:
    exclude_high_fa: Boolean Yes/No (exclude participants with fa rate > 0.4)
    Returns:
    c_cond: list of dataframes containing mean confidence for each participant
    and congruency condition
    d_cond: list of dataframes containing d-prime for each participant and
    cue condition
    fa_cond: list of dataframes containing false alarm rates for each
    participant and cue condition
    hit_cond: list of dataframes containing hit rates for each participant
    and cue condition
    conf_con: list of dataframes containing mean confidence
    and congruency condition
    """

    # Set up data path
    behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'

    # Load data
    data = pd.read_csv(behavpath + '\\prepro_behav_data.csv')

    # Drop participants because of excessively high fa rates in high cue condition
    if exclude_high_fa:
        mask = data['ID'].isin([15, 43, 46])
        data = data[~mask]

    # Calculate hit rates by participant and cue condition
    signal = data[data.isyes == 1]
    signal_grouped = signal.groupby(['ID', 'cue']).mean()['sayyes']

    # Calculate noise rates by participant and cue condition
    noise = data[data.isyes == 0]
    noise_grouped = noise.groupby(['ID', 'cue']).mean()['sayyes']

    # Calculate d prime and criterion scores
    def calc_dprime(hitrate, farate):
        return stats.norm.ppf(hitrate) - stats.norm.ppf(farate)

    def calc_criterion(hitrate, farate):
        return -0.5 * (stats.norm.ppf(hitrate) + stats.norm.ppf(farate))

    hitrate_low = signal_grouped.unstack()[0.25]
    farate_low = noise_grouped.unstack()[0.25]
    d_prime_low = [calc_dprime(h, f) for h, f in zip(hitrate_low, farate_low)]
    criterion_low = [calc_criterion(h, f) for h, f in zip(hitrate_low,
                                                          farate_low)]

    hitrate_high = signal_grouped.unstack()[0.75]
    farate_high = noise_grouped.unstack()[0.75]
    d_prime_high = [calc_dprime(h, f) for h, f in zip(hitrate_high,
                                                      farate_high)]
    criterion_high = [calc_criterion(h, f) for h, f in zip(hitrate_high,
                                                           farate_high)]

    # create boolean mask for participants with very high farates
    faratehigh_indices = np.where(farate_high > 0.4)  # 3 participants with fa rates > 0.4

    c_cond = pd.DataFrame([criterion_low, criterion_high]).T
    d_cond = pd.DataFrame([d_prime_low, d_prime_high]).T
    fa_cond = [farate_low, farate_high]
    hit_cond = [hitrate_low, hitrate_high]

    # Filter for correct trials only
    correct_only = data[data.correct == 1]

    # Calculate mean confidence for each participant and congruency condition
    data_grouped = correct_only.groupby(['ID', 'congruency']).mean()['conf']
    con_condition = data_grouped.unstack()[1].reset_index()
    incon_condition = data_grouped.unstack()[0].reset_index()
    conf_con = [con_condition, incon_condition]

    # Calculate mean accuracy for each participant and ce condition
    data_grouped = data.groupby(['ID', 'cue']).mean()['correct']
    acc_high = data_grouped.unstack()[0.75].reset_index()
    acc_low = data_grouped.unstack()[0.25].reset_index()
    acc_cue = [acc_low, acc_high]

    # Calculate the difference in d-prime and c between the high/low condition
    d_diff = np.array(d_prime_low) - np.array(d_prime_high)
    c_diff = np.array(criterion_low) - np.array(criterion_high)

    return d_diff, c_diff, conf_con, d_cond, c_cond, fa_cond, hit_cond


def plot_figure1_grid(blue='#2a95ffff', pink='#ff2a2aff',
                      medcolor=['black', 'black'],
                      savepath_fig1='D:\\expecon_ms\\figs\\behavior'):

    """Plot the figure 1 grid and the behavioral data, including accuracy plot
    Parameters
    ----------
        colors : list of strings
            The colors to use for the low and high cue condition
            medcolor : list of strings
            The colors to use for the median lines in the boxplots
            """

    colors=[blue, pink]

    fig = plt.figure(figsize=(8, 10), tight_layout=True)  # original working was 10,12
    gs = gridspec.GridSpec(6, 4)

    schem_01_ax = fig.add_subplot(gs[0:2, 0:])
    schem_01_ax.set_yticks([])
    schem_01_ax.set_xticks([])

    schem_02_ax = fig.add_subplot(gs[2:4,0:])
    schem_02_ax.set_yticks([])
    schem_02_ax.set_xticks([])

    hr_ax = fig.add_subplot(gs[4, 0])
    # Plot individual data points 
    for index in range(len(hit_cond[0])):
        hr_ax.plot(1, hit_cond[0].iloc[index],
                   marker='', markersize=8, color=colors[0], 
                   markeredgecolor=colors[0], alpha=.5)
        hr_ax.plot(2, hit_cond[1].iloc[index],
                   marker='', markersize=8, color=colors[1], 
                   markeredgecolor=colors[1], alpha=.5)
        hr_ax.plot([1, 2], [hit_cond[0].iloc[index], hit_cond[1].iloc[index]],
                   marker='', markersize=0, color='gray', alpha=.25)

    hr_box = hr_ax.boxplot([hit_cond[0], hit_cond[1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(hr_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(hr_box['medians'], medcolor):
        patch.set_color(color)
        
    hr_ax.set_ylabel('hit rate', fontname="Arial", fontsize=14)
    hr_ax.set_yticklabels(['0', '0.5', '1.0'], fontname="Arial", fontsize=12)
    hr_ax.text(1.3, 1, '***', verticalalignment='center', fontname='Arial',
               fontsize='18')

    fa_ax = fig.add_subplot(gs[5, 0])
    for index in range(len(fa_cond[0])):
        fa_ax.plot(1, fa_cond[0].iloc[index],
                   marker='', markersize=8, color=colors[0], 
                   markeredgecolor=colors[0], alpha=.5)
        fa_ax.plot(2, fa_cond[1].iloc[index],
                   marker='', markersize=8, color=colors[1], 
                   markeredgecolor=colors[1], alpha=.5)
        fa_ax.plot([1, 2], [fa_cond[0].iloc[index], fa_cond[1].iloc[index]],
                   marker='', markersize=0, color='gray', alpha=.25)

    fa_box = fa_ax.boxplot([fa_cond[0], fa_cond[1]], patch_artist=True)
    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(fa_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(fa_box['medians'], medcolor):
        patch.set_color(color)

    fa_ax.set_ylabel('fa rate', fontname="Arial", fontsize=14)
    fa_ax.set_yticklabels(['0', '0.5', '1.0'], fontname="Arial", fontsize=12)
    fa_ax.text(1.3, 1, '***', verticalalignment='center', fontname='Arial', 
               fontsize='18')

    crit_ax = fig.add_subplot(gs[4:, 1])
    crit_ax .set_ylabel('c', fontname="Arial", fontsize=14)
    crit_ax.text(1.3, 1.5, '***', verticalalignment='center', fontname='Arial',
                 fontsize='18')

    crit_ax.set_ylim(-0.5, 1.5)
    crit_ax.set_yticks([-0.5, 0.5, 1.5])
    crit_ax.set_yticklabels(['-0.5', '0.5', '1.5'], fontname="Arial", 
                            fontsize=12)

    for index in range(len(c_cond[0])):
        crit_ax.plot(1, c_cond[0][index],
                     marker='o', markersize=0, color=colors[0], 
                     markeredgecolor=colors[0], alpha=.5)
        crit_ax.plot(2, c_cond[1][index],
                     marker='o', markersize=0, color=colors[1],
                     markeredgecolor=colors[1], alpha=.5)
        crit_ax.plot([1, 2], [c_cond[0][index], c_cond[1][index]],
                     marker='', markersize=0, color='gray', alpha=.25)
        
    crit_box = crit_ax.boxplot([c_cond[0], c_cond[1]],
                               patch_artist=True)  # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(crit_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(crit_box['medians'], medcolor):
        patch.set_color(color)

    dprime_ax = fig.add_subplot(gs[4:, 2])
    dprime_ax .set_ylabel('dprime', fontname="Arial", fontsize=14)
    dprime_ax.text(1.4, 3, 'n.s', verticalalignment='center', fontname='Arial',
                   fontsize='13')

    dprime_ax.set_ylim(0, 3)
    dprime_ax.set_yticks([0, 1.5, 3])
    dprime_ax.set_yticklabels(['0', '1.5', '3.0'], fontname="Arial", 
                              fontsize=12)

    for index in range(len(d_cond[0])):
        dprime_ax.plot(1, d_cond[0][index],
                       marker='o',markersize=0, color=colors[0], 
                       markeredgecolor=colors[0], alpha=.5)
        dprime_ax.plot(2, d_cond[1][index],
                       marker='o', markersize=0, color=colors[1], 
                       markeredgecolor=colors[1], alpha=.5)
        dprime_ax.plot([1, 2], [d_cond[0][index], d_cond[1][index]],
                       marker='', markersize=0, color='gray', alpha=.25)
        
    dprime_box = dprime_ax.boxplot([d_cond[0], d_cond[1]],
                                   patch_artist=True)  # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(dprime_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(dprime_box['medians'], medcolor):
        patch.set_color(color)

    conf_ax = fig.add_subplot(gs[4:, 3])
    conf_ax .set_ylabel('mean confidence', fontname="Arial", fontsize=14)
    conf_ax.set_yticklabels(['0', '0.5', '1.0'], fontname="Arial", fontsize=12)
    conf_ax.text(1.3, 1, '***', verticalalignment='center', fontname='Arial',
                 fontsize='18')

    # Plot individual data points 
    for index in range(len(conf_con[0])):
        conf_ax.plot(1, conf_con[0].iloc[index, 1],
                     marker='', markersize=8, color=colors[0],
                     markeredgecolor=colors[0], alpha=.5)
        conf_ax.plot(2, conf_con[1].iloc[index, 1],
                     marker='', markersize=8, color=colors[1],
                     markeredgecolor=colors[1], alpha=.5)
        conf_ax.plot([1, 2], [conf_con[0].iloc[index, 1], 
                             conf_con[1].iloc[index, 1]],
                     marker='', markersize=0, color='gray', alpha=.25)

    conf_box = conf_ax.boxplot([conf_con[0].iloc[:, 1], conf_con[1].iloc[:, 1]],
                               patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    colors = ['white', 'black']
    for patch, color in zip(conf_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    medcolor = ['black', 'white']
    for patch, color in zip(conf_box['medians'], medcolor):
        patch.set_color(color)

    for plots in [hr_ax, fa_ax, conf_ax]:
        plots.set_ylim(0, 1)
        plots.set_yticks([0, 0.5, 1.0])

    for plots in [hr_ax, fa_ax, crit_ax, dprime_ax, conf_ax]:
        plots.spines['top'].set_visible(False)
        plots.spines['right'].set_visible(False)
        plots.set_xticks([1, 2])
        plots.set_xlim(0.5, 2.5)
        plots.set_xticklabels(['', ''])
        if plots != hr_ax:
            plots.set_xticklabels(['0.25', '0.75'], fontname="Arial",
                                  fontsize=12)
            plots.set_xlabel('P (Stimulus)', fontname="Arial", fontsize=14)
        if plots == conf_ax:
            plots.set_xticklabels(['congruent', 'incongruent'],
                                  fontname="Arial", fontsize=12,
                                  rotation=30)
            plots.set_xlabel('')

    fig.savefig(savepath_fig1 + "\\figure1_exclhighfa.svg",dpi=300, bbox_inches='tight',
                format='svg')
    plt.show()

    # save accuracy plot for supplementary material

    # Plot individual data points 
    for index in range(len(acc_cue[0])):
        plt.plot(1, acc_cue[0].iloc[index, 1],
                 marker='', markersize=8, color=colors[0],
                 markeredgecolor=colors[0], alpha=.5)
        plt.plot(2, acc_cue[1].iloc[index, 1],
                 marker='', markersize=8, color=colors[1],
                 markeredgecolor=colors[1], alpha=.5)
        plt.plot([1, 2], [acc_cue[0].iloc[index, 1],
                 acc_cue[1].iloc[index, 1]],
                 marker='', markersize=0, color='gray', alpha=.25)

    hr_box = plt.boxplot([acc_cue[0].iloc[:, 1], acc_cue[1].iloc[:, 1]], 
                         patch_artist=True)

    hr_ax.set_ylabel('accuracy', fontname="Arial", fontsize=14)
    hr_ax.set_yticklabels(['0', '0.5', '1.0'], fontname="Arial", fontsize=12)
    hr_ax.text(1.3, 1, '***', verticalalignment='center', fontname='Arial',
               fontsize='18')
    plt.savefig(savepath_fig1 + "\\acc_cue.svg", dpi=300, bbox_inches='tight',
                format='svg')
    plt.show()


def stats_figure1():
    """stats for Figure 1
    Parameters  ---------- None.
    Returns
    -------
    None.
    """
    # non parametric
    t, p = stats.wilcoxon(c_cond[0], c_cond[1])
    print(f'c: {p}')

    t, p = stats.wilcoxon(d_cond[0], d_cond[1])
    print(f'dprime: {p}')

    t, p = stats.wilcoxon(hit_cond[0], hit_cond[1])
    print(f'hitrate: {p}')

    t, p = stats.wilcoxon(fa_cond[0], fa_cond[1])
    print(f'farate: {p}')

    t, p = stats.wilcoxon(conf_con[0].iloc[:, 1], conf_con[1].iloc[:, 1])
    print(f'confidence in correct trials only: {p}')

    t, p = stats.wilcoxon(acc_cue[0].iloc[:, 1], acc_cue[1].iloc[:, 1])
    print(f'accuracy between conditions: {p}')