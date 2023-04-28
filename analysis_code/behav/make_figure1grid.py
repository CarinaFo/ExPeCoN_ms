import os
import pandas as pd
import scipy.stats as stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np

# Set up data paths
behavpath = 'D:\\expecon_ms\\data\\behav'
savepath = 'D:\\expecon_ms\\figs\\behavior'

# Load data
data = pd.read_csv(behavpath + '\\clean_bb.csv')

# Add a 'congruency' column
data['congruency'] = ((data.cue == 0.25) & (data.sayyes == 0)) | ...
((data.cue == 0.75) & (data.sayyes == 1))

# add lagged variables
data['lagsayyes'] = data['sayyes'].shift(1)
data['lagcue'] = data['cue'].shift(1)
data['lagisyes'] = data['isyes'].shift(1)
data['lagcorrect'] = data['correct'].shift(1)

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
criterion_low = [calc_criterion(h, f) for h, f in zip(hitrate_low, farate_low)]

hitrate_high = signal_grouped.unstack()[0.75]
farate_high = noise_grouped.unstack()[0.75]
d_prime_high = [calc_dprime(h, f) for h, f in zip(hitrate_high, farate_high)]
criterion_high = [calc_criterion(h, f) for h, f in zip(hitrate_high, farate_high)]

# Filter for correct trials only
correct_only = data[data.correct == 1]

# Calculate mean confidence for each participant and congruency condition
data_grouped = correct_only.groupby(['ID', 'congruency']).mean()['conf']
con_condition = data_grouped.unstack()[1].reset_index()
incon_condition = data_grouped.unstack()[0].reset_index()

# Calculate the difference in d-prime and criterion between the high and low condition
d_diff = np.array(d_prime_low) - np.array(d_prime_high)
c_diff = np.array(criterion_low) - np.array(criterion_high)

# Define colors for the plot
blue = '#2a95ffff'
pink = '#ff2a2aff'

# Define a function to plot the figure


def plot_figure1_grid(colors=[blue, pink], medcolor=['black', 'black']):

    """Plot the figure 1 grid and the behavioral data
    Parameters
    ----------
        colors : list of strings
            The colors to use for the low and high cue condition
            medcolor : list of strings
            The colors to use for the median lines in the boxplots
            """

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
    for index in range(len(hitrate_low)):
        hr_ax.plot(1, hitrate_low.iloc[index],
                   marker='', markersize=8, color=colors[0], 
                   markeredgecolor=colors[0], alpha=.5)
        hr_ax.plot(2, hitrate_high.iloc[index],
                   marker='', markersize=8, color=colors[1], 
                   markeredgecolor=colors[1], alpha=.5)
        hr_ax.plot([1, 2], [hitrate_low.iloc[index], hitrate_high.iloc[index]],
                   marker='', markersize=0, color='gray', alpha=.25)

    hr_box = hr_ax.boxplot([hitrate_low, hitrate_high], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(hr_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(hr_box['medians'], medcolor):
        patch.set_color(color)
        
    hr_ax.set_ylabel('hit rate', fontname="Arial", fontsize=14)
    hr_ax.set_yticklabels(['0', '0.5', '1.0'], fontname="Arial", fontsize=12)
    hr_ax.text(1.3, 1, '***', verticalalignment='center', fontname='Arial', fontsize='18')

    fa_ax = fig.add_subplot(gs[5, 0])
    for index in range(len(farate_high)):
        fa_ax.plot(1, farate_low.iloc[index],
                   marker='', markersize=8, color=colors[0], 
                   markeredgecolor=colors[0], alpha=.5)
        fa_ax.plot(2, farate_high.iloc[index],
                   marker='', markersize=8, color=colors[1], 
                   markeredgecolor=colors[1], alpha=.5)
        fa_ax.plot([1, 2], [farate_low.iloc[index], farate_high.iloc[index]],
                   marker='', markersize=0, color='gray', alpha=.25)

    fa_box = fa_ax.boxplot([farate_low, farate_high], patch_artist=True)
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

    for index in range(len(criterion_low)):
        crit_ax.plot(1, criterion_low[index],
                     marker='o', markersize=0, color=colors[0], 
                     markeredgecolor=colors[0], alpha=.5)
        crit_ax.plot(2, criterion_high[index],
                     marker='o', markersize=0, color=colors[1],
                     markeredgecolor=colors[1], alpha=.5)
        crit_ax.plot([1, 2], [criterion_low[index], criterion_high[index]],
                     marker='', markersize=0, color='gray', alpha=.25)
        
    crit_box = crit_ax.boxplot([criterion_low, criterion_high],
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

    for index in range(len(d_prime_low)):
        dprime_ax.plot(1, d_prime_low[index],
                       marker='o',markersize=0, color=colors[0], 
                       markeredgecolor=colors[0], alpha=.5)
        dprime_ax.plot(2, d_prime_high[index],
                       marker='o', markersize=0, color=colors[1], 
                       markeredgecolor=colors[1], alpha=.5)
        dprime_ax.plot([1, 2], [d_prime_low[index], d_prime_high[index]],
                       marker='', markersize=0, color='gray', alpha=.25)
        
    dprime_box = dprime_ax.boxplot([d_prime_low, d_prime_high],
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
    for index in range(len(hitrate_high)):
        conf_ax.plot(1, con_condition.iloc[index, 1],
                     marker='', markersize=8, color=colors[0],
                     markeredgecolor=colors[0], alpha=.5)
        conf_ax.plot(2, incon_condition.iloc[index, 1],
                     marker='', markersize=8, color=colors[1],
                     markeredgecolor=colors[1], alpha=.5)
        conf_ax.plot([1, 2], [con_condition.iloc[index, 1], 
                             incon_condition.iloc[index, 1]],
                     marker='', markersize=0, color='gray', alpha=.25)

    conf_box = conf_ax.boxplot([con_condition.iloc[:, 1], incon_condition.iloc[:, 1]],
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

    fig.savefig(savepath + "\\figure1.svg",dpi=300, bbox_inches='tight',
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
    t, p = stats.wilcoxon(criterion_high, criterion_low)
    print(f'c: {p}')

    t, p = stats.wilcoxon(d_prime_high, d_prime_low)
    print(f'dprime: {p}')

    t, p = stats.wilcoxon(hitrate_high, hitrate_low)
    print(f'hitrate: {p}')

    t, p = stats.wilcoxon(farate_high, farate_low)
    print(f'farate: {p}')

    t, p = stats.wilcoxon(con_condition.iloc[:, 1], incon_condition.iloc[:, 1])
    print(f'confidence in correct trials only: {p}')

# Figure 2 is not cleaned up yet


def figure2():
    """Figure 2
    Parameters  ---------- None.
    Returns
    -------
    None
    """
    # load data
    # Correlate criterion with dprime

    os.chdir(savepath)

    diff_c = np.array(criterion_high) - np.array(criterion_low)
    diff_d = np.array(d_prime_high) - np.array(d_prime_low)

    sns.regplot(x=diff_c, y=diff_d, color='black', scatter_kws={'s': 50}, robust=True, fit_reg=True)
    plt.xlabel('change in dprime', fontname="Arial", fontsize=14)
    plt.ylabel('change in c', fontname="Arial", fontsize=14)
    plt.savefig('c_d_corr.png')
    plt.savefig('c_d_corr.svg')

    plt.show()
    stats.pearsonr(diff_c, diff_d) # correlation of 0.45, p < .01

    # Previous choice bias

    # load choice bias
    choice = pd.read_csv("D:\\expecon_ms\\data\\behav\\previous_choice.csv")
    choice = choice.drop(9)
    choice = choice.reset_index(drop=True)

    # plot overall choice bias


    # Create strip plot with 'tip' column
    sns.stripplot(x=list(range(1,40)), y=choice.prev)

    # Set axis labels and title
    sns.set_style("whitegrid")
    sns.set_context("talk")
    sns.set_palette("colorblind")
    sns.despine()
    plt.xlabel("Participant ID")
    plt.ylabel("Previous choice bias")
    plt.xticks([])
    plt.savefig('prev_choice_groups.png')
    plt.savefig('prev_choice_groups.svg')

    # Show the plot
    plt.show()

    # only for the high exp trials

    sns.stripplot(x=list(range(1,40)), y=choice.prev_high)

    # Set axis labels and title
    sns.set_style("whitegrid")
    sns.set_context("talk")
    sns.set_palette("colorblind")
    sns.despine()
    plt.xlabel("Participant ID")
    plt.ylabel("Previous choice bias")
    plt.xticks([])
    # Show the plot
    plt.show()

    sns.stripplot(x=list(range(1,40)), y=choice.prev_low)
    """
    """ # Set axis labels and title
    sns.set_style("whitegrid")
    sns.set_context("talk")
    sns.set_palette("colorblind")
    sns.despine()
    plt.xlabel("participant")
    plt.ylabel("previous choice bias")
    plt.xticks([])

    # Show the plot
    plt.show()

    # Define repeater and alternator as boolean mask

    rep = choice.prev>0 # 19 repeaters
    alt = choice.prev<0 # 20 alternator

    # check correlation between criterion and choice bias for both groups

    # repeaters, sign. neg correlation (-0.73)

    sns.regplot(x=diff_c[rep], y=choice.prev[rep], color='black', scatter_kws={'s': 50}, robust=True, fit_reg=True)
    plt.xlabel('change in c', fontname="Arial", fontsize=14)
    plt.ylabel('choice bias', fontname="Arial", fontsize=14)
    plt.savefig('diff_c_prev_rep.png')
    plt.savefig('diff_c_prev_rep.svg')
    plt.show()

    stats.pearsonr(diff_c[rep], choice.prev[rep]) # overall correlation of 0.5

    # alternators, no sign. correlation (r=0.16, p=0.5)

    sns.regplot(x=diff_c[alt], y=choice.prev[alt], color='black', scatter_kws={'s': 50}, robust=True, fit_reg=True)
    plt.xlabel('change in c', fontname="Arial", fontsize=14)
    plt.ylabel('choice bias', fontname="Arial", fontsize=14)
    plt.savefig('diff_c_prev_alt.png')
    plt.savefig('diff_c_prev_alt.svg')
    plt.show()

    stats.pearsonr(diff_c[alt], choice.prev[alt]) 

    # repeaters
    # load questionnaire data (intolerance of uncertainty)

    q_df = pd.read_csv(r"D:\expecon\data\behav_brain\questionnaire_data\q_clean.csv")
    # Get the row index of the original DataFrame

    # Create a Boolean mask to identify the rows containing the values to drop
    mask = q_df['ID'].isin([16,32,42,45])

    # Drop the rows containing the values to drop
    q_df = q_df.drop(q_df[mask].index)

    q_df = q_df.reset_index()

    row_index = q_df.index

    clean_q = q_df.dropna(subset = "iu18_A")

    # Get the row index of the dropped rows
    dropped_row_index = row_index.difference(clean_q.index)

    # drop the 2 NaN from questionnaire data
    choice = choice.drop([dropped_row_index], axis=0)
    diff_c = np.delete(diff_c, [dropped_row_index], axis=0)

    # correlate c diff and prev choice bias with IUQ

    stats.pearsonr(stats.zscore(clean_q.iu_sum), diff_c)

    # Plot regplot 

    sns.regplot(x=stats.zscore(clean_q.iu_sum), y=diff_c, color='black', scatter_kws={'s': 50}, robust=True, fit_reg=True)
    plt.xlabel('Intolerance of Uncertainty zscored', fontname="Arial", fontsize=14)
    plt.ylabel('previous choice bias', fontname="Arial", fontsize=14)
    plt.savefig('prev_choice_IUQ.png')
    plt.savefig('prev_choice_IUQ.svg')

    # Show the plot
    plt.show()

    sns.regplot(x=stats.zscore(clean_q.iu_sum), y=diff_c, color='black', scatter_kws={'s': 50}, robust=True, fit_reg=True)
    plt.xlabel('Intolerance of Uncertainty zscored', fontname="Arial", fontsize=14)
    plt.ylabel('criterion change', fontname="Arial", fontsize=14)
    plt.savefig('c_diff_IUQ.png')
    plt.savefig('c_diff_IUQ.svg')

    # Show the plot
    plt.show()