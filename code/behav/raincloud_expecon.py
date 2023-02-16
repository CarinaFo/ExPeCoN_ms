### Script to make raincloud plots with the data of expecon

### Written by Anastassia Loukianov

### last update: 16.02.2023

# Import packages
import pandas as pd
import scipy.stats
import seaborn as sns
import os
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

import ptitprince as pt

# Choose directory
os.chdir('/Users/loukianov//Desktop/Berlin/Rainclouds/Plots')
os.listdir('/Users/loukianov//Desktop/Berlin/Rainclouds/Plots')

# Read dataframe
data = pd.read_csv("clean_bb.csv")
data = data.drop(data.columns[0], axis=1)

# Define parameters for the esthetics of the plot
blue = '#2a95ffff'
pink = '#ff2a2aff'
alpha = 0.5
colors = [blue, pink]

green = '#2aff95'
purple = '#2a2aff'
colors2 = [purple, green]

def prepare_data():
    """this function clean the data for each variable. The data is saved as two list (condition and congruency) with the dataframes that will be passed into the function"""
    # Prepare the data for the hitrate
    signal = data[data.isyes == 1]
    signal_grouped = signal.groupby(['ID', 'cue']).mean()['sayyes']
    signal_grouped.head()

    low_condition = signal_grouped.unstack()[0.25].reset_index()
    high_condition = signal_grouped.unstack()[0.75].reset_index()

    # Prepare the data to input for the function
    hitrate_merge = pd.merge(low_condition, high_condition)
    hitrate_melt = hitrate_merge.melt(id_vars='ID', value_name='Value', var_name='Variable')

    hitrate_condition = hitrate_melt.rename(columns={'Value': 'hitrate', 'Variable': 'condition'})

    # Prepare the data for the false alarm rate
    noise = data[data.isyes == 0]
    noise_grouped = noise.groupby(['ID', 'cue']).mean()['sayyes']
    noise_grouped.head()

    low_condition = noise_grouped.unstack()[0.25].reset_index()
    high_condition = noise_grouped.unstack()[0.75].reset_index()

    # Prepare the data to input for the function
    farate_merge = pd.merge(low_condition, high_condition)
    farate_melt = farate_merge.melt(id_vars='ID', value_name='Value', var_name='Variable')

    farate_condition = farate_melt.rename(columns={'Value': 'farate', 'Variable': 'condition'})

    # Calculate SDT from hitrate and false alarm rate
    hitrate_low = signal_grouped.unstack()[0.25]
    farate_low = noise_grouped.unstack()[0.25]

    d_prime_low = pd.Series([stats.norm.ppf(h) - stats.norm.ppf(f) for h, f in zip(hitrate_low, farate_low)])
    criterion_low = pd.Series([-0.5 * (stats.norm.ppf(h) + stats.norm.ppf(f)) for h, f in zip(hitrate_low, farate_low)])

    hitrate_high = signal_grouped.unstack()[0.75]
    farate_high = noise_grouped.unstack()[0.75]
    # Avoid infinity
    farate_high[16] = 0.00001

    d_prime_high = pd.Series([stats.norm.ppf(h) - stats.norm.ppf(f) for h, f in zip(hitrate_high, farate_high)])
    criterion_high = pd.Series([-0.5 * (stats.norm.ppf(h) + stats.norm.ppf(f)) for h, f in zip(hitrate_high, farate_high)])

    # Prepare the data for the criterion
    criterion_high = pd.DataFrame(criterion_high, columns=['values'])
    criterion = pd.DataFrame(criterion_low).join(criterion_high)
    criterion.rename(columns={0: '0.25', 'values': '0.75'}, inplace=True)

    criterion['ID'] = criterion.index

    criterion_melt = criterion.melt(id_vars='ID', value_name='Value', var_name='Variable')

    criterion_condition = criterion_melt.rename(columns={'Value': 'criterion', 'Variable': 'condition'})

    # Prepare the data for the sensitivity
    d_prime_high = pd.DataFrame(d_prime_high, columns=['values'])
    d_prime = pd.DataFrame(d_prime_low).join(d_prime_high)
    d_prime.rename(columns={0: '0.25', 'values': '0.75'}, inplace=True)

    d_prime['ID'] = d_prime.index

    d_prime_melt = d_prime.melt(id_vars='ID', value_name='Value', var_name='Variable')

    d_prime_condition = d_prime_melt.rename(columns={'Value': 'd_prime', 'Variable': 'condition'})

    # Congruency effects on confidence

    # Create new column that defines congruency
    data['congruency'] = ((data.cue == 0.25) & (data.sayyes == 0) | (data.cue == 0.75) & (data.sayyes == 1))

    data_grouped = data.groupby(['ID', 'congruency']).mean()['conf']

    con_confidence = data_grouped.unstack()[True].reset_index()
    incon_confidence = data_grouped.unstack()[False].reset_index()

    # Prepare the data to input for the function
    confidence_merge = pd.merge(con_confidence, incon_confidence).melt(id_vars='ID', value_name='Value', var_name='Variable')

    confidence_congruency = confidence_merge.rename(columns={'Value': 'confidence','Variable': 'congruency'})

    # Prepare the data for accuracy

    # Create a new column to define accuracy
    data['correct'] = ((data.isyes == 1) & (data.sayyes == 1) | (data.isyes == 0) & (data.sayyes == 0))

    data_grouped = data.groupby(['ID', 'congruency']).mean()['correct']

    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()

    # Prepare the data to use for the function
    congruency_merge = pd.merge(con_condition, incon_condition)
    accuracy_congruency = congruency_merge.melt(id_vars='ID', value_name='Value', var_name='Variable')

    accuracy_congruency = accuracy_congruency.rename(columns={'Value': 'accuracy','Variable': 'congruency'})

    data_grouped_cue = data.groupby(['ID','cue']).mean()['correct']

    acc_low = data_grouped_cue.unstack()[0.25].reset_index()
    acc_high = data_grouped_cue.unstack()[0.75].reset_index()

    accuracy_merge = pd.merge(acc_low,acc_high)
    accuracy_condition = accuracy_merge.melt(id_vars='ID', value_name='Value', var_name='Variable')

    accuracy_condition = accuracy_condition.rename(columns={'Value': 'accuracy','Variable': 'condition'})

    list_condition = [hitrate_condition, farate_condition, criterion_condition, d_prime_condition, accuracy_condition]
    list_congruency = [confidence_congruency, accuracy_congruency]
    return list_condition, list_congruency

# Loop with the dataframe in the function to make the raincloud (raincloud_function)
def plot_condition(list_condition):
    """this function plots the raincloud with a loop that use the list (list_condition) and make the rainclouds with the set of colors read and blue.
    The rainclouds are saved in the directory. A wilcoxon test is also calculated for each and print the p and the t value"""

    fig_names = ['hit_cue', 'fa_cue', 'crit_cue', 'dprime_cue', 'acc_cue']

    for i, n in zip(list_condition, fig_names):
        dx = i.iloc[:, 1].astype(float)
        dy = i.iloc[:, 2]

        raincloud_function(dx, dy, i, colors,n , "v", savefigs=True)

        plt.show()

        x = i.iloc[:41, 2].astype(float)
        y = i.iloc[41:, 2].astype(float)

        result = scipy.stats.wilcoxon(x, y, zero_method='wilcox', correction=False, alternative='two-sided', axis=0)
        print(result)

def plot_congruency(list_congruency):
    """this function plots the raincloud with a loop that use the list (list_congruency) and make the rainclouds with the set of colors2 green and blue.
    The rainclouds are saved in the directory. A wilcoxon test is also calculated for each and print the p and the t value"""

    fig_names = ['conf_cong', 'acc_cong']

    for i, n in zip(list_congruency, fig_names):
        dx = i.iloc[:, 1].astype(float)
        dy = i.iloc[:, 2]

        raincloud_function(dx, dy, i, colors2, n, "v", savefigs=True)

        plt.show()

        x = i.iloc[:41, 2].astype(float)
        y = i.iloc[41:, 2].astype(float)

        result = scipy.stats.wilcoxon(x, y, zero_method='wilcox', correction=False, alternative='two-sided', axis=0)
        print(result)

def raincloud_function(dx, dy, signal, colors,n , ort, savefigs=True):
    """this function makes raincloud plots for each variable. The data is saved as a dataframe and stored
    as a svg file"""

    [fig, ax] = plt.subplots()
    ax = pt.half_violinplot(x=dx, y=dy, data=signal, palette=colors,
                            scale="area", width=.6, inner=None, orient=ort, ax=ax)
    ax = sns.stripplot(x=dx, y=dy, data=signal, palette=colors, edgecolor="white",
                       size=3, jitter=1, zorder=0, orient=ort)
    ax = sns.boxplot(x=dx, y=dy, data=signal, color="black", width=.15, zorder=10,
                     showcaps=True, boxprops={'facecolor': 'none', "zorder": 10},
                     showfliers=True, whiskerprops={'linewidth': 2, "zorder": 10},
                     saturation=1, orient=ort)

    if savefigs:

        name_fig = f"raincloud_{n}.svg"
        fig.savefig(name_fig)
        fig.show()