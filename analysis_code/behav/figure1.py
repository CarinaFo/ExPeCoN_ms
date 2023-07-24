import math
import os
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as sm
from matplotlib.font_manager import FontProperties as fm

# Set Arial as the default font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14


def prepro_behavioral_data():
 
    """ This function preprocesses the behavioral data.
    We remove trials with no response or super fast responses
    and add aditional columns to the dataframe.
    Returns:
    data: pandas dataframe containing the preprocessed behavioral data
    """

    behavpath = Path('D:/expecon_ms/data/behav/behav_df/')

    # Load the behavioral data from the specified path
    data = []
    for root, dirs, files in os.walk(behavpath):
        for name in files:
            if 'behav_cleaned_for_eeg.csv' in name:
                data = pd.read_csv(os.path.join(root, name))

    # Clean up the dataframe by dropping unnecessary columns
    columns_to_drop = ["Unnamed: 0.2", 'Unnamed: 0.1', 'Unnamed: 0']
    data = data.drop(columns_to_drop, axis=1)

    # Change the block number for participant 7's block 3
    data.loc[(144*2):(144*3), 'block'] = 4

    # add a column that indicates correct responses & congruency
    data['correct'] = data.sayyes == data.isyes
    # Add a 'congruency' column
    data['congruency'] = ((data.cue == 0.25) & (data.sayyes == 0)) | ((
                          data.cue == 0.75) & (data.sayyes == 1))
    # Add a 'congruency stimulus' column
    data['congruency_stim'] = ((data.cue == 0.25) & (data.isyes == 0)) | ((
                          data.cue == 0.75) & (data.isyes == 1))
    
    # add a column that combines the confidence ratings and the 
    # detection response
    conf_resp = [4 if data.loc[i, 'sayyes'] == 1 and
                 data.loc[i, 'conf'] == 1 else
                 3 if data.loc[i, 'sayyes'] == 0 and
                 data.loc[i, 'conf'] == 1 else
                 2 if data.loc[i, 'sayyes'] == 1 and
                 data.loc[i, 'conf'] == 0 else
                 1 for i in range(len(data))]
    
    data['conf_resp'] = conf_resp

    # add lagged variables
    data['prevsayyes'] = data['sayyes'].shift(1)
    data['prevcue'] = data['cue'].shift(1)
    data['previsyes'] = data['isyes'].shift(1)
    data['prevcorrect'] = data['correct'].shift(1)
    data['prevconf'] = data['conf'].shift(1)
    data['prevconf_resp'] = data['conf_resp'].shift(1)

    excl_noresp = len(data[data.respt1 == 2.5])
    excl_fastresp = len(data[data.respt1 < 0.1])

    # remove no response trials or super fast responses
    data = data.drop(data[data.respt1 == 2.5].index)
    data = data.drop(data[data.respt1 < 0.1].index)

    # save the preprocessing data
    data.to_csv(f'{behavpath}behav_data_exclrts.csv')


def calculate_sdt_dataframe(df, signal_col, response_col, subject_col, condition_col):
    """
    Calculates SDT measures (d' and criterion) for each participant and each condition based on a dataframe.
    
    Arguments:
    df -- Pandas dataframe containing the data
    signal_col -- Name of the column indicating signal presence (e.g., 'signal')
    response_col -- Name of the column indicating participant response (e.g., 'response')
    subject_col -- Name of the column indicating participant ID (e.g., 'subject_id')
    condition_col -- Name of the column indicating condition (e.g., 'condition')
    
    Returns:
    Pandas dataframe containing the calculated SDT measures (d' and criterion) for each participant and condition.
    """
    
    # Apply Hautus correction and calculate SDT measures for each participant and condition
    # Hautus correction depends on the condition (0.25 or 0.75)
    results = []
    subjects = df[subject_col].unique()
    conditions = df[condition_col].unique()
    
    for subject in subjects:
        for condition in conditions:
            subset = df[(df[subject_col] == subject) &
                         (df[condition_col] == condition)]
            
            detect_hits = subset[(subset[signal_col] == True) &
                                 (subset[response_col] == True)].shape[0]
            detect_misses = subset[(subset[signal_col] == True) &
                                   (subset[response_col] == False)].shape[0]
            false_alarms = subset[(subset[signal_col] == False) &
                                  (subset[response_col] == True)].shape[0]
            correct_rejections = subset[(subset[signal_col] == False) &
                                        (subset[response_col] == False)].shape[0]
            
            # log linear correction (Hautus, 1995)
            if condition == 0.75:
                hit_rate = (detect_hits + 0.5) / (detect_hits + detect_misses + 1)
                false_alarm_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)
            else:
                hit_rate = (detect_hits + 0.5) / (detect_hits + detect_misses + 1)
                false_alarm_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)

            d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate)
            criterion = -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(false_alarm_rate))
            
            results.append((subject, condition, hit_rate, false_alarm_rate, d_prime, criterion))
    
    # Create a new dataframe with the results
    results_df = pd.DataFrame(results, columns=[subject_col, condition_col, 
                                                'hit_rate', 'fa_rate', 'dprime',
                                                'criterion'])
    
    return results_df


def exclude_data():
    """
    Excludes participants and trials from the data based on the exclusion criteria.
    Arguments:
    None

    Returns:
    data -- Pandas dataframe containing the data

    """

    # Set up data path
    behavpath = Path('D:/expecon_ms/data/behav/behav_df/')

    # Load data
    data = pd.read_csv(f'{behavpath}behav_data_exclrts.csv')

    # Calculate hit rates by participant and cue condition
    signal = data[data.isyes == 1]
    hitrate_per_subject = signal.groupby(['ID']).mean()['sayyes']

    print(f"Mean hit rate: {np.mean(hitrate_per_subject):.2f}")
    print(f"Standard deviation: {np.std(hitrate_per_subject):.2f}")
    print(f"Minimum hit rate: {np.min(hitrate_per_subject):.2f}")
    print(f"Maximum hit rate: {np.max(hitrate_per_subject):.2f}")

    # Calculate hit rates by participant and block condition
    hitrate_per_block = signal.groupby(['ID', 'block']).mean()['sayyes']

    # Filter the grouped object based on hit rate conditions
    filtered_groups = hitrate_per_block[(hitrate_per_block > 0.9) | (hitrate_per_block < 0.2)]

    # Extract the ID and block information from the filtered groups
    remove_hitrates = filtered_groups.reset_index()

    # Calculate hit rates by participant and cue condition
    noise = data[data.isyes == 0]
    farate_per_block = noise.groupby(['ID', 'block']).mean()['sayyes']

    # Filter the grouped object based on hit rate conditions
    filtered_groups = farate_per_block[farate_per_block > 0.4]

    # Extract the ID and block information from the filtered groups
    remove_farates = filtered_groups.reset_index()

    # Filter the grouped objects based on the conditions
    #filtered_groups = hitrate_per_block[hitrate_per_block < farate_per_block]  # Hit rate < False alarm rate
    filtered_groups = hitrate_per_block[hitrate_per_block - farate_per_block < 0.05]  # Difference < 0.1

    # Extract the ID and block information from the filtered groups
    hitvsfarate = filtered_groups.reset_index()

    # Concatenate the dataframes
    combined_df = pd.concat([remove_hitrates, remove_farates, hitvsfarate])

    # Remove duplicate rows based on 'ID' and 'block' columns
    unique_df = combined_df.drop_duplicates(subset=['ID', 'block'])

    # Merge the big dataframe with unique_df to retain only the non-matching rows
    filtered_df = data.merge(unique_df, on=['ID', 'block'], how='left',
                             indicator=True,
                             suffixes=('', '_y'))
    filtered_df = filtered_df[filtered_df['_merge'] == 'left_only']

    # Remove the '_merge' column
    data = filtered_df.drop('_merge', axis=1)

    data.to_csv(f'{behavpath}prepro_behav_data.csv')
    
    return data


def prepare_for_plotting(exclude_high_fa=False):

    """
    Prepares the data for plotting.
    Arguments:
    exclude_high_fa -- Boolean indicating whether to exclude participants with high false alarm rates
    
    Returns:
    data -- Pandas dataframe containing the data
    """

    data = exclude_data()

    # calculate hit rates, false alarm rates, d-prime, and criterion per participant and cue condition
    # and per condition
    df_sdt = calculate_sdt_dataframe(data, "isyes", "sayyes", "ID", "cue")

    # create boolean mask for participants with very high farates
    faratehigh_indices = np.where(df_sdt.fa_rate[df_sdt.cue == 0.75] > 0.4)  
    # 3 participants with fa rates > 0.4
    print(f"Index of participants with high farates: {faratehigh_indices}")

    if exclude_high_fa is True:
        # exclude participants with high farates
        indices = [f+7 for f in faratehigh_indices]  # add 7 to the indices to get the correct participant number
        data = data[~data['ID'].isin(indices[0])]

    # calculate hit rates, false alarm rates, d-prime, and criterion per participant and cue condition
    # and per condition
    df_sdt = calculate_sdt_dataframe(data, "isyes", "sayyes", "ID", "cue")

    # Filter for correct trials only
    correct_only = data[data.correct == 1]
    incorrect_only = data[data.correct == 0]
    yes_response = correct_only[correct_only.sayyes == 1]
    no_response = correct_only[correct_only.sayyes == 0]

    # Calculate mean confidence for each participant and congruency condition
    data_grouped = correct_only.groupby(['ID', 'congruency']).mean()['conf']
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    conf_con = [con_condition, incon_condition]

    # Calculate mean confidence for each participant and congruency condition
    data_grouped = yes_response.groupby(['ID', 'congruency']).mean()['conf']
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    conf_con_yes = [con_condition, incon_condition]

    # Calculate mean confidence for each participant and congruency condition
    data_grouped = no_response.groupby(['ID', 'congruency']).mean()['conf']
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    conf_con_no = [con_condition, incon_condition]

    # Calculate mean rts for each participant and congruency condition
    data_grouped = correct_only.groupby(['ID', 'congruency_stim']).mean()['respt1']
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    rt_con = [con_condition, incon_condition]

    # Calculate mean rts for each participant and congruency condition
    data_grouped = incorrect_only.groupby(['ID', 'congruency_stim']).mean()['respt1']
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    rt_con_incorrect = [con_condition, incon_condition]

    # Calculate mean rts for each participant and congruency condition
    data_grouped = yes_response.groupby(['ID', 'congruency_stim']).mean()['respt1']
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    rt_con_yes = [con_condition, incon_condition]

    # Calculate mean rts for each participant and congruency condition
    data_grouped = no_response.groupby(['ID', 'congruency_stim']).mean()['respt1']
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    rt_con_no = [con_condition, incon_condition]

    # Calculate mean accuracy for each participant and cue condition
    data_grouped = data.groupby(['ID', 'cue']).mean()['correct']
    acc_high = data_grouped.unstack()[0.75].reset_index()
    acc_low = data_grouped.unstack()[0.25].reset_index()
    acc_cue = [acc_low, acc_high]

    # Calculate mean confidence for each participant and cue condition
    data_grouped = data.groupby(['ID', 'cue']).mean()['conf']
    conf_high = data_grouped.unstack()[0.75].reset_index()
    conf_low = data_grouped.unstack()[0.25].reset_index()
    conf_cue = [conf_low, conf_high]
    
    conditions = df_sdt, conf_con, conf_cue, acc_cue, conf_con_yes, \
                 conf_con_no, rt_con, rt_con_yes, rt_con_incorrect
    
    return conditions, exclude_high_fa

def correlate_sdt_with_questionnaire():
    
    # load questionarre data (uncertainty of intolerance)
    q_path = Path('D:/expecon_ms/questionnaire/q_clean.csv')

    q_data = pd.read_csv(q_path)

    data = exclude_data()

    df_sdt = calculate_sdt_dataframe(data, "isyes", "sayyes", "ID", "cue")

    high_c = list(df_sdt.criterion[df_sdt.cue ==0.75])
    low_c = list(df_sdt.criterion[df_sdt.cue == 0.25])

    high_d = list(df_sdt.dprime[df_sdt.cue ==0.75])
    low_d = list(df_sdt.dprime[df_sdt.cue == 0.25])

    diff_c= [x - y for x, y in zip(low_c, high_c)]
    diff_d = [x - y for x, y in zip(low_d, high_d)]

    # Assuming diff_c and q_data are numpy arrays or pandas Series

    # Filter the arrays based on the condition q_data['iu_sum'] > 0
    filtered_diff_c = np.array(diff_c)[q_data['iu_sum'] > 0]
    filtered_iu_sum = q_data['iu_sum'][q_data['iu_sum'] > 0]

    df = pd.DataFrame({'y': filtered_diff_c, 'X': filtered_iu_sum})

    # Select columns representing questionnaire score and criterion change
    X = df['X']
    y = df['y']

    # Drop missing values if necessary
    df.dropna(subset=['X', 'y'], inplace=True)

    # Fit the linear regression model
    X = sm.add_constant(X)  # Add constant term for intercept
    regression_model = sm.OLS(y, X)
    regression_results = regression_model.fit()

    # Extract slope, intercept, and p-value
    slope = regression_results.params[1]
    intercept = regression_results.params[0]
    p_value = regression_results.pvalues[1]

    # Make predictions
    predictions = regression_results.predict(X)

    # Set black or gray colors for the plot
    data_color = 'black'
    regression_line_color = 'gray'
    text_color = 'black'

    # Plot the regression line using seaborn regplot
    sns.regplot(data=df, x='X', y='y', color=data_color)
    plt.xlabel('Intolerance of Uncertainty score (zscored)', color=text_color)
    plt.ylabel('Criterion Change', color=text_color)
    plt.annotate(f'Slope: {slope:.2f}\n p-value: {p_value:.2f}', 
                 xy=(0.05, 0.85), xycoords='axes fraction',
                 color=text_color)
    
    savefig_path = Path('D:/expecon_ms/figs/manuscript_figures/Figure2B/')
                        
    plt.savefig(f'{savefig_path}linreg_c_q.png', dpi=300)
    plt.savefig(f'{savefig_path}linreg_c_q.svg', dpi=300)
    plt.show()


def diff_from_optimal_criterion():

    """ This function calculates the difference between the optimal criterion
    and the mean criterion for each participant and cue condition."""

    # calculate the optimal criterion (c) for a given base rate of signal and catch trials, and cost of hit and false alarm
    def calculate_optimal_c(base_rate_signal, base_rate_catch, cost_hit, cost_false_alarm):
        llr = math.log((base_rate_signal / base_rate_catch) * (cost_hit / cost_false_alarm))
        c = -0.5 * llr
        return c

    c_low = calculate_optimal_c(0.25, 0.75, 1, 1)
    print("Optimal criterion (c) for low stimulus probability:", c_low)

    c_high = calculate_optimal_c(0.75, 0.25, 1, 1)
    print("Optimal criterion (c) for high stimulus probability:", c_high)

    subop_low = [c - c_low for c in df_sdt.criterion[df_sdt.cue == 0.25]]
    mean_low = np.mean(subop_low)
    std_low = np.std(subop_low)
    print(mean_low)
    print(std_low)
    
    subop_high = [c - c_high for c in df_sdt.criterion[df_sdt.cue == 0.75]]
    mean_high = np.mean(subop_high)
    std_high = np.std(subop_high)
    print(mean_high)
    print(std_high)


def plot_figure1_grid(savepath_fig1=Path('D:/expecon_ms/figs/manuscript_figures/Figure1/')):

    """Plot the figure 1 grid and the behavioral data

    Parameters
    ----------
    savepath_fig1 : str
        Path to save the figure to

    Returns
    -------
    None
    """

    # load data
    conditions, exclude_high_fa = prepare_for_plotting(exclude_high_fa=True)

    # unpack data
    df_sdt, conf_con, conf_cue, acc_cue, conf_con_yes, conf_con_no, rt_con, \
    rt_con_yes, rt_con_incorrect = conditions
    
    # set colors for both conditions
    blue = '#0571b0'  # 0.25 cue
    red = '#ca0020' # 0.75 cue

    colors = [blue, red]
    medcolor = ['black', 'black']

    fig = plt.figure(figsize=(8, 10), tight_layout=True)  # original working was 10,12
    gs = gridspec.GridSpec(6, 4)

    schem_01_ax = fig.add_subplot(gs[0:2, 0:])
    schem_01_ax.set_yticks([])
    schem_01_ax.set_xticks([])

    schem_02_ax = fig.add_subplot(gs[2:4, 0:])
    schem_02_ax.set_yticks([])
    schem_02_ax.set_xticks([])

    hr_ax = fig.add_subplot(gs[4, 0])

    # Plot hit rate
    for index in range(len(df_sdt.hit_rate[df_sdt.cue == 0.25])):
        hr_ax.plot(1, df_sdt.hit_rate[df_sdt.cue == 0.25].iloc[index],
                   marker='', markersize=8, color=colors[0], 
                   markeredgecolor=colors[0], alpha=.5)
        hr_ax.plot(2, df_sdt.hit_rate[df_sdt.cue == 0.75].iloc[index],
                   marker='', markersize=8, color=colors[1], 
                   markeredgecolor=colors[1], alpha=.5)
        hr_ax.plot([1, 2], [df_sdt.hit_rate[df_sdt.cue == 0.25].iloc[index],
                            df_sdt.hit_rate[df_sdt.cue == 0.75].iloc[index]],
                   marker='', markersize=0, color='gray', alpha=.25)

    hr_box = hr_ax.boxplot([df_sdt.hit_rate[df_sdt.cue == 0.25], 
                            df_sdt.hit_rate[df_sdt.cue == 0.75]], 
                            patch_artist=True)

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

    # Plot fa rate
    farate_ax = fig.add_subplot(gs[5, 0])
 
    for index in range(len(df_sdt.fa_rate[df_sdt.cue == 0.25])):
        farate_ax.plot(1, df_sdt.fa_rate[df_sdt.cue == 0.25].iloc[index],
                   marker='', markersize=8, color=colors[0], 
                   markeredgecolor=colors[0], alpha=.5)
        farate_ax.plot(2, df_sdt.fa_rate[df_sdt.cue == 0.75].iloc[index],
                   marker='', markersize=8, color=colors[1], 
                   markeredgecolor=colors[1], alpha=.5)
        farate_ax.plot([1, 2], [df_sdt.fa_rate[df_sdt.cue == 0.25].iloc[index],
                            df_sdt.fa_rate[df_sdt.cue == 0.75].iloc[index]],
                   marker='', markersize=0, color='gray', alpha=.25)

    fa_box = farate_ax.boxplot([df_sdt.fa_rate[df_sdt.cue == 0.25], 
                            df_sdt.fa_rate[df_sdt.cue == 0.75]], 
                            patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(fa_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(fa_box['medians'], medcolor):
        patch.set_color(color)

    farate_ax.set_ylabel('fa rate', fontname="Arial", fontsize=14)
    farate_ax.set_yticklabels(['0', '0.5', '1.0'], fontname="Arial", fontsize=12)
    farate_ax.text(1.3, 1, '***', verticalalignment='center', fontname='Arial', 
               fontsize='18')

    # Plot dprime
    dprime_ax = fig.add_subplot(gs[4:, 1])

    # Plot individual data points 
    for index in range(len(df_sdt.dprime[df_sdt.cue == 0.25])):
        dprime_ax.plot(1, df_sdt.dprime[df_sdt.cue == 0.25].iloc[index],
                   marker='', markersize=8, color=colors[0], 
                   markeredgecolor=colors[0], alpha=.5)
        dprime_ax.plot(2, df_sdt.dprime[df_sdt.cue == 0.75].iloc[index],
                   marker='', markersize=8, color=colors[1], 
                   markeredgecolor=colors[1], alpha=.5)
        dprime_ax.plot([1, 2], [df_sdt.dprime[df_sdt.cue == 0.25].iloc[index],
                            df_sdt.dprime[df_sdt.cue == 0.75].iloc[index]],
                   marker='', markersize=0, color='gray', alpha=.25)

    dprime_box = dprime_ax.boxplot([df_sdt.dprime[df_sdt.cue == 0.25], 
                            df_sdt.dprime[df_sdt.cue == 0.75]], patch_artist=True)

    for patch, color in zip(dprime_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(dprime_box['medians'], medcolor):
        patch.set_color(color)

    dprime_ax .set_ylabel('dprime', fontname="Arial", fontsize=14)
    dprime_ax.text(1.4, 3, 'ns', verticalalignment='center', fontname='Arial',
                   fontsize='13')
    dprime_ax.set_ylim(0, 3.0)
    dprime_ax.set_yticks([0, 1.5, 3.0])
    dprime_ax.set_yticklabels(['0', '1.5', '3.0'], fontname="Arial", 
                              fontsize=12)
    
    # Plot criterion
    crit_ax = fig.add_subplot(gs[4:, 2])

    # Plot individual data points 
    for index in range(len(df_sdt.criterion[df_sdt.cue == 0.25])):
        crit_ax.plot(1, df_sdt.criterion[df_sdt.cue == 0.25].iloc[index],
                   marker='', markersize=8, color=colors[0], 
                   markeredgecolor=colors[0], alpha=.5)
        crit_ax.plot(2, df_sdt.criterion[df_sdt.cue == 0.75].iloc[index],
                   marker='', markersize=8, color=colors[1], 
                   markeredgecolor=colors[1], alpha=.5)
        crit_ax.plot([1, 2], [df_sdt.criterion[df_sdt.cue == 0.25].iloc[index],
                            df_sdt.criterion[df_sdt.cue == 0.75].iloc[index]],
                   marker='', markersize=0, color='gray', alpha=.25)

    crit_box = crit_ax.boxplot([df_sdt.criterion[df_sdt.cue == 0.25], 
                            df_sdt.criterion[df_sdt.cue == 0.75]], 
                            patch_artist=True)
                        
    for patch, color in zip(crit_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(crit_box['medians'], medcolor):
        patch.set_color(color)

    crit_ax .set_ylabel('c', fontname="Arial", fontsize=14)
    crit_ax.text(1.4, 1.5, '***', verticalalignment='center', fontname='Arial',
                   fontsize='13')
    crit_ax.set_ylim(-0.5, 1.5)
    crit_ax.set_yticks([-0.5, 0.5, 1.5])
    crit_ax.set_yticklabels(['-0.5', '0.5', '1.5'], fontname="Arial", 
                              fontsize=12)
    # Plot confidence
    conf_ax = fig.add_subplot(gs[4:, 3])

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
        
    conf_ax .set_ylabel('high confidence', fontname="Arial", fontsize=14)
    conf_box = conf_ax.boxplot([conf_con[0].iloc[:, 1], conf_con[1].iloc[:, 1]],
                               patch_artist=True)
    
    conf_ax.text(1.4, 1.0, '***', verticalalignment='center', fontname='Arial',
                   fontsize='13')

    # Set the face color and alpha for the boxes in the plot
    colors = ['white', 'black']
    for patch, color in zip(conf_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    medcolor = ['black', 'white']
    for patch, color in zip(conf_box['medians'], medcolor):
        patch.set_color(color)

    for plots in [hr_ax, farate_ax, conf_ax]:
        plots.set_ylim(0, 1)
        plots.set_yticks([0, 0.5, 1.0])

    for plots in [hr_ax, farate_ax, dprime_ax, crit_ax, conf_ax]:
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

    if exclude_high_fa is True:
        fig.savefig(f'{savepath_fig1}figure1_exclhighfa.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    else:
        fig.savefig(f'{savepath_fig1}figure1.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    plt.show()


def calc_stats():

    """ Calculate statistics and effect sizes for the behavioral data."""

    conditions,_ = prepare_for_plotting(exclude_high_fa=True)

    # only for dprime, crit, hitrate, farate and confidence congruency
    df_sdt = conditions[0]
    conf = conditions[1]

    cond_list = ['criterion', 'hit_rate', 'fa_rate', 'dprime']
    for cond in cond_list:
        ci_lower, ci_upper = bootstrap_ci_effect_size_wilcoxon(x1=df_sdt[cond][df_sdt.cue == 0.25].reset_index(drop=True), 
                                                            x2=df_sdt[cond][df_sdt.cue == 0.75].reset_index(drop=True))

        ci_lower, ci_upper = bootstrap_ci_effect_size_wilcoxon(conf[0].reset_index(drop=True).drop('ID', axis=1).iloc[:, 0]
                                                                    ,conf[1].reset_index(drop=True).drop('ID', axis=1).iloc[:, 0])

def effect_wilcoxon(x1, x2):
    """
    Calculate effect size for the Wilcoxon signed-rank test (paired samples).

    Parameters:
    - x1: numpy array or list, the first sample
    - x2: numpy array or list, the second sample

    Returns:
    - r: float, rank biserial correlation coefficient
    - statistic: float, test statistic from the Wilcoxon signed-rank test
    - p_value: float, p-value from the Wilcoxon signed-rank test
    """

    if len(x1) != len(x2):
        raise ValueError("The two samples must have the same length for paired analysis.")

    statistic, p_value = stats.wilcoxon(x1, x2)

    # effect size rank biserial

    n = len(x1)

    r = 1 - (2 * statistic) / (n * (n + 1))

    output = [r, statistic, p_value]

    return output


def bootstrap_ci_effect_size_wilcoxon(x1, x2, n_iterations=1000, alpha=0.95):
    
    """
    Calculate the confidence interval for rank biserial as an effect size for 
    the Wilcoxon signed-rank test (paired samples).
    
    Parameters:
    - x1: numpy array or list, the first sample
    - x2: numpy array or list, the second sample
    - n_iterations: int, the number of bootstrap iterations
    - alpha: float, the confidence level
    
    Returns:
    - lower_percentile: float, the lower percentile of the confidence interval
    - upper_percentile: float, the upper percentile of the confidence interval
    """

    np.random.seed(0)  # Set a random seed for reproducibility
    n = len(x1)
    effect_sizes = []

    out = effect_wilcoxon(x1, x2)
    r, statistic, p_value = out

    # Print the result with description
    print("r (Effect Size):", r)
    print("Test Statistic:", statistic)
    print("p-value:", p_value)

    for _ in range(n_iterations):
        # Perform random sampling with replacement
        indices = np.random.randint(0, n, size=n)
        x1_sample = x1[indices]
        x2_sample = x2[indices]
        effect_size = effect_wilcoxon(x1_sample, x2_sample)
        effect_sizes.append(effect_size[0])

    lower_percentile = (1 - alpha) / 2
    upper_percentile = 1 - lower_percentile
    ci_lower = np.percentile(effect_sizes, lower_percentile * 100)
    ci_upper = np.percentile(effect_sizes, upper_percentile * 100)

    print("Effect size (r) 95% CI:", (ci_lower, ci_upper))

    return ci_lower, ci_upper


def supplementary_plots(savepath_fig1=Path('D:/expecon_ms/figs/manuscript_figures/Figure1/'),
                        exclude_high_fa=False):

    # set colors for both conditions
    blue = '#2a95ffff'  # 0.25 cue
    red = '#ff2a2aff'

    colors = [blue, red]
    medcolor = ['black', 'black']

    # load data
    conditions, exclude_high_fa = prepare_for_plotting(exclude_high_fa=False)

    # unpack data
    df_sdt, conf_con, conf_cue, acc_cue, conf_con_yes, conf_con_no, rt_con, \
    rt_con_yes, rt_con_incorrect = conditions

    # save accuracy and confidence for both conditions
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

    acc_box = plt.boxplot([acc_cue[0].iloc[:, 1], acc_cue[1].iloc[:, 1]], 
                         patch_artist=True)
    
    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(acc_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(acc_box['medians'], medcolor):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], ['0.25', '0.75'])    
    plt.xlabel('P (Stimulus)', fontname="Arial", fontsize=14)
    plt.ylabel('accuracy', fontname="Arial", fontsize=14)

    if exclude_high_fa is True:
        plt.savefig(f'{savepath_fig1}acc_cue_exclhighfa.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    else:
        plt.savefig(f'{savepath_fig1}acc_cue.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    plt.show()

    # is confidence higher for correct than incorrect responses?
    for index in range(len(conf_cue[0])):
        plt.plot(1, conf_cue[0].iloc[index, 1],
                 marker='', markersize=8, color=colors[0],
                 markeredgecolor=colors[0], alpha=.5)
        plt.plot(2, conf_cue[1].iloc[index, 1],
                 marker='', markersize=8, color=colors[1],
                 markeredgecolor=colors[1], alpha=.5)
        plt.plot([1, 2], [conf_cue[0].iloc[index, 1],
                 conf_cue[1].iloc[index, 1]],
                 marker='', markersize=0, color='gray', alpha=.25)

    conf_box = plt.boxplot([conf_cue[0].iloc[:, 1], conf_cue[1].iloc[:, 1]],
                            patch_artist=True)
    
    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(conf_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(conf_box['medians'], medcolor):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], ['0.25', '0.75'])    
    plt.xlabel('P (Stimulus)', fontname="Arial", fontsize=14)
    plt.ylabel('accuracy', fontname="Arial", fontsize=14)

    plt.ylabel('confidence', fontname="Arial", fontsize=14)

    if exclude_high_fa is True:
        plt.savefig(f'{savepath_fig1}conf_cue_exclhighfa.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    else:
        plt.savefig(f'{savepath_fig1}conf_cue.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    plt.show()

    # congruency on confidence for yes and no responses
    colors = ['white', 'black']

    # Plot individual data points 
    for index in range(len(conf_con_yes[0])):
        plt.plot(1, conf_con_yes[0].iloc[index, 1],
                 marker='', markersize=8, color=colors[0],
                 markeredgecolor=colors[0], alpha=.5)
        plt.plot(2, conf_con_yes[1].iloc[index, 1],
                 marker='', markersize=8, color=colors[1],
                 markeredgecolor=colors[1], alpha=.5)
        plt.plot([1, 2], [conf_con_yes[0].iloc[index, 1],
                 conf_con_yes[1].iloc[index, 1]],
                 marker='', markersize=0, color='gray', alpha=.25)

    conf_con_yes_box = plt.boxplot([conf_con_yes[0].iloc[:, 1], 
                                    conf_con_yes[1].iloc[:, 1]], 
                                    patch_artist=True)
    
    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(conf_con_yes_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(conf_con_yes_box['medians'], medcolor):
        patch.set_color(color)

        # Set x-axis tick labels
    plt.xticks([1, 2], ['Congruent', 'Incongruent']) 
    plt.xlabel('Yes responses', fontname="Arial", fontsize=14)
    plt.ylabel('Confidence', fontname="Arial", fontsize=14)

    if exclude_high_fa is True:
        plt.savefig(f'{savepath_fig1}conf_con_yes_exclhighfa.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    else:
        plt.savefig(f'{savepath_fig1}conf_con_yes.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    plt.show()

    # congruency on confidence for no responses
    for index in range(len(conf_con_no[0])):
        plt.plot(1, conf_con_no[0].iloc[index, 1],
                 marker='', markersize=8, color=colors[0],
                 markeredgecolor=colors[0], alpha=.5)
        plt.plot(2, conf_con_no[1].iloc[index, 1],
                 marker='', markersize=8, color=colors[1],
                 markeredgecolor=colors[1], alpha=.5)
        plt.plot([1, 2], [conf_con_no[0].iloc[index, 1],
                 conf_con_no[1].iloc[index, 1]],
                 marker='', markersize=0, color='gray', alpha=.25)

    conf_con_no_box = plt.boxplot([conf_con_no[0].iloc[:, 1],
                                   conf_con_no[1].iloc[:, 1]],
                                   patch_artist=True)
    
    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(conf_con_no_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(conf_con_no_box['medians'], medcolor):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], ['Congruent', 'Incongruent']) 
    plt.xlabel('No responses', fontname="Arial", fontsize=14)
    plt.ylabel('Mean confidence', fontname="Arial", fontsize=14)

    if exclude_high_fa is True:
        plt.savefig(f'{savepath_fig1}conf_con_no_exclhighfa.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    else:
        plt.savefig(f'{savepath_fig1}conf_con_no.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    plt.show()

    # Reaction times for stimulus congruent trials (correct only)
    for index in range(len(rt_con[0])):
        plt.plot(1, rt_con[0].iloc[index, 1],
                 marker='', markersize=8, color=colors[0],
                 markeredgecolor=colors[0], alpha=.5)
        plt.plot(2, rt_con[1].iloc[index, 1],
                 marker='', markersize=8, color=colors[1],
                 markeredgecolor=colors[1], alpha=.5)
        plt.plot([1, 2], [rt_con[0].iloc[index, 1],
                 rt_con[1].iloc[index, 1]],
                 marker='', markersize=0, color='gray', alpha=.25)

    rt_con_box = plt.boxplot([rt_con[0].iloc[:, 1], 
                                    rt_con[1].iloc[:, 1]], 
                                    patch_artist=True)
    
    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(rt_con_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(rt_con_box['medians'], medcolor):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], ['Congruent', 'Incongruent']) 
    plt.xlabel('Correct trials only', fontname="Arial", fontsize=14)
    plt.ylabel('Mean response time', fontname="Arial", fontsize=14)

    if exclude_high_fa is True:
        plt.savefig(f'{savepath_fig1}rt_con_exclhighfa.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    else:
        plt.savefig(f'{savepath_fig1}rt_con.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    plt.show()

    # Incorrect trials only
    for index in range(len(rt_con_incorrect[0])):
        plt.plot(1, rt_con_incorrect[0].iloc[index, 1],
                 marker='', markersize=8, color=colors[0],
                 markeredgecolor=colors[0], alpha=.5)
        plt.plot(2, rt_con_incorrect[1].iloc[index, 1],
                 marker='', markersize=8, color=colors[1],
                 markeredgecolor=colors[1], alpha=.5)
        plt.plot([1, 2], [rt_con_incorrect[0].iloc[index, 1],
                 rt_con_incorrect[1].iloc[index, 1]],
                 marker='', markersize=0, color='gray', alpha=.25)

    rt_con_in_box = plt.boxplot([rt_con_incorrect[0].iloc[:, 1], 
                                    rt_con_incorrect[1].iloc[:, 1]], 
                                    patch_artist=True)
    
    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(rt_con_in_box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(rt_con_in_box['medians'], medcolor):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], ['Congruent', 'Incongruent']) 
    plt.xlabel('Incorrect trials only', fontname="Arial", fontsize=14)
    plt.ylabel('Mean response time', fontname="Arial", fontsize=14)

    if exclude_high_fa is True:
        plt.savefig(f'{savepath_fig1}rt_con_incorrect_exclhighfa.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    else:
        plt.savefig(f'{savepath_fig1}rt_con_incorrect.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    plt.show()

    # Plot relationship between dprime and criterion
    x = df_sdt.criterion[df_sdt.cue == 0.75]
    y = df_sdt.dprime[df_sdt.cue == 0.75]

    reg = sns.regplot(x, y)

    # Fit a linear regression model to calculate the p-value
    model = sm.OLS(x, sm.add_constant(y))
    results = model.fit()
    p_value = results.pvalues[1]

    # Retrieve the regression line equation and round the coefficients
    slope, intercept = reg.get_lines()[0].get_data()
    slope = round(slope[1], 2)
    intercept = round(intercept[1], 2)

    # Add the regression line equation and p-value to the plot with Arial font and size 14
    equation = f'y = {slope}x + {intercept}'
    p_value_text = f'p-value: {p_value:.4f}'

    font = {'family': 'Arial', 'size': 14}

    plt.annotate(equation, xy=(0.05, 0.9), xycoords='axes fraction',
                 fontproperties=font)
    plt.annotate(p_value_text, xy=(0.05, 0.8), xycoords='axes fraction',
                 fontproperties=font)

    plt.xlabel('dprime 0.75', fontname="Arial", fontsize=14)
    plt.ylabel('c 0.75', fontname="Arial", fontsize=14)

    if exclude_high_fa is True:
        plt.savefig(f'{savepath_fig1}dprime_c_exclhighfa.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
    else:
        plt.savefig(f'{savepath_fig1}dprime_c.svg', dpi=300,
                    bbox_inches='tight',
                    format='svg')
 