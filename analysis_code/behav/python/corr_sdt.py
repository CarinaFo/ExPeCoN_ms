# This script contains functions to correlate sdt data with questionnaire data 
# and to calculate the difference between the optimal criterion and the mean criterion 

# Author: Carina Forster
# email: forster@cbs.mpg.de

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pathlib import Path
import scipy.stats as stats
import subprocess

# Specify the file path for which you want the last commit date
file_path = Path("D:/expecon_ms/analysis_code/behav/python/corr_sdt.py")

last_commit_date = subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", file_path]).decode("utf-8").strip()
print("Last Commit Date for", file_path, ":", last_commit_date)

# Set Arial as the default font
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

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


def correlate_sdt_with_questionnaire(expecon=1):

    """ This function correlates the criterion change with the 
    intolerance of uncertainty questionnaire score and plots a regression plot.
    expecon = 1: analyze expecon 1 behavioral data
    expecon = 2: analyze expecon 2 behavioral data
    """

    if expecon == 1:
        # Set up data path
        behavpath = Path('D:/expecon_ms/data/behav/behav_df')
    else: 
        # analyze expecon 2 behavioral data
        behavpath = Path('D:/expecon_2/behav')
    
    data = pd.read_csv(f'{behavpath}{Path("/")}prepro_behav_data.csv')
    
    # load questionarre data (uncertainty of intolerance)
    q_path = Path('D:/expecon_ms/questionnaire/q_clean.csv')

    q_data = pd.read_csv(q_path)

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