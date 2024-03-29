#!/usr/bin/python3
"""
The script contains functions to correlate sdt data with questionnaire data.

Moreover, it contains functions to calculate the difference between the optimal criterion and the mean criterion.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023
"""

# %% Import
import math
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

from expecon_ms.configs import PROJECT_ROOT, params, paths

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/behav/corr_sdt.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# Set Arial as the default font
plt.rcParams.update({
    "font.size": params.plot.font.size,
    "font.family": params.plot.font.family,
    "font.sans-serif": params.plot.font.sans_serif,
})

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def correlate_sdt_with_questionnaire(expecon: int):
    """
    Correlate the criterion change with the intolerance of uncertainty questionnaire score and plots a regression plot.

    Args:
    ----
    expecon: int: expecon = 1: analyze expecon 1 behavioral data | expecon = 2: analyze expecon 2 behavioral data

    """
    # Set up data path
    behavior_path = Path(paths.data.behavior if expecon == 1 else paths.expecon2.behavior)

    data = pd.read_csv(behavior_path / "prepro_behav_data.csv")

    # load questionnaire data (uncertainty of intolerance)
    q_data = pd.read_csv(paths.questionnaire.q_clean)

    df_sdt = calculate_sdt_dataframe(
        df=data, signal_col="isyes", response_col="sayyes", subject_col="ID", condition_col="cue"
    )

    # calculate criterion
    high_c = list(df_sdt.criterion[df_sdt.cue == params.high_p])
    low_c = list(df_sdt.criterion[df_sdt.cue == params.low_p])

    # low - high stimulus probability
    diff_c = [x - y for x, y in zip(low_c, high_c)]

    # analyze only rows from participants did complete the questionnaire
    filtered_diff_c = np.array(diff_c)[q_data["iu_sum"] > 0]
    filtered_iu_sum = q_data["iu_sum"][q_data["iu_sum"] > 0]

    df_q = pd.DataFrame({"y": filtered_diff_c, "X": filtered_iu_sum})

    # Select columns representing questionnaire score and criterion change
    x = df_q["X"]
    y = df_q["y"]

    # Drop missing values if necessary
    df_q = df_q.dropna(subset=["X", "y"])

    # Fit the linear regression model
    x = sm.add_constant(x)  # Add constant term for intercept
    regression_model = sm.OLS(y, x)
    regression_results = regression_model.fit()

    # Extract slope, intercept, and p-value
    slope = regression_results.params[1]
    regression_results.params[0]
    print(regression_results.summary())
    p_value = regression_results.pvalues[1]

    # Make predictions
    regression_results.predict(x)

    # Set black or gray colors for the plot
    data_color = "black"
    text_color = "black"

    # Plot the regression line using seaborn regplot
    sns.regplot(data=df_q, x="X", y="y", color=data_color)
    plt.xlabel(xlabel="Intolerance of Uncertainty score (zscored)", color=text_color)
    plt.ylabel(ylabel="Criterion Change", color=text_color)
    plt.annotate(
        text=f"Slope: {slope:.2f}\n p-value: {p_value:.2f}",
        xy=(0.05, 0.85),
        xycoords="axes fraction",
        color=text_color,
    )

    for fm in ["png", "svg"]:
        plt.savefig(Path(paths.figures.manuscript.figure2) / f"linreg_c_q.{fm}", dpi=300)
    plt.show()


def diff_from_optimal_criterion(expecon: int):
    """
    Calculate the difference between the optimal criterion & the criterion for each participant and each condition.

    Args:
    ----
    expecon: int: expecon = 1: analyze expecon 1 behavioral data | expecon = 2: analyze expecon 2 behavioral data

    """
    # Set up data path
    behavior_path = Path(paths.data.behavior if expecon == 1 else paths.expecon2.behavior)

    data = pd.read_csv(behavior_path / "prepro_behav_data.csv")

    # calculate SDT parameters
    df_sdt = calculate_sdt_dataframe(
        df=data, signal_col="isyes", response_col="sayyes", subject_col="ID", condition_col="cue"
    )

    c_low = calculate_optimal_c(params.low_p, params.high_p, 1, 1)
    print("Optimal criterion (c) for low stimulus probability:", c_low)

    c_high = calculate_optimal_c(params.high_p, params.low_p, 1, 1)
    print("Optimal criterion (c) for high stimulus probability:", c_high)

    sub_op_low = [c - c_low for c in df_sdt.criterion[df_sdt.cue == params.low_p]]
    mean_low = np.mean(sub_op_low)
    std_low = np.std(sub_op_low)
    print(f"mean difference from optimal criterion for low stimulus probability: {mean_low}")
    print(f"standard deviation for optimal criterion for low stimulus probability: {std_low}")

    sub_op_high = [c - c_high for c in df_sdt.criterion[df_sdt.cue == params.high_p]]
    mean_high = np.mean(sub_op_high)
    std_high = np.std(sub_op_high)
    print(f"mean difference from optimal criterion for high stimulus probability: {mean_high}")
    print(f"standard deviation for optimal criterion for high stimulus probability: {std_high}")


# Helper functions #############################


def calculate_sdt_dataframe(
    df: pd.DataFrame, signal_col: str, response_col: str, subject_col: str, condition_col: str
) -> pd.DataFrame:
    """
    Calculate SDT measures (d' and criterion) for each participant and each condition based on a dataframe.

    Args:
    ----
    df: Pandas dataframe containing the data
    signal_col: Name of the column indicating signal presence (e.g., 'signal')
    response_col: Name of the column indicating participant response (e.g., 'response')
    subject_col: Name of the column indicating participant ID (e.g., 'subject_id')
    condition_col: Name of the column indicating condition (e.g., 'condition')

    Returns:
    -------
    Pandas dataframe containing the calculated SDT measures (d' and criterion) for each participant and condition.

    """
    # Apply Hautus correction and calculate SDT measures for each participant and condition
    results = []
    subjects = df[subject_col].unique()
    conditions = df[condition_col].unique()

    for subject in subjects:
        for condition in conditions:
            subset = df[(df[subject_col] == subject) & (df[condition_col] == condition)]

            detect_hits = subset[(subset[signal_col] is True) & (subset[response_col] is True)].shape[0]
            detect_misses = subset[(subset[signal_col] is True) & (subset[response_col] is False)].shape[0]
            false_alarms = subset[(subset[signal_col] is False) & (subset[response_col] is True)].shape[0]
            correct_rejections = subset[(subset[signal_col] is False) & (subset[response_col] is False)].shape[0]

            # log linear correction (Hautus, 1995)
            hit_rate = (detect_hits + 0.5) / (detect_hits + detect_misses + 1)
            false_alarm_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)

            d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate)
            criterion = -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(false_alarm_rate))

            results.append((subject, condition, hit_rate, false_alarm_rate, d_prime, criterion))

    # Create a new dataframe with the results
    return pd.DataFrame(results, columns=[subject_col, condition_col, "hit_rate", "fa_rate", "dprime", "criterion"])


def calculate_optimal_c(base_rate_signal, base_rate_catch, cost_hit, cost_false_alarm) -> float:
    """
    Calculate the optimal criterion (c).

    Do this for a given base rate of signal and catch trials, & cost of hit and false alarm.
    """
    llr = math.log((base_rate_signal / base_rate_catch) * (cost_hit / cost_false_alarm))
    return -0.5 * llr  # c


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
