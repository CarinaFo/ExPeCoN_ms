#!/usr/bin/python3
"""
The script contains functions to analyze and plot the behavioral data for the ExPeCoN study.

The ExPeCoN study investigates stimulus probabilities and the influence on perception and confidence in a
near-threshold somatosensory detection task in two paradigms that vary in their probability environment:

dataset 1 : stable environment, 144 trials in 5 blocks, 43 participants,
           stimulus probability is cued before blocks of 12 trials.
dataset 2: variable environment, 120 trials in 5 blocks, 40 participants,
           stimulus probability is cued before each trial.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023
"""

# %% Import
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import gridspec
from scipy import stats

from expecon_ms.configs import PROJECT_ROOT, params, paths

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/behav/figure1.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# Set Arial as the default font
plt.rcParams.update({
    "font.size": params.plot.font.size,
    "font.family": "Arial",  # params.plot.font.family,
    # "font.sans-serif": params.plot.font.sans_serif
})

# Set save paths (figure 1 is study 1 (block design) and figure 2 is study2 (single trial design))
Path(paths.figures.manuscript.figure1).mkdir(parents=True, exist_ok=True)
Path(paths.figures.manuscript.figure2).mkdir(parents=True, exist_ok=True)

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def prepro_behavioral_data(expecon: int):
    """
    Preprocess the behavioral data.

    Remove trials with no response or superfast responses
    and add additional columns to the dataframe (congruency etc.)

    Args:
    ----
        expecon: int: which dataset: expecon 1 or expecon 2

    Returns:
    -------
    data: pandas dataframe containing the preprocessed behavioral data

    """
    if expecon == 1:
        # Load the behavioral data from the specified path
        data = pd.read_csv(Path(paths.data.behavior) / "behav_cleaned_for_eeg_expecon1.csv")

        # Clean up the dataframe by dropping unnecessary columns
        columns_to_drop = [col for col in data.columns if "Unnamed" in col]

        data = data.drop(columns_to_drop, axis=1)

        # Change the block number for participant 7's block 3
        data.loc[(144 * 2) : (144 * 3), "block"] = 4

    else:
        data = pd.read_csv(Path(paths.data.behavior) / "behav_cleaned_for_eeg_expecon2.csv")

        # ID to exclude (missing stimulation in block 1 and 2)
        id_to_exclude = 13

        # Excluding the ID from the DataFrame
        data = data[data["ID"] != id_to_exclude]

        # rename columns
        data = data.rename(columns={"stim_type": "isyes"})  # stimulus (1 = signal)
        data = data.rename(columns={"resp1": "sayyes"})  # detection response (1 = Yes)
        data = data.rename(columns={"resp2": "conf"})  # confidence (binary)
        data = data.rename(columns={"resp1_t": "respt1"})  # detection rt
        data = data.rename(columns={"resp2_t": "respt2"})  # confidence rt

        data[["sayyes", "isyes", "cue", "conf", "respt1", "resp2_t"]] = data[
            ["sayyes", "isyes", "cue", "conf", "respt1", "respt2"]
        ].apply(pd.to_numeric, errors="coerce")

    # reset index
    data = data.reset_index()

    # add a column that indicates correct responses & congruency
    data["correct"] = data.sayyes == data.isyes
    # Add a 'congruency' column
    data["congruency"] = ((data.cue == params.low_p) & (data.sayyes == 0)) | (
        (data.cue == params.high_p) & (data.sayyes == 1)
    )
    # Add a 'congruency stimulus' column
    data["congruency_stim"] = ((data.cue == params.low_p) & (data.isyes == 0)) | (
        (data.cue == params.high_p) & (data.isyes == 1)
    )

    # add a column that combines the confidence ratings and the
    # detection response
    data["conf_resp"] = [
        4
        if data.loc[i, "sayyes"] == 1 and data.loc[i, "conf"] == 1
        else 3
        if data.loc[i, "sayyes"] == 0 and data.loc[i, "conf"] == 1
        else 2
        if data.loc[i, "sayyes"] == 1 and data.loc[i, "conf"] == 0
        else 1
        for i in range(len(data))
    ]

    # add lagged variables
    data["prevresp"] = data.groupby(["ID", "block"])["sayyes"].shift(1)
    data["prevconf"] = data.groupby(["ID", "block"])["conf"].shift(1)
    data["prevconf_resp"] = data.groupby(["ID", "block"])["conf_resp"].shift(1)
    data["prevcorrect"] = data.groupby(["ID", "block"])["correct"].shift(1)
    data["prevcue"] = data.groupby(["ID", "block"])["cue"].shift(1)
    data["prevrespt1"] = data.groupby(["ID", "block"])["respt1"].shift(1)
    data["prevrespt2"] = data.groupby(["ID", "block"])["respt2"].shift(1)
    data["previsyes"] = data.groupby(["ID", "block"])["isyes"].shift(1)

    # remove no response trials or super fast responses
    data = data.drop(data[data.respt1 == params.behavioral_cleaning.rt_max].index)
    data = data.drop(data[data.respt1 < params.behavioral_cleaning.rt_min].index)

    # save the preprocessed dataframe
    data.to_csv(Path(paths.data.behavior, f"behav_data_exclrts_{expecon!s}.csv"))

    return data


def exclude_data(expecon: int = 1):
    """
    Exclude experimental blocks from the data based on the exclusion criteria (hit rates, fa rates).

    Args:
    ----
    expecon: int: which dataset to use: expecon 1 or expecon 2

    Returns:
    -------
    data: Pandas dataframe containing the data

    """
    # Load data
    data = pd.read_csv(Path(paths.data.behavior, f"behav_data_exclrts_{expecon!s}.csv"))

    # Calculate hit rates by participant and cue condition
    signal = data[data.isyes == 1]
    hit_rate_per_subject = signal.groupby(["ID"])["sayyes"].mean()

    print(f"Mean hit rate: {np.mean(hit_rate_per_subject):.2f}")
    print(f"Standard deviation: {np.std(hit_rate_per_subject):.2f}")
    print(f"Minimum hit rate: {np.min(hit_rate_per_subject):.2f}")
    print(f"Maximum hit rate: {np.max(hit_rate_per_subject):.2f}")

    # Calculate hit rates by participant and block condition
    hit_rate_per_block = signal.groupby(["ID", "block"])["sayyes"].mean()

    # Filter the grouped object based on hit rate conditions
    hit_rate_abn = hit_rate_per_block[
        (hit_rate_per_block > params.behavioral_cleaning.hitrate_max)
        | (hit_rate_per_block < params.behavioral_cleaning.hitrate_min)
    ]
    print(
        f"Blocks with hit rates > {params.behavioral_cleaning.hitrate_max} or "
        f"< {params.behavioral_cleaning.hitrate_min}: ",
        len(hit_rate_abn),
    )

    # Extract the ID and block information from the filtered groups
    remove_hit_rates = hit_rate_abn.reset_index()

    # Calculate hit rates by participant and cue condition
    noise = data[data.isyes == 0]
    fa_rate_per_block = noise.groupby(["ID", "block"])["sayyes"].mean()

    # Filter the grouped object based on fa rate conditions
    fa_rate_abn = fa_rate_per_block[fa_rate_per_block > params.behavioral_cleaning.farate_max]
    print(f"Blocks with false alarm rates > {params.behavioral_cleaning.farate_max}:", len(fa_rate_abn))

    # Extract the ID and block information from the filtered groups
    remove_fa_rates = fa_rate_abn.reset_index()

    # Filter the grouped objects based on the conditions
    hit_fa = hit_rate_per_block[hit_rate_per_block - fa_rate_per_block < params.alpha]  # Difference < 5 %
    print("Blocks with hit rates < false alarm rates: ", len(hit_fa))

    # Extract the ID and block information from the filtered groups
    hit_vs_fa_rate = hit_fa.reset_index()

    # Concatenate the dataframes
    combined_df = pd.concat([remove_hit_rates, remove_fa_rates, hit_vs_fa_rate])

    # Remove duplicate rows based on 'ID' and 'block' columns
    unique_df = combined_df.drop_duplicates(subset=["ID", "block"])

    # Merge the big dataframe with unique_df to retain only the non-matching rows
    filtered_df = data.merge(unique_df, on=["ID", "block"], how="left", indicator=True, suffixes=("", "_y"))

    filtered_df = filtered_df[filtered_df["_merge"] == "left_only"]

    # Remove the '_merge' column
    data = filtered_df.drop("_merge", axis=1)

    data.to_csv(Path(paths.data.behavior, f"prepro_behav_data_{expecon!s}.csv"))

    return data


def calculate_mean_sdt_param_changes(expecon=1):
    """
    Calculate the mean change in hit rate, false alarm rate, dprime, and criterion between the cue conditions.

    Args:
    ----
    expecon: int: which dataset to use: expecon 1 or expecon 2

    Returns:
    -------
    None

    """
    # load cleaned dataframe
    df_study = exclude_data(expecon=expecon)

    # calculate hit rates, false alarm rates, d-prime, and criterion per participant and cue condition
    df_sdt = calculate_sdt_dataframe(df_study, "isyes", "sayyes", "ID", "cue")

    # calculate hit rate change between conditions per participant
    diff_hit = (
        df_sdt.hit_rate[df_sdt.cue == params.high_p].reset_index()
        - df_sdt.hit_rate[df_sdt.cue == params.low_p].reset_index()
    )

    # calculate fa rate change between conditions per participant
    diff_fa = (
        df_sdt.fa_rate[df_sdt.cue == params.high_p].reset_index()
        - df_sdt.fa_rate[df_sdt.cue == params.low_p].reset_index()
    )

    # calculate dprime change between conditions per participant
    diff_dprime = (
        df_sdt.dprime[df_sdt.cue == params.high_p].reset_index()
        - df_sdt.dprime[df_sdt.cue == params.low_p].reset_index()
    )

    # calculate criterion change between conditions per participant
    diff_crit = (
        df_sdt.criterion[df_sdt.cue == params.high_p].reset_index()
        - df_sdt.criterion[df_sdt.cue == params.low_p].reset_index()
    )

    # Filter for correct trials only
    correct_only = df_study[df_study.correct == 1]

    # Calculate mean confidence for each participant and congruency condition
    data_grouped = correct_only.groupby(["ID", "congruency"])["conf"].mean()
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()

    diff_congruency = con_condition[True] - incon_condition[False]

    print(diff_hit.mean())
    print(diff_fa.mean())
    print(diff_dprime.mean())
    print(diff_crit.mean())
    print(diff_congruency.mean())


def calculate_cohens_d_and_criterion_change(expecon: int = 1, 
                                            subject_col: str = 'ID', 
                                            cue_col: str = 'cue', 
                                            criterion_col: str = 'criterion', 
                                            high_cue_value: float = 0.75, 
                                            low_cue_value: float = 0.25,
                                            correct_trials_only: bool = True,
                                            sdt: bool = False):
    """
    Calculate Cohen's d for the difference in criterion between high and low cue conditions.
    Also prints the number of subjects with a negative criterion change.

    Parameters:
    df (pd.DataFrame): DataFrame containing subject data.
    subject_col (str): Column name for subjects.
    cue_col (str): Column name for cue conditions.
    criterion_col (str): Column name for decision criteria.
    high_cue_value (float): Value for high cue condition.
    low_cue_value (float): Value for low cue condition.

    Returns:
    float: Cohen's d value for the criterion difference.
    """
    # load cleaned dataframe
    df_raw = exclude_data(expecon=expecon)

    df = calculate_sdt_dataframe(df_raw, "isyes", "sayyes", "ID", "cue")

    # Get unique subjects
    subjects = df[subject_col].unique()
    
    # Initialize lists to store criteria
    high_criteria = []
    low_criteria = []
    
    # Iterate over each subject to get their criteria for high and low cues
    for subject in subjects:
        high_criterion = df[(df[subject_col] == subject) & (df[cue_col] == high_cue_value)][criterion_col]
        low_criterion = df[(df[subject_col] == subject) & (df[cue_col] == low_cue_value)][criterion_col]
        
        if not high_criterion.empty and not low_criterion.empty:
            high_criteria.append(high_criterion.values[0])
            low_criteria.append(low_criterion.values[0])

    # Calculate differences in criterion
    differences = np.array(high_criteria) - np.array(low_criteria)

    # Count the number of subjects with negative criterion change
    negative_criterion_change_count = np.sum(differences < 0)
    print(f"Number of subjects with higher {criterion_col} in low probability condition: {negative_criterion_change_count}/{len(subjects)}")

    # Calculate Cohen's d
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    n = len(differences)

    if std_diff == 0:
        return float('nan')  # Avoid division by zero if standard deviation is zero

    cohens_d = mean_diff / std_diff

    print(f"Cohen's d {criterion_col}: {cohens_d}")
    print(f'mean difference in {criterion_col}: {mean_diff}')

    # calculate the effect size for the difference in mean confidence between congruent and incongruent conditions
    if sdt:
        return cohens_d, mean_diff
    else:
        # Filter for correct trials only
        df = df_raw[df_raw.correct == 1]
        # calculate mean confidence for each participant and congruency condition
        data_grouped = df.groupby([subject_col, 'congruency'])['conf'].mean()
        con_condition = data_grouped.unstack()[True].reset_index()
        incon_condition = data_grouped.unstack()[False].reset_index()

        diff_congruency = con_condition[True] - incon_condition[False]
     
        # calculate Cohens d for the difference in confidence between congruent and incongruent conditions
        mean_diff = diff_congruency.mean()
        std_diff = diff_congruency.std(ddof=1)
        n = len(diff_congruency)

        if std_diff == 0:
            return float('nan')
        
        cohens_d = mean_diff / std_diff

        print(f"Cohen's d congruency: {cohens_d}")
        print(f'mean difference in congruency: {mean_diff}')

        # print for how many participants the confidence is higher in the congruent condition
        print(f"Number of subjects with higher confidence in congruent condition: {np.sum(diff_congruency > 0)}/{len(subjects)}")
    
    return cohens_d, mean_diff


def calculate_response_bias(expecon: int = 1, subject_col: str = 'ID', cue_col: str = 'cue', response_col: str = 'sayyes'):
    """
    Calculate the response bias for each subject in the dataset.

    Parameters:
    df (pd.DataFrame): DataFrame containing subject data.
    subject_col (str): Column name for subjects.
    cue_col (str): Column name for cue conditions.
    response_col (str): Column name for responses.

    Returns:
    list: List of response biases for each subject.
    """
    # load cleaned dataframe
    df = exclude_data(expecon=expecon)

    # add column with response repetition
    df['response_repeat'] = df['prevresp'] == df[response_col]

    response_biases = df.groupby([subject_col])['response_repeat'].mean().reset_index()

    # how many participants tend to repeat their previous response
    print(f"Number of subjects with a response bias < .05: {np.sum(np.array(response_biases) < 0.5)}")
    # is the response bias signficantly different from 0.5
    t, p = stats.ttest_1samp(response_biases, 0.5)
    print(f"t-value: {t}")
    print(f"p-value: {p}")

    print(f"Mean response bias: {np.mean(response_biases)}")
    
    # now we want to check if their is a difference in bias between low and high previous confidence
    response_bias = df.groupby([subject_col, 'prevconf'])['response_repeat'].mean().reset_index()

    # check if the response bias is significantly different between the previous confidence conditions
    t, p = stats.ttest_rel(
        response_bias[response_bias['prevconf'] == 0].response_repeat,
        response_bias[response_bias['prevconf'] == 1].response_repeat,
    )

    print(f"t-value: {t}")
    print(f"p-value: {p}")

    # print mean response bias for each previous confidence condition
    print(response_bias.groupby('prevconf')['response_repeat'].mean())

    # is the bias significantly different from 0.5 for confident and unconfident trials
    t, p = stats.ttest_1samp(response_bias[response_bias['prevconf'] == 0].response_repeat, 0.5)
    print(f"t-value: {t}")
    print(f"p-value: {p}")

    t, p = stats.ttest_1samp(response_bias[response_bias['prevconf'] == 1].response_repeat, 0.5)
    print(f"t-value: {t}")
    print(f"p-value: {p}")

    # calculate only for previous confident trials

    # calculate previous response bias for each subject and each cue condition
    response_bias = df.groupby([subject_col, cue_col, 'prevconf'])['response_repeat'].mean().reset_index()

    # check if the response bias is significantly different between the cue conditions for previous confident trials
    t, p = stats.ttest_rel(
        response_bias[(response_bias['prevconf'] == 1) & (response_bias[cue_col] == 0.25)].response_repeat,
        response_bias[(response_bias['prevconf'] == 1) & (response_bias[cue_col] == 0.75)].response_repeat,
    )

    print(f"p-value confident: {p}")

    # same for previous unconfident trials
    t, p = stats.ttest_rel(
        response_bias[(response_bias['prevconf'] == 0) & (response_bias[cue_col] == 0.25)].response_repeat,
        response_bias[(response_bias['prevconf'] == 0) & (response_bias[cue_col] == 0.75)].response_repeat,
    )

    print(f"p-value unconfident: {p}")

    # print mean response bias for each cue condition and previous confidence condition
    print(response_bias.groupby([cue_col, 'prevconf'])['response_repeat'].mean())




def plot_mean_response_and_confidence(
    expecon: int,
    blue="#0571b0",
    red="#ca0020",
    no_col="#088281",
    yes_col="#d01c8b",
    savepath=paths.figures.manuscript.figure2_suppl,
):
    """
    Plot the mean detection response and mean confidence for each cue condition with a boxplot.

    Calculate test-statistics (Wilcoxon signed-rank test)
    for the mean detection response and confidence between the cue conditions.

    Args:
    ----
    blue: color for low cue condition
    red: color for high cue condition
    no_col: color for no response
    yes_col: color for yes response
    savepath: path to save the figure to
    expecon: int: which dataset to use: expecon 1 or expecon 2

    Returns:
    -------
    None

    """
    # load cleaned dataframe
    data, expecon = exclude_data(expecon=expecon)

    # Calculate mean response per ID and cue
    mean_resp_id_cue = data.groupby(["cue", "ID"])["sayyes"].mean().reset_index()

    # Calculate mean response per ID and cue
    mean_prevresp_id_cue = data.groupby(["cue", "ID"])["prevresp"].mean().reset_index()

    # Calculate mean confidence per ID and response
    mean_conf_id_resp = data.groupby(["sayyes", "ID"])["conf"].mean().reset_index()

    # Create boxplots
    # response distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="cue", y="sayyes", data=mean_resp_id_cue, palette=[blue, red])
    sns.stripplot(x="cue", y="sayyes", data=mean_resp_id_cue, color="black", size=4, jitter=True)
    plt.xlabel("stimulus probability")
    plt.ylabel("% yes responses")
    plt.savefig(Path(savepath, f"choice_cue_{expecon}.svg"))
    plt.savefig(Path(savepath, f"choice_cue_{expecon}.png"))
    plt.show()

    # Perform the Wilcoxon signed-rank test
    wilcoxon_statistic, p_value = stats.wilcoxon(
        mean_resp_id_cue[mean_resp_id_cue.cue == params.low_p].sayyes,
        mean_resp_id_cue[mean_resp_id_cue.cue == params.high_p].sayyes,
    )
    print(f"Wilcoxon statistic: {wilcoxon_statistic}")
    print(f"p-value: {p_value}")

    # previous response distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="cue", y="prevresp", data=mean_prevresp_id_cue, palette=[blue, red])
    sns.stripplot(x="cue", y="prevresp", data=mean_prevresp_id_cue, color="black", size=4, jitter=True)
    plt.xlabel("stimulus probability")
    plt.ylabel("% previous yes responses")
    plt.savefig(Path(savepath, f"prevchoice_cue_{expecon}.svg"))
    plt.savefig(Path(savepath, f"prevchoice_cue_{expecon}.png"))
    plt.show()

    # Perform the Wilcoxon signed-rank test
    wilcoxon_statistic, p_value = stats.wilcoxon(
        mean_prevresp_id_cue[mean_prevresp_id_cue.cue == params.low_p].prevresp,
        mean_prevresp_id_cue[mean_prevresp_id_cue.cue == params.high_p].prevresp,
    )
    print(f"Wilcoxon statistic: {wilcoxon_statistic}")
    print(f"p-value: {p_value}")

    # confidence distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="sayyes", y="conf", data=mean_conf_id_resp, palette=[no_col, yes_col])
    sns.stripplot(x="sayyes", y="conf", data=mean_conf_id_resp, color="black", size=4, jitter=True)
    plt.xlabel("detection response")
    plt.ylabel("% high confidence")
    plt.xticks(ticks=[0, 1], labels=["No", "Yes"])  # Set custom tick labels
    plt.savefig(Path(savepath, f"choice_conf_{expecon}.svg"))
    plt.savefig(Path(savepath, f"choice_conf_{expecon}.png"))
    plt.show()

    wilcoxon_statistic, p_value = stats.wilcoxon(
        mean_conf_id_resp[mean_conf_id_resp.sayyes == 1].conf, mean_conf_id_resp[mean_conf_id_resp.sayyes == 0].conf
    )
    print(f"Wilcoxon statistic: {wilcoxon_statistic}")
    print(f"p-value: {p_value}")

    return "saved figures"


def prepare_for_plotting(exclude_high_fa: bool, expecon: int):
    """
    Prepare the data for plotting.

    Args:
    ----
    exclude_high_fa: Boolean: indicating whether to exclude participants with high false alarm rates
    expecon: which dataset: int: expecon 1 or expecon 2

    Returns:
    -------
    data: Pandas dataframe containing the data

    """
    data = exclude_data(expecon=expecon)

    # calculate hit rates, false alarm rates, d-prime, and criterion per participant and cue condition
    # and per condition
    df_sdt = calculate_sdt_dataframe(data, "isyes", "sayyes", "ID", "cue")

    # create a boolean mask for participants with very high fa rates
    fa_rate_high_indices = np.where(
        df_sdt.fa_rate[df_sdt.cue == params.high_p] > params.behavioral_cleaning.farate_max
    )
    # Three participants with fa rates > 0.4
    print(f"Index of participants with high fa-rates: {fa_rate_high_indices}")

    if exclude_high_fa:
        # exclude participants with high fa-rates
        add = 7 if expecon == 1 else 1  # add to the indices to get the correct participant number
        indices = [f + add for f in fa_rate_high_indices]
        data = data[~data["ID"].isin(indices[0])]

    # calculate hit rates, false alarm rates, d-prime, and criterion per participant and cue condition
    df_sdt = calculate_sdt_dataframe(data, "isyes", "sayyes", "ID", "cue")

    # Filter for correct trials only
    correct_only = data[data.correct == 1]
    yes_responses = correct_only[correct_only.sayyes == 1]
    no_responses = correct_only[correct_only.sayyes == 0]

    # Calculate mean confidence for each participant and congruency condition
    data_grouped = correct_only.groupby(["ID", "congruency"])["conf"].mean()
    yes_grouped = yes_responses.groupby(["ID", "congruency"])["conf"].mean()
    no_grouped = no_responses.groupby(["ID", "congruency"])["conf"].mean()

    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    con_yes_condition = yes_grouped.unstack()[True].reset_index()
    incon_yes_condition = yes_grouped.unstack()[False].reset_index()
    con_no_condition = no_grouped.unstack()[True].reset_index()
    incon_no_condition = no_grouped.unstack()[False].reset_index()

    conf_con = [con_condition, incon_condition]
    conf_yes = [con_yes_condition, incon_yes_condition]
    conf_no = [con_no_condition, incon_no_condition]

    conditions = df_sdt, conf_con, conf_yes, conf_no

    return conditions, exclude_high_fa


def plot_figure1_grid(expecon: int, exclude_high_fa: bool):
    """
    Plot the figure 1 grid and the behavioral data for the EXPECON study.

    Args:
    ----
    expecon: int : which study to analyze
    exclude_high_fa: bool: whether to exclude participants with high false alarm rates

    Return:
    ------
    None

    """
    # set the save path
    savepath_fig1 = Path(paths.figures.manuscript.figure1) if expecon == 1 else Path(paths.figures.manuscript.figure2)

    # load data
    conditions, exclude_high_fa = prepare_for_plotting(exclude_high_fa=exclude_high_fa, expecon=expecon)

    # unpack data
    df_sdt, conf_con, _, _ = conditions  # _, _ == conf_yes, conf_no

    # set colors for both conditions
    blue = "#0571b0"  # params.low_p=0.25 cue
    red = "#ca0020"  # params.high_p=params.high_p cue

    colors = [blue, red]
    med_color = ["black", "black"]

    fig = plt.figure(figsize=(8, 10), tight_layout=True)  # the original working was 10,12
    gs = gridspec.GridSpec(nrows=6, ncols=4)

    schem_01_ax = fig.add_subplot(gs[0:2, 0:])
    schem_01_ax.set_yticks([])
    schem_01_ax.set_xticks([])

    schem_02_ax = fig.add_subplot(gs[2:4, 0:])
    schem_02_ax.set_yticks([])
    schem_02_ax.set_xticks([])

    hr_ax = fig.add_subplot(gs[4, 0])

    # Plot hit rate
    for index in range(len(df_sdt.hit_rate[df_sdt.cue == params.low_p])):
        hr_ax.plot(
            1,
            df_sdt.hit_rate[df_sdt.cue == params.low_p].iloc[index],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        hr_ax.plot(
            2,
            df_sdt.hit_rate[df_sdt.cue == params.high_p].iloc[index],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        hr_ax.plot(
            [1, 2],
            [
                df_sdt.hit_rate[df_sdt.cue == params.low_p].iloc[index],
                df_sdt.hit_rate[df_sdt.cue == params.high_p].iloc[index],
            ],
            marker="",
            markersize=0,
            color="gray",
            alpha=params.low_p,
        )

    hr_box = hr_ax.boxplot(
        [df_sdt.hit_rate[df_sdt.cue == params.low_p], df_sdt.hit_rate[df_sdt.cue == params.high_p]], patch_artist=True
    )

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(hr_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(params.high_p)

    # Set the color for the medians in the plot
    for patch, color in zip(hr_box["medians"], med_color):
        patch.set_color(color)

    hr_ax.set_ylabel("hit rate", fontname="Arial", fontsize=14)
    hr_ax.set_yticklabels(["0", "0.5", "1.0"], fontname="Arial", fontsize=12)
    hr_ax.text(1.3, 1, "***", verticalalignment="center", fontname="Arial", fontsize="18")

    # Plot fa rate
    fa_rate_ax = fig.add_subplot(gs[5, 0])

    for index in range(len(df_sdt.fa_rate[df_sdt.cue == params.low_p])):
        fa_rate_ax.plot(
            1,
            df_sdt.fa_rate[df_sdt.cue == params.low_p].iloc[index],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        fa_rate_ax.plot(
            2,
            df_sdt.fa_rate[df_sdt.cue == params.high_p].iloc[index],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        fa_rate_ax.plot(
            [1, 2],
            [
                df_sdt.fa_rate[df_sdt.cue == params.low_p].iloc[index],
                df_sdt.fa_rate[df_sdt.cue == params.high_p].iloc[index],
            ],
            marker="",
            markersize=0,
            color="gray",
            alpha=params.low_p,
        )

    fa_box = fa_rate_ax.boxplot(
        [df_sdt.fa_rate[df_sdt.cue == params.low_p], df_sdt.fa_rate[df_sdt.cue == params.high_p]], patch_artist=True
    )

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(fa_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(params.high_p)

    # Set the color for the medians in the plot
    for patch, color in zip(fa_box["medians"], med_color):
        patch.set_color(color)

    fa_rate_ax.set_ylabel("fa rate", fontname="Arial", fontsize=14)
    fa_rate_ax.set_yticklabels(["0", "0.5", "1.0"], fontname="Arial", fontsize=12)
    fa_rate_ax.text(1.3, 1, "***", verticalalignment="center", fontname="Arial", fontsize="18")

    # Plot dprime
    dprime_ax = fig.add_subplot(gs[4:, 1])

    # Plot individual data points
    for index in range(len(df_sdt.dprime[df_sdt.cue == params.low_p])):
        dprime_ax.plot(
            1,
            df_sdt.dprime[df_sdt.cue == params.low_p].iloc[index],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        dprime_ax.plot(
            2,
            df_sdt.dprime[df_sdt.cue == params.high_p].iloc[index],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        dprime_ax.plot(
            [1, 2],
            [
                df_sdt.dprime[df_sdt.cue == params.low_p].iloc[index],
                df_sdt.dprime[df_sdt.cue == params.high_p].iloc[index],
            ],
            marker="",
            markersize=0,
            color="gray",
            alpha=params.low_p,
        )

    dprime_box = dprime_ax.boxplot(
        [df_sdt.dprime[df_sdt.cue == params.low_p], df_sdt.dprime[df_sdt.cue == params.high_p]], patch_artist=True
    )

    for patch, color in zip(dprime_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(params.high_p)

    # Set the color for the medians in the plot
    for patch, color in zip(dprime_box["medians"], med_color):
        patch.set_color(color)

    dprime_ax.set_ylabel("dprime", fontname="Arial", fontsize=14)
    dprime_ax.text(1.4, 3, "n.s.", verticalalignment="center", fontname="Arial", fontsize="13")
    dprime_ax.set_ylim(0, 3.0)
    dprime_ax.set_yticks([0, 1.5, 3.0])
    dprime_ax.set_yticklabels(["0", "1.5", "3.0"], fontname="Arial", fontsize=12)

    # Plot criterion
    crit_ax = fig.add_subplot(gs[4:, 2])

    # Plot individual data points
    for index in range(len(df_sdt.criterion[df_sdt.cue == params.low_p])):
        crit_ax.plot(
            1,
            df_sdt.criterion[df_sdt.cue == params.low_p].iloc[index],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        crit_ax.plot(
            2,
            df_sdt.criterion[df_sdt.cue == params.high_p].iloc[index],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        crit_ax.plot(
            [1, 2],
            [
                df_sdt.criterion[df_sdt.cue == params.low_p].iloc[index],
                df_sdt.criterion[df_sdt.cue == params.high_p].iloc[index],
            ],
            marker="",
            markersize=0,
            color="gray",
            alpha=params.low_p,
        )

    crit_box = crit_ax.boxplot(
        [df_sdt.criterion[df_sdt.cue == params.low_p], df_sdt.criterion[df_sdt.cue == params.high_p]],
        patch_artist=True,
    )

    for patch, color in zip(crit_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(params.high_p)

    # Set the color for the medians in the plot
    for patch, color in zip(crit_box["medians"], med_color):
        patch.set_color(color)

    crit_ax.set_ylabel("c", fontname="Arial", fontsize=14)
    crit_ax.text(1.4, 1.5, "***", verticalalignment="center", fontname="Arial", fontsize="13")
    crit_ax.set_ylim(-0.5, 1.5)
    crit_ax.set_yticks([-0.5, 0.5, 1.5])
    crit_ax.set_yticklabels(["-0.5", "0.5", "1.5"], fontname="Arial", fontsize=12)
    # Plot confidence
    conf_ax = fig.add_subplot(gs[4:, 3])

    # Plot individual data points
    for index in range(len(conf_con[0])):
        conf_ax.plot(
            1,
            conf_con[0].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        conf_ax.plot(
            2,
            conf_con[1].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        conf_ax.plot(
            [1, 2],
            [conf_con[0].iloc[index, 1], conf_con[1].iloc[index, 1]],
            marker="",
            markersize=0,
            color="gray",
            alpha=params.low_p,
        )

    conf_ax.set_ylabel("high confidence", fontname="Arial", fontsize=14)
    conf_box = conf_ax.boxplot([conf_con[0].iloc[:, 1], conf_con[1].iloc[:, 1]], patch_artist=True)

    conf_ax.text(1.4, 1.0, "***", verticalalignment="center", fontname="Arial", fontsize="13")

    # Set the face color and alpha for the boxes in the plot
    colors = ["white", "black"]
    for patch, color in zip(conf_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(params.high_p)

    # Set the color for the medians in the plot
    med_color = ["black", "white"]
    for patch, color in zip(conf_box["medians"], med_color):
        patch.set_color(color)

    for plots in [hr_ax, fa_rate_ax, conf_ax]:
        plots.set_ylim(0, 1)
        plots.set_yticks([0, 0.5, 1.0])

    for plots in [hr_ax, fa_rate_ax, dprime_ax, crit_ax, conf_ax]:
        plots.spines["top"].set_visible(False)
        plots.spines["right"].set_visible(False)
        plots.set_xticks([1, 2])
        plots.set_xlim(0.5, 2.5)
        plots.set_xticklabels(["", ""])
        if plots != hr_ax:
            plots.set_xticklabels([str(params.low_p), str(params.high_p)], fontname="Arial", fontsize=12)
            plots.set_xlabel("P (Stimulus)", fontname="Arial", fontsize=14)
        if plots == conf_ax:
            plots.set_xticklabels(["congruent", "incongruent"], fontname="Arial", fontsize=12, rotation=30)
            plots.set_xlabel("")

    for fm in ["svg", "png"]:
        if exclude_high_fa:
            fig.savefig(savepath_fig1 / f"figure1_exclhighfa_{expecon}.{fm}", dpi=300, bbox_inches="tight", format=fm)
        else:
            fig.savefig(savepath_fig1 / f"figure1_{expecon}.{fm}", dpi=300, bbox_inches="tight", format=fm)
        plt.show()

    return "saved figure 1"


def calc_stats(expecon: int):
    """
    Calculate statistics and effect sizes for the behavioral data.

    Args:
    ----
    expecon: int : which study to analyze

    """
    conditions, _ = prepare_for_plotting(expecon=expecon, exclude_high_fa=False)

    # only for dprime, crit, hit-rate, fa-rate and confidence congruency
    df_sdt = conditions[0]
    conf = conditions[1]

    cond_list = ["criterion", "hit_rate", "fa_rate", "dprime"]
    for cond in cond_list:
        bootstrap_ci_effect_size_wilcoxon(
            x1=df_sdt[cond][df_sdt.cue == params.low_p].reset_index(drop=True),
            x2=df_sdt[cond][df_sdt.cue == params.high_p].reset_index(drop=True),
        )

        bootstrap_ci_effect_size_wilcoxon(
            conf[0].reset_index(drop=True).drop("ID", axis=1).iloc[:, 0],
            conf[1].reset_index(drop=True).drop("ID", axis=1).iloc[:, 0],
        )


def effect_wilcoxon(x1, x2):
    """
    Calculate effect size for the Wilcoxon signed-rank test (paired samples).

    Args:
    ----
    x1: numpy array or list, the first sample
    x2: numpy array or list, the second sample

    Return:
    ------
    r: float, rank biserial correlation coefficient
    statistic: float, test statistic from the Wilcoxon signed-rank test
    p_value: float, p-value from the Wilcoxon signed-rank test

    """
    if len(x1) != len(x2):
        msg = "The two samples must have the same length for paired analysis."
        raise ValueError(msg)

    statistic, p_value = stats.wilcoxon(x1, x2)

    # effect size rank biserial
    n = len(x1)
    r = 1 - (2 * statistic) / (n * (n + 1))

    return [r, statistic, p_value]


def bootstrap_ci_effect_size_wilcoxon(x1, x2, n_iterations=1000, alpha=0.95):
    """
    Calculate the confidence interval.

    The confidence interval is for rank biserial as an effect size for the Wilcoxon signed-rank test (paired samples).

    Args:
    ----
    x1: numpy array or list, the first sample
    x2: numpy array or list, the second sample
    n_iterations: int, the number of bootstrap iterations
    alpha: float, the confidence level

    Returns:
    -------
    lower_percentile: float, the lower percentile of the confidence interval
    upper_percentile: float, the upper percentile of the confidence interval

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


def supplementary_plots(expecon: int):
    """
    Create supplementary plots for the behavioral data.

    Args:
    ----
    expecon: int : which study to analyze

    """
    # set the save path
    savepath_fig1 = Path(paths.figures.manuscript.figure1) if expecon == 1 else Path(paths.expecon2.figures)

    # set colors for both conditions
    blue = "#2a95ffff"  # params.low_p cue
    red = "#ff2a2aff"

    colors = [blue, red]
    med_color = ["black", "black"]

    data, expecon = exclude_data(expecon=expecon)

    # calculate hit rates, false alarm rates, d-prime, and criterion per participant and cue condition
    # and per condition
    df_sdt = calculate_sdt_dataframe(data, "isyes", "sayyes", "ID", "cue")

    # Filter for correct trials only
    correct_only = data[data.correct == 1]
    incorrect_only = data[data.correct == 0]

    # filter for yes and no responses
    yes_response = correct_only[correct_only.sayyes == 1]
    no_response = correct_only[correct_only.sayyes == 0]

    # Calculate mean accuracy for each participant and cue condition
    data_grouped = data.groupby(["ID", "cue"])["correct"].mean()
    acc_high = data_grouped.unstack()[params.high_p].reset_index()
    acc_low = data_grouped.unstack()[params.low_p].reset_index()
    acc_cue = [acc_low, acc_high]

    # accuracy per condition
    for index in range(len(acc_cue[0])):
        plt.plot(
            1,
            acc_cue[0].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        plt.plot(
            2,
            acc_cue[1].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        plt.plot(
            [1, 2],
            [acc_cue[0].iloc[index, 1], acc_cue[1].iloc[index, 1]],
            marker="",
            markersize=0,
            color="gray",
            alpha=params.low_p,
        )

    acc_box = plt.boxplot([acc_cue[0].iloc[:, 1], acc_cue[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(acc_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(params.high_p)

    # Set the color for the medians in the plot
    for patch, color in zip(acc_box["medians"], med_color):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], [str(params.low_p), str(params.high_p)])
    plt.xlabel(xlabel="P (Stimulus)", fontname="Arial", fontsize=14)
    plt.ylabel(ylabel="accuracy", fontname="Arial", fontsize=14)

    plt.savefig(savepath_fig1 / "acc_cue.svg", dpi=300, bbox_inches="tight", format="svg")
    plt.show()

    # Calculate mean confidence for each participant and cue condition
    data_grouped = data.groupby(["ID", "cue"])["conf"].mean()
    conf_high = data_grouped.unstack()[params.high_p].reset_index()
    conf_low = data_grouped.unstack()[params.low_p].reset_index()
    conf_cue = [conf_low, conf_high]

    # is confidence higher for a certain cue?
    for index in range(len(conf_cue[0])):
        plt.plot(
            1,
            conf_cue[0].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        plt.plot(
            2,
            conf_cue[1].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        plt.plot(
            [1, 2],
            [conf_cue[0].iloc[index, 1], conf_cue[1].iloc[index, 1]],
            marker="",
            markersize=0,
            color="gray",
            alpha=params.low_p,
        )

    conf_box = plt.boxplot([conf_cue[0].iloc[:, 1], conf_cue[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(conf_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(params.high_p)

    # Set the color for the medians in the plot
    for patch, color in zip(conf_box["medians"], med_color):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], [str(params.low_p), str(params.high_p)])
    plt.xlabel(xlabel="P (Stimulus)", fontname="Arial", fontsize=14)
    plt.ylabel(ylabel="confidence", fontname="Arial", fontsize=14)

    plt.savefig(savepath_fig1 / "conf_cue.svg", dpi=300, bbox_inches="tight", format="svg")
    plt.show()

    # Calculate mean confidence for each participant and congruency condition
    # for yes responses only
    data_grouped = yes_response.groupby(["ID", "congruency"])["conf"].mean()
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    conf_con_yes = [con_condition, incon_condition]

    # congruency on confidence for yes and no responses
    colors = ["white", "black"]

    # Plot individual data points
    for index in range(len(conf_con_yes[0])):
        plt.plot(
            1,
            conf_con_yes[0].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        plt.plot(
            2,
            conf_con_yes[1].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        plt.plot(
            [1, 2],
            [conf_con_yes[0].iloc[index, 1], conf_con_yes[1].iloc[index, 1]],
            marker="",
            markersize=0,
            color="gray",
            alpha=params.low_p,
        )

    conf_con_yes_box = plt.boxplot([conf_con_yes[0].iloc[:, 1], conf_con_yes[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(conf_con_yes_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(params.high_p)

    # Set the color for the medians in the plot
    for patch, color in zip(conf_con_yes_box["medians"], med_color):
        patch.set_color(color)

        # Set x-axis tick labels
    plt.xticks([1, 2], ["Congruent", "Incongruent"])
    plt.xlabel(xlabel="Yes responses", fontname="Arial", fontsize=14)
    plt.ylabel(ylabel="Confidence", fontname="Arial", fontsize=14)

    plt.savefig(savepath_fig1 / "conf_con_yes.svg", dpi=300, bbox_inches="tight", format="svg")
    plt.show()

    # Calculate mean confidence for each participant and congruency condition
    # for no responses only
    data_grouped = no_response.groupby(["ID", "congruency"])["conf"].mean()
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    conf_con_no = [con_condition, incon_condition]

    # congruency on confidence for no responses
    for index in range(len(conf_con_no[0])):
        plt.plot(
            1,
            conf_con_no[0].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        plt.plot(
            2,
            conf_con_no[1].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        plt.plot(
            [1, 2],
            [conf_con_no[0].iloc[index, 1], conf_con_no[1].iloc[index, 1]],
            marker="",
            markersize=0,
            color="gray",
            alpha=params.low_p,
        )

    conf_con_no_box = plt.boxplot([conf_con_no[0].iloc[:, 1], conf_con_no[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(conf_con_no_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(params.high_p)

    # Set the color for the medians in the plot
    for patch, color in zip(conf_con_no_box["medians"], med_color):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], ["Congruent", "Incongruent"])
    plt.xlabel(xlabel="No responses", fontname="Arial", fontsize=14)
    plt.ylabel(ylabel="Mean confidence", fontname="Arial", fontsize=14)

    plt.savefig(savepath_fig1 / "conf_con_no.svg", dpi=300, bbox_inches="tight", format="svg")
    plt.show()

    # Calculate mean rts for each participant and congruency condition
    data_grouped = correct_only.groupby(["ID", "congruency_stim"])["respt1"].mean()
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    rt_con = [con_condition, incon_condition]

    # Reaction times for stimulus congruent trials (correct only)
    for index in range(len(rt_con[0])):
        plt.plot(
            1, rt_con[0].iloc[index, 1], marker="", markersize=8, color=colors[0], markeredgecolor=colors[0], alpha=0.5
        )
        plt.plot(
            2, rt_con[1].iloc[index, 1], marker="", markersize=8, color=colors[1], markeredgecolor=colors[1], alpha=0.5
        )
        plt.plot(
            [1, 2],
            [rt_con[0].iloc[index, 1], rt_con[1].iloc[index, 1]],
            marker="",
            markersize=0,
            color="gray",
            alpha=params.low_p,
        )

    rt_con_box = plt.boxplot([rt_con[0].iloc[:, 1], rt_con[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(rt_con_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(params.high_p)

    # Set the color for the medians in the plot
    for patch, color in zip(rt_con_box["medians"], med_color):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], ["Congruent", "Incongruent"])
    plt.xlabel(xlabel="Correct trials only", fontname="Arial", fontsize=14)
    plt.ylabel(ylabel="Mean response time", fontname="Arial", fontsize=14)

    plt.savefig(savepath_fig1 / "rt_con.svg", dpi=300, bbox_inches="tight", format="svg")
    plt.show()

    # Calculate mean rts for each participant and congruency condition
    data_grouped = incorrect_only.groupby(["ID", "congruency_stim"])["respt1"].mean()
    con_condition = data_grouped.unstack()[True].reset_index()
    incon_condition = data_grouped.unstack()[False].reset_index()
    rt_con_incorrect = [con_condition, incon_condition]

    # Incorrect trials only
    for index in range(len(rt_con_incorrect[0])):
        plt.plot(
            1,
            rt_con_incorrect[0].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        plt.plot(
            2,
            rt_con_incorrect[1].iloc[index, 1],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        plt.plot(
            [1, 2],
            [rt_con_incorrect[0].iloc[index, 1], rt_con_incorrect[1].iloc[index, 1]],
            marker="",
            markersize=0,
            color="gray",
            alpha=params.low_p,
        )

    rt_con_in_box = plt.boxplot([rt_con_incorrect[0].iloc[:, 1], rt_con_incorrect[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(rt_con_in_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(params.high_p)

    # Set the color for the medians in the plot
    for patch, color in zip(rt_con_in_box["medians"], med_color):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], ["Congruent", "Incongruent"])
    plt.xlabel(xlabel="Incorrect trials only", fontname="Arial", fontsize=14)
    plt.ylabel(ylabel="Mean response time", fontname="Arial", fontsize=14)

    plt.savefig(savepath_fig1 / "rt_con_incorrect.svg", dpi=300, bbox_inches="tight", format="svg")
    plt.show()

    # Plot relationship between dprime and criterion
    x = df_sdt.criterion[df_sdt.cue == params.high_p]
    y = df_sdt.dprime[df_sdt.cue == params.high_p]

    reg = sns.regplot(x=x, y=y)

    # Fit a linear regression model to calculate the p-value
    model = sm.OLS(x, sm.add_constant(y))
    results = model.fit()
    p_value = results.pvalues[1]

    # Retrieve the regression line equation and round the coefficients
    slope, intercept = reg.get_lines()[0].get_data()
    slope = round(slope[1], 2)
    intercept = round(intercept[1], 2)

    # Add the regression line equation and p-value to the plot with Arial font and size 14
    equation = f"y = {slope}x + {intercept}"
    p_value_text = f"p-value: {p_value:.4f}"

    font = {"family": params.plot.font.sans_serif, "size": params.plot.font.size}

    plt.annotate(equation, xy=(0.05, 0.9), xycoords="axes fraction", fontproperties=font)
    plt.annotate(p_value_text, xy=(0.05, 0.8), xycoords="axes fraction", fontproperties=font)

    plt.xlabel(xlabel=f"dprime {params.high_p}", fontname=params.plot.font.sans_serif, fontsize=params.plot.font.size)
    plt.ylabel(ylabel=f"c {params.high_p}", fontname=params.plot.font.sans_serif, fontsize=params.plot.font.size)

    plt.savefig(savepath_fig1 / "dprime_c.svg", dpi=300, bbox_inches="tight", format="svg")

    return "saved all supplementary plots"


# Helper functions ###############################################


def calculate_sdt_dataframe(df_study, signal_col, response_col, subject_col, condition_col):
    """
    Calculate SDT measures (d' and criterion) for each participant and each condition based on a dataframe.

    Args:
    ----
    df_study: Pandas dataframe containing the data
    signal_col: Name of the column indicating signal presence (e.g., 'signal')
    response_col: Name of the column indicating participant response (e.g., 'response')
    subject_col: Name of the column indicating participant ID (e.g., 'ID')
    condition_col: Name of the column indicating condition (e.g., 'condition')

    Returns:
    -------
    Pandas dataframe containing the calculated SDT measures (d' and criterion)
    for each participant and condition.

    """
    # Apply Hautus correction and calculate SDT measures for each participant
    # and each condition
    results = []
    subjects = df_study[subject_col].unique()
    conditions = df_study[condition_col].unique()

    for subject in subjects:
        for condition in conditions:
            subset = df_study[(df_study[subject_col] == subject) & (df_study[condition_col] == condition)]

            detect_hits = subset[(subset[signal_col] == True) & (subset[response_col] == True)].shape[0]
            detect_misses = subset[(subset[signal_col] == True) & (subset[response_col] == False)].shape[0]
            false_alarms = subset[(subset[signal_col] == False) & (subset[response_col] == True)].shape[0]
            correct_rejections = subset[(subset[signal_col] == False) & (subset[response_col] == False)].shape[0]

            # log linear correction (Hautus, 1995)
            hit_rate = (detect_hits + 0.5) / (detect_hits + detect_misses + 1)
            false_alarm_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)

            d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate)
            criterion = -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(false_alarm_rate))

            results.append((subject, condition, hit_rate, false_alarm_rate, d_prime, criterion))

    # Create a new dataframe with the results
    return pd.DataFrame(results, columns=[subject_col, condition_col, "hit_rate", "fa_rate", "dprime", "criterion"])


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
