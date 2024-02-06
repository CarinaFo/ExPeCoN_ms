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

from expecon_ms.configs import PROJECT_ROOT, path_to

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# Specify the file path for which you want the last commit date
__file__path = Path(PROJECT_ROOT, "code/expecon_ms/behav/figure1.py")  # == __file__

last_commit_date = (
    subprocess.check_output(["git", "log", "-1", "--format=%cd", "--follow", __file__path]).decode("utf-8").strip()
)
print("Last Commit Date for", __file__path, ":", last_commit_date)

# Set Arial as the default font
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 14

# set up behavioral data path
behav_path = Path(path_to.data.behavior)

# set save paths (figure 1 is study 1 (block design) and figure 2 is study2 (single trial design))
save_path_fig1 = Path(path_to.figures.manuscript.figure1)
save_path_fig2 = Path(path_to.figures.manuscript.figure2)
save_path_fig1.mkdir(parents=True, exist_ok=True)
save_path_fig2.mkdir(parents=True, exist_ok=True)
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
        data = pd.read_csv(behav_path / "behav_cleaned_for_eeg_expecon1.csv")

        # Clean up the dataframe by dropping unnecessary columns
        columns_to_drop = [col for col in data.columns if "Unnamed" in col]

        data = data.drop(columns_to_drop, axis=1)

        # Change the block number for participant 7's block 3
        data.loc[(144 * 2): (144 * 3), "block"] = 4

    else:
        data = pd.read_csv(behav_path / "behav_cleaned_for_eeg_expecon2.csv")

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
    data["congruency"] = ((data.cue == 0.25) & (data.sayyes == 0)) | ((data.cue == 0.75) & (data.sayyes == 1))
    # Add a 'congruency stimulus' column
    data["congruency_stim"] = ((data.cue == 0.25) & (data.isyes == 0)) | ((data.cue == 0.75) & (data.isyes == 1))

    # add a column that combines the confidence ratings and the
    # detection response
    data['conf_resp'] = [
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
    data = data.drop(data[data.respt1 == 2.5].index)
    data = data.drop(data[data.respt1 < 0.1].index)

    # save the preprocessed dataframe
    data.to_csv(f"{behav_path}{Path('/')}behav_data_exclrts_{str(expecon)}.csv")

    return data

def reproduce_interaction_non_model_based(expecon: int):
    """
    Reproduce the interaction between stim. probability and the previous response.

    Args:
    ----
    expecon: int: which dataset to use: expecon 1 or expecon 2

    Returns:
    -------
    None

    """
    # load the preprocessed data
    data = pd.read_csv(f"{behav_path}{Path('/')}prepro_behav_data_{str(expecon)}.csv")

    # calculate the interaction between the cue condition and the previous response on the current response
    # for each participant
    interaction = data.groupby(["ID", "prevresp", "cue"])[["sayyes"]].mean() \
                      .unstack().unstack()
    interaction.plot(kind="box")
    # rename x labels and tilt them for better readability
    plt.xticks([0, 1, 2, 3], ["0.25_prev_no", "0.25_prev_yes",
                              "0.75_prev_no", "0.75_prev_yes"])
    plt.xticks(rotation=45)
    # set y label
    plt.ylabel("mean detection response")
    # run dependent samples ttests between the 4 conditions
    print(stats.ttest_rel(interaction["sayyes"][0.25][0], interaction["sayyes"][0.75][0]))
    print(stats.ttest_rel(interaction["sayyes"][0.25][1], interaction["sayyes"][0.75][1]))
    print(stats.ttest_rel(interaction["sayyes"][0.25][0], interaction["sayyes"][0.25][1]))
    print(stats.ttest_rel(interaction["sayyes"][0.75][0], interaction["sayyes"][0.75][1]))

    if expecon == 1:
        plt.savefig(f"{save_path_fig1}{Path('/')}interaction_prevresp_cue_{str(expecon)}.png")
        # save svg file
        plt.savefig(f"{save_path_fig1}{Path('/')}interaction_prevresp_cue_{str(expecon)}.svg")
    else:
        plt.savefig(f"{save_path_fig2}{Path('/')}interaction_prevresp_cue_{str(expecon)}.png")
        # save svg file
        plt.savefig(f"{save_path_fig2}{Path('/')}interaction_prevresp_cue_{str(expecon)}.svg")

    plt.show()

def exclude_data(expecon: int):
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
    data = pd.read_csv(f'{behav_path}{Path("/")}behav_data_exclrts_{str(expecon)}.csv')

    # Calculate hit rates by participant and cue condition
    signal = data[data.isyes == 1]
    hit_rate_per_subject = signal.groupby(["ID"])["sayyes"].mean()

    print(f"Mean hit rate: {np.mean(hit_rate_per_subject):.2f}")
    print(f"Standard deviation: {np.std(hit_rate_per_subject):.2f}")
    print(f"Minimum hit rate: {np.min(hit_rate_per_subject):.2f}")
    print(f"Maximum hit rate: {np.max(hit_rate_per_subject):.2f}")

    # Calculate hit rates by participant and block condition
    hitrate_per_block = signal.groupby(["ID", "block"])["sayyes"].mean()

    # Filter the grouped object based on hit rate conditions
    hitrate_abn = hitrate_per_block[(hitrate_per_block > 0.9) | (hitrate_per_block < 0.2)]
    print("Blocks with hit rates > 0.9 or < 0.2: ", len(hitrate_abn))

    # Extract the ID and block information from the filtered groups
    remove_hit_rates = hitrate_abn.reset_index()

    # Calculate hit rates by participant and cue condition
    noise = data[data.isyes == 0]
    fa_rate_per_block = noise.groupby(["ID", "block"])["sayyes"].mean()

    # Filter the grouped object based on fa rate conditions
    fa_rate_abn = fa_rate_per_block[fa_rate_per_block > 0.4]
    print("Blocks with false alarm rates > 0.4: ", len(fa_rate_abn))

    # Extract the ID and block information from the filtered groups
    remove_fa_rates = fa_rate_abn.reset_index()

    # Filter the grouped objects based on the conditions
    hit_fa = hitrate_per_block[hitrate_per_block - fa_rate_per_block < 0.05]  # Difference < 5 %
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

    data.to_csv(f'{behav_path}{Path("/")}prepro_behav_data_{str(expecon)}.csv')

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
    df = exclude_data(expecon=expecon)

    # calculate hit rates, false alarm rates, d-prime, and criterion per participant and cue condition
    df_sdt = calculate_sdt_dataframe(df, "isyes", "sayyes", "ID", "cue")

    # calculate hit rate change between conditions per participant
    diff_hit =  df_sdt.hit_rate[df_sdt.cue == 0.75].reset_index() - df_sdt.hit_rate[df_sdt.cue == 0.25].reset_index()

    # calculate fa rate change between conditions per participant
    diff_fa = df_sdt.fa_rate[df_sdt.cue == 0.75].reset_index() - df_sdt.fa_rate[df_sdt.cue == 0.25].reset_index()

    # calculate dprime change between conditions per participant
    diff_dprime = df_sdt.dprime[df_sdt.cue == 0.75].reset_index() - df_sdt.dprime[df_sdt.cue == 0.25].reset_index()

    # calculate criterion change between conditions per participant
    diff_crit = df_sdt.criterion[df_sdt.cue == 0.75].reset_index() - df_sdt.criterion[df_sdt.cue == 0.25].reset_index()

    # Filter for correct trials only
    correct_only = df[df.correct == 1]

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


def plot_mean_response_and_confidence(
    expecon: int,
    blue="#0571b0", red="#ca0020",
    no_col="#088281", yes_col="#d01c8b",
    savepath=path_to.figures.manuscript.figure2_suppl):
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
    sns.stripplot(x="cue", y="sayyes", data=mean_resp_id_cue, color="black", 
                  size=4, jitter=True)
    plt.xlabel("stimulus probability")
    plt.ylabel("% yes responses")
    plt.savefig(Path(savepath, f"choice_cue_{expecon}.svg"))
    plt.savefig(Path(savepath, f"choice_cue_{expecon}.png"))
    plt.show()

    # Perform the Wilcoxon signed-rank test
    wilcoxon_statistic, p_value = stats.wilcoxon(
        mean_resp_id_cue[mean_resp_id_cue.cue == 0.25].sayyes,
        mean_resp_id_cue[mean_resp_id_cue.cue == 0.75].sayyes
    )
    print(f"Wilcoxon statistic: {wilcoxon_statistic}")
    print(f"p-value: {p_value}")

    # previous response distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="cue", y="prevresp", data=mean_prevresp_id_cue, palette=[blue, red])
    sns.stripplot(x="cue", y="prevresp", data=mean_prevresp_id_cue, color="black",
                  size=4, jitter=True)
    plt.xlabel("stimulus probability")
    plt.ylabel("% previous yes responses")
    plt.savefig(Path(savepath, f"prevchoice_cue_{expecon}.svg"))
    plt.savefig(Path(savepath, f"prevchoice_cue_{expecon}.png"))
    plt.show()

    # Perform the Wilcoxon signed-rank test
    wilcoxon_statistic, p_value = stats.wilcoxon(
        mean_prevresp_id_cue[mean_prevresp_id_cue.cue == 0.25].prevresp,
        mean_prevresp_id_cue[mean_prevresp_id_cue.cue == 0.75].prevresp
    )
    print(f"Wilcoxon statistic: {wilcoxon_statistic}")
    print(f"p-value: {p_value}")

    # confidence distribution
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="sayyes", y="conf", data=mean_conf_id_resp, palette=[no_col, yes_col])
    sns.stripplot(x="sayyes", y="conf", data=mean_conf_id_resp, color="black", 
                  size=4, jitter=True)
    plt.xlabel("detection response")
    plt.ylabel("% high confidence")
    plt.xticks(ticks=[0, 1], labels=["No", "Yes"])  # Set custom tick labels
    plt.savefig(Path(savepath, f"choice_conf_{expecon}.svg"))
    plt.savefig(Path(savepath, f"choice_conf_{expecon}.png"))
    plt.show()

    wilcoxon_statistic, p_value = stats.wilcoxon(
    mean_conf_id_resp[mean_conf_id_resp.sayyes == 1].conf,
    mean_conf_id_resp[mean_conf_id_resp.sayyes == 0].conf
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

    # create a boolean mask for participants with very high farates
    fa_rate_high_indices = np.where(df_sdt.fa_rate[df_sdt.cue == 0.75] > 0.4)
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
    savepath_fig1 = Path(save_path_fig1 if expecon == 1 else save_path_fig2)

    # load data
    conditions, exclude_high_fa = prepare_for_plotting(exclude_high_fa=exclude_high_fa,
                                                        expecon=expecon)

    # unpack data
    df_sdt, conf_con, conf_yes, conf_no = conditions

    # set colors for both conditions
    blue = "#0571b0"  # 0.25 cue
    red = "#ca0020"  # 0.75 cue

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
    for index in range(len(df_sdt.hit_rate[df_sdt.cue == 0.25])):
        hr_ax.plot(
            1,
            df_sdt.hit_rate[df_sdt.cue == 0.25].iloc[index],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        hr_ax.plot(
            2,
            df_sdt.hit_rate[df_sdt.cue == 0.75].iloc[index],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        hr_ax.plot(
            [1, 2],
            [df_sdt.hit_rate[df_sdt.cue == 0.25].iloc[index], df_sdt.hit_rate[df_sdt.cue == 0.75].iloc[index]],
            marker="",
            markersize=0,
            color="gray",
            alpha=0.25,
        )

    hr_box = hr_ax.boxplot(
        [df_sdt.hit_rate[df_sdt.cue == 0.25], df_sdt.hit_rate[df_sdt.cue == 0.75]], patch_artist=True
    )

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(hr_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(hr_box["medians"], med_color):
        patch.set_color(color)

    hr_ax.set_ylabel("hit rate", fontname="Arial", fontsize=14)
    hr_ax.set_yticklabels(["0", "0.5", "1.0"], fontname="Arial", fontsize=12)
    hr_ax.text(1.3, 1, "***", verticalalignment="center", fontname="Arial", fontsize="18")

    # Plot fa rate
    fa_rate_ax = fig.add_subplot(gs[5, 0])

    for index in range(len(df_sdt.fa_rate[df_sdt.cue == 0.25])):
        fa_rate_ax.plot(
            1,
            df_sdt.fa_rate[df_sdt.cue == 0.25].iloc[index],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        fa_rate_ax.plot(
            2,
            df_sdt.fa_rate[df_sdt.cue == 0.75].iloc[index],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        fa_rate_ax.plot(
            [1, 2],
            [df_sdt.fa_rate[df_sdt.cue == 0.25].iloc[index], df_sdt.fa_rate[df_sdt.cue == 0.75].iloc[index]],
            marker="",
            markersize=0,
            color="gray",
            alpha=0.25,
        )

    fa_box = fa_rate_ax.boxplot(
        [df_sdt.fa_rate[df_sdt.cue == 0.25], df_sdt.fa_rate[df_sdt.cue == 0.75]], patch_artist=True
    )

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(fa_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(fa_box["medians"], med_color):
        patch.set_color(color)

    fa_rate_ax.set_ylabel("fa rate", fontname="Arial", fontsize=14)
    fa_rate_ax.set_yticklabels(["0", "0.5", "1.0"], fontname="Arial", fontsize=12)
    fa_rate_ax.text(1.3, 1, "***", verticalalignment="center", fontname="Arial", fontsize="18")

    # Plot dprime
    dprime_ax = fig.add_subplot(gs[4:, 1])

    # Plot individual data points
    for index in range(len(df_sdt.dprime[df_sdt.cue == 0.25])):
        dprime_ax.plot(
            1,
            df_sdt.dprime[df_sdt.cue == 0.25].iloc[index],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        dprime_ax.plot(
            2,
            df_sdt.dprime[df_sdt.cue == 0.75].iloc[index],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        dprime_ax.plot(
            [1, 2],
            [df_sdt.dprime[df_sdt.cue == 0.25].iloc[index], df_sdt.dprime[df_sdt.cue == 0.75].iloc[index]],
            marker="",
            markersize=0,
            color="gray",
            alpha=0.25,
        )

    dprime_box = dprime_ax.boxplot(
        [df_sdt.dprime[df_sdt.cue == 0.25], df_sdt.dprime[df_sdt.cue == 0.75]], patch_artist=True
    )

    for patch, color in zip(dprime_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

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
    for index in range(len(df_sdt.criterion[df_sdt.cue == 0.25])):
        crit_ax.plot(
            1,
            df_sdt.criterion[df_sdt.cue == 0.25].iloc[index],
            marker="",
            markersize=8,
            color=colors[0],
            markeredgecolor=colors[0],
            alpha=0.5,
        )
        crit_ax.plot(
            2,
            df_sdt.criterion[df_sdt.cue == 0.75].iloc[index],
            marker="",
            markersize=8,
            color=colors[1],
            markeredgecolor=colors[1],
            alpha=0.5,
        )
        crit_ax.plot(
            [1, 2],
            [df_sdt.criterion[df_sdt.cue == 0.25].iloc[index], df_sdt.criterion[df_sdt.cue == 0.75].iloc[index]],
            marker="",
            markersize=0,
            color="gray",
            alpha=0.25,
        )

    crit_box = crit_ax.boxplot(
        [df_sdt.criterion[df_sdt.cue == 0.25], df_sdt.criterion[df_sdt.cue == 0.75]], patch_artist=True
    )

    for patch, color in zip(crit_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

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
            alpha=0.25,
        )

    conf_ax.set_ylabel("high confidence", fontname="Arial", fontsize=14)
    conf_box = conf_ax.boxplot([conf_con[0].iloc[:, 1], conf_con[1].iloc[:, 1]], patch_artist=True)

    conf_ax.text(1.4, 1.0, "***", verticalalignment="center", fontname="Arial", fontsize="13")

    # Set the face color and alpha for the boxes in the plot
    colors = ["white", "black"]
    for patch, color in zip(conf_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

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
            plots.set_xticklabels(["0.25", "0.75"], fontname="Arial", fontsize=12)
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

    """Calculate statistics and effect sizes for the behavioral data.
    
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
            x1=df_sdt[cond][df_sdt.cue == 0.25].reset_index(drop=True),
            x2=df_sdt[cond][df_sdt.cue == 0.75].reset_index(drop=True),
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
    savepath_fig1 = save_path_fig1 if expecon == 1 else Path(path_to.expecon2.figures)

    # set colors for both conditions
    blue = "#2a95ffff"  # 0.25 cue
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
    acc_high = data_grouped.unstack()[0.75].reset_index()
    acc_low = data_grouped.unstack()[0.25].reset_index()
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
            alpha=0.25,
        )

    acc_box = plt.boxplot([acc_cue[0].iloc[:, 1], acc_cue[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(acc_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(acc_box["medians"], med_color):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], ["0.25", "0.75"])
    plt.xlabel(xlabel="P (Stimulus)", fontname="Arial", fontsize=14)
    plt.ylabel(ylabel="accuracy", fontname="Arial", fontsize=14)

    plt.savefig(savepath_fig1 / "acc_cue.svg", dpi=300, bbox_inches="tight", format="svg")
    plt.show()

    # Calculate mean confidence for each participant and cue condition
    data_grouped = data.groupby(["ID", "cue"])["conf"].mean()
    conf_high = data_grouped.unstack()[0.75].reset_index()
    conf_low = data_grouped.unstack()[0.25].reset_index()
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
            alpha=0.25,
        )

    conf_box = plt.boxplot([conf_cue[0].iloc[:, 1], conf_cue[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(conf_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    # Set the color for the medians in the plot
    for patch, color in zip(conf_box["medians"], med_color):
        patch.set_color(color)

    # Set x-axis tick labels
    plt.xticks([1, 2], ["0.25", "0.75"])
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
            alpha=0.25,
        )

    conf_con_yes_box = plt.boxplot([conf_con_yes[0].iloc[:, 1], conf_con_yes[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(conf_con_yes_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

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
            alpha=0.25,
        )

    conf_con_no_box = plt.boxplot([conf_con_no[0].iloc[:, 1], conf_con_no[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(conf_con_no_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

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
            alpha=0.25,
        )

    rt_con_box = plt.boxplot([rt_con[0].iloc[:, 1], rt_con[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(rt_con_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

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
            alpha=0.25,
        )

    rt_con_in_box = plt.boxplot([rt_con_incorrect[0].iloc[:, 1], rt_con_incorrect[1].iloc[:, 1]], patch_artist=True)

    # Set the face color and alpha for the boxes in the plot
    for patch, color in zip(rt_con_in_box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

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
    x = df_sdt.criterion[df_sdt.cue == 0.75]
    y = df_sdt.dprime[df_sdt.cue == 0.75]

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

    font = {"family": "Arial", "size": 14}

    plt.annotate(equation, xy=(0.05, 0.9), xycoords="axes fraction", fontproperties=font)
    plt.annotate(p_value_text, xy=(0.05, 0.8), xycoords="axes fraction", fontproperties=font)

    plt.xlabel(xlabel="dprime 0.75", fontname="Arial", fontsize=14)
    plt.ylabel(ylabel="c 0.75", fontname="Arial", fontsize=14)

    plt.savefig(savepath_fig1 / "dprime_c.svg", dpi=300, bbox_inches="tight", format="svg")

    return "saved all supplementary plots"


########################################### Helper functions ###############################################

def calculate_sdt_dataframe(df, signal_col, response_col, subject_col, condition_col):
    """
    Calculate SDT measures (d' and criterion) for each participant and each condition based on a dataframe.

    Args:
    ----
    df: Pandas dataframe containing the data
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
    subjects = df[subject_col].unique()
    conditions = df[condition_col].unique()

    for subject in subjects:
        for condition in conditions:
            subset = df[(df[subject_col] == subject) & (df[condition_col] == condition)]

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
