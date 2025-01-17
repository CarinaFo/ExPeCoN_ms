"""
The script contains functions to analyze and plot the behavioral data for the ExPeCoN study.

Also, it produces suppl. fig. 5.2

The ExPeCoN study investigates stimulus probabilities and the influence on perception and confidence in a
near-threshold somatosensory detection task in two paradigms that vary in their probability environment:

dataset 1 : stable environment, 144 trials in 5 blocks, 43 participants,
           stimulus probability is cued before blocks of 12 trials.
dataset 2: variable environment, 120 trials in 5 blocks, 40 participants,
           stimulus probability is cued before each trial.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2024
"""
# %% Import
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels
from scipy import stats

from expecon_ms.configs import paths


def reproduce_interaction_non_model_based():
    """
    Reproduce the interaction between stim. probability and the previous response.

    Args:
    ----
    expecon: int: which dataset to use: expecon 1 or expecon 2

    Returns:
    -------
    None

    """
    # Load the data
    data = pd.read_csv(Path(paths.data.behavior, "prepro_behav_data_expecon1_2.csv"))

    # drop columns with only NaNs
    data = data.dropna(axis=1, how="all")

    # remove rows with missing values
    data = data.dropna()

    # Assuming your dataframe is named 'data'
    # Group by ID, prev_response, and cue and calculate d' and criterion with Hautus correction
    sdt_results = calculate_sdt_dataframe(data, "isyes", "sayyes", "ID", "prevresp", "cue", "study")

    # Melt the dataframe to long format
    sdt_results_long = pd.melt(
        sdt_results,
        id_vars=["ID", "prevresp", "cue", "study"],
        value_vars=["dprime", "criterion"],
        var_name="Measure",
        value_name="Value",
    )

    # Define color palette for the different cue conditions
    colors = ["#0571b0ff", "#ca0020ff"]

    # Create a figure with 2 rows for d' and criterion
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))

    # Plot Criterion for Study 1
    sns.boxplot(
        x="prevresp",
        y="Value",
        hue="cue",
        data=sdt_results_long[(sdt_results_long["Measure"] == "criterion") & (sdt_results_long["study"] == 1)],
        ax=axes[0],
        showmeans=False,
        palette=colors,
    )
    sns.swarmplot(
        x="prevresp",
        y="Value",
        hue="cue",
        data=sdt_results_long[(sdt_results_long["Measure"] == "criterion") & (sdt_results_long["study"] == 1)],
        ax=axes[0],
        color=".25",
        alpha=0.5,
        dodge=True,
    )

    axes[0].set_title("Stable environment: \nCriterion by Previous Response")
    axes[0].set_xlabel("Previous Response")
    axes[0].set_ylabel("Criterion")
    axes[0].set_xticklabels(["No", "Yes"])
    axes[0].set_ylim(-1, 2)  # Set the y-axis limits here
    axes[0].get_legend().remove()

    # Plot Criterion for Study 2
    sns.boxplot(    
        x="prevresp",
        y="Value",
        hue="cue",
        data=sdt_results_long[(sdt_results_long["Measure"] == "criterion") & (sdt_results_long["study"] == 2)],
        ax=axes[1],
        showmeans=False,
        palette=colors,
    )
    sns.swarmplot(
        x="prevresp",
        y="Value",
        hue="cue",
        data=sdt_results_long[(sdt_results_long["Measure"] == "criterion") & (sdt_results_long["study"] == 2)],
        ax=axes[1],
        color=".25",
        alpha=0.5,
        dodge=True,
    )

    axes[1].set_title("Volatile environment: \nCriterion by Previous Response")
    axes[1].set_xlabel("Previous Response")
    axes[1].set_ylabel("Criterion")
    axes[1].set_xticklabels(["No", "Yes"])
    axes[1].set_ylim(-1, 2)  # Set the y-axis limits here
    axes[1].get_legend().remove()

    plt.tight_layout()
    plt.savefig(Path(paths.figures.manuscript.figure2, f"interaction_prevresp_cue_criterion.png"))
    plt.savefig(Path(paths.figures.manuscript.figure2, f"interaction_prevresp_cue_criterion.svg"))
    plt.show()

    # run anova for the criterion between high and low cue conditions conditioned on previous response
    # and control for multiple comparisons
    import pingouin as pg

    for study in [1, 2]:
        # control for multiple comparisons
        subset = sdt_results[sdt_results["study"] == study]
        # Run a repeated-measures ANOVA
        anova_results = pg.rm_anova(
                        data=subset,
                        dv='criterion',        # Dependent variable
                        within=['cue', 'prevresp'],  # Within-subject factors
                        subject='ID',          # Subject identifier
                        detailed=True          # Get detailed output
                    )
        print(anova_results)

        # if interaction is significant, run pairwise tests
        # Extract p-value for the interaction
        interaction_pval = anova_results.loc[anova_results['Source'] == 'cue * prevresp', 'p-unc'].values[0]

        # Check significance and run specific tests
        if interaction_pval < 0.05:
            # Example: Compare Stim_Prob ('High' vs. 'Low') within each level of Prev_Response
            for response in subset['prevresp'].unique():
                subset_prev = subset[subset['prevresp'] == response]
                posthoc_interaction = pg.pairwise_ttests(
                    data=subset_prev, dv='criterion', within='cue', subject='ID', padjust='bonf'
                )
                print(f"Post hoc for Previous Response = {response}:\n", posthoc_interaction)

            # Example: Compare Prev_Response ('Yes' vs. 'No') within each level of Stim_Prob
            for prob in subset['cue'].unique():
                subset_prob = subset[subset['cue'] == prob]
                posthoc_interaction = pg.pairwise_ttests(
                    data=subset_prob, dv='criterion', within='prevresp', subject='ID', padjust='bonf'
                )
                print(f"Post hoc for Stimulus Probability = {prob}:\n", posthoc_interaction)

        else:

            # Pairwise tests for Stim_Prob
            posthoc_stim = pg.pairwise_tests(
                data=subset, dv='criterion', within='cue', subject='ID', padjust='bonf'
            )
            print(posthoc_stim)

            # Pairwise tests for Prev_Response
            posthoc_prev = pg.pairwise_tests(
                data=subset, dv='criterion', within='prevresp', subject='ID', padjust='bonf'
            )
            print(posthoc_prev)


    # Create a figure with 2 rows for d' and criterion
    fig, axes = plt.subplots(1, 2, figsize=(12, 10))

    # Plot Dprime for Study 1
    sns.boxplot(
        x="prevresp",
        y="Value",
        hue="cue",
        data=sdt_results_long[(sdt_results_long["Measure"] == "dprime") & (sdt_results_long["study"] == 1)],
        ax=axes[0],
        showmeans=False,
        palette=colors,
    )
    sns.swarmplot(
        x="prevresp",
        y="Value",
        hue="cue",
        data=sdt_results_long[(sdt_results_long["Measure"] == "dprime") & (sdt_results_long["study"] == 1)],
        ax=axes[0],
        color=".25",
        alpha=0.5,
        dodge=True,
    )

    axes[0].set_title("Stable environment: \nDprime by Previous Response")
    axes[0].set_xlabel("Previous Response")
    axes[0].set_ylabel("Dprime")
    axes[0].set_xticklabels(["No", "Yes"])
    axes[0].get_legend().remove()
    axes[0].set_ylim(0, 3)  # Set the y-axis limits here

    # Plot Dprime for Study 2
    sns.boxplot(
        x="prevresp",
        y="Value",
        hue="cue",
        data=sdt_results_long[(sdt_results_long["Measure"] == "dprime") & (sdt_results_long["study"] == 2)],
        ax=axes[1],
        showmeans=False,
        palette=colors,
    )
    sns.swarmplot(
        x="prevresp",
        y="Value",
        hue="cue",
        data=sdt_results_long[(sdt_results_long["Measure"] == "dprime") & (sdt_results_long["study"] == 2)],
        ax=axes[1],
        color=".25",
        alpha=0.5,
        dodge=True,
    )

    axes[1].set_title("Volatile environment: \nDprime by Previous Response")
    axes[1].set_xlabel("Previous Response")
    axes[1].set_ylabel("Dprime")
    axes[1].set_xticklabels(["No", "Yes"])
    axes[1].set_ylim(0, 3)  # Set the y-axis limits here
    axes[1].get_legend().remove()

    plt.tight_layout()
    plt.savefig(Path(paths.figures.manuscript.figure2, f"interaction_prevresp_cue_dprime.png"))
    plt.savefig(Path(paths.figures.manuscript.figure2, f"interaction_prevresp_cue_dprime.svg"))
    plt.show()

    # now run ANOVA and post hoc tests

    for study in [1, 2]:
        # control for multiple comparisons
        subset = sdt_results[sdt_results["study"] == study]
        # Run a repeated-measures ANOVA
        anova_results = pg.rm_anova(
                        data=subset,
                        dv='dprime',        # Dependent variable
                        within=['cue', 'prevresp'],  # Within-subject factors
                        subject='ID',          # Subject identifier
                        detailed=True          # Get detailed output
                    )
        print(anova_results)

        # if interaction is significant, run pairwise tests
        # Extract p-value for the interaction
        interaction_pval = anova_results.loc[anova_results['Source'] == 'cue * prevresp', 'p-unc'].values[0]

        # Check significance and run specific tests
        if interaction_pval < 0.05:
            # Example: Compare Stim_Prob ('High' vs. 'Low') within each level of Prev_Response
            for response in subset['prevresp'].unique():
                subset_prev = subset[subset['prevresp'] == response]
                posthoc_interaction = pg.pairwise_ttests(
                    data=subset_prev, dv='dprime', within='cue', subject='ID', padjust='bonf'
                )
                print(f"Post hoc for Previous Response = {response}:\n", posthoc_interaction)

            # Example: Compare Prev_Response ('Yes' vs. 'No') within each level of Stim_Prob
            for prob in subset['cue'].unique():
                subset_prob = subset[subset['cue'] == prob]
                posthoc_interaction = pg.pairwise_ttests(
                    data=subset_prob, dv='dprime', within='prevresp', subject='ID', padjust='bonf'
                )
                print(f"Post hoc for Stimulus Probability = {prob}:\n", posthoc_interaction)

        else:

            # Pairwise tests for Stim_Prob
            posthoc_stim = pg.pairwise_tests(
                data=subset, dv='dprime', within='cue', subject='ID', padjust='bonf'
            )
            print(posthoc_stim)

            # Pairwise tests for Prev_Response
            posthoc_prev = pg.pairwise_tests(
                data=subset, dv='dprime', within='prevresp', subject='ID', padjust='bonf'
            )
            print(posthoc_prev)


# Helper functions ###############################################


def calculate_sdt_dataframe(
    df_study, signal_col, response_col, subject_col, condition1_col, condition2_col, study_col
):
    """
    Calculate SDT measures (d' and criterion) for each participant and each condition based on a dataframe.

    Args:
    ----
    df_study: Pandas dataframe containing the data
    signal_col: Name of the column indicating signal presence (e.g., 'signal')
    response_col: Name of the column indicating participant response (e.g., 'response')
    subject_col: Name of the column indicating participant ID (e.g., 'ID')
    condition1_col: Name of the column indicating condition (e.g., 'condition')
    condition2_col: Name of the column indicating condition (e.g., 'condition')

    Returns:
    -------
    Pandas dataframe containing the calculated SDT measures (d' and criterion)
    for each participant and condition.

    """
    # Initialize a list to store the results
    results = []

    # Iterate over unique values of study
    for study in df_study[study_col].unique():
        df_study_subset = df_study[df_study[study_col] == study]

        # Iterate over unique subjects in the study
        for subject in df_study_subset[subject_col].unique():
            df_subject_subset = df_study_subset[df_study_subset[subject_col] == subject]

            # Iterate over unique conditions in the study
            for cond1 in df_subject_subset[condition1_col].unique():
                for cond2 in df_subject_subset[condition2_col].unique():
                    subset = df_subject_subset[
                        (df_subject_subset[condition1_col] == cond1) & (df_subject_subset[condition2_col] == cond2)
                    ]

                    # Count the occurrences of different response types
                    detect_hits = subset[(subset[signal_col] == 1) & (subset[response_col] == 1)].shape[0]
                    detect_misses = subset[(subset[signal_col] == 1) & (subset[response_col] == 0)].shape[0]
                    false_alarms = subset[(subset[signal_col] == 0) & (subset[response_col] == 1)].shape[0]
                    correct_rejections = subset[(subset[signal_col] == 0) & (subset[response_col] == 0)].shape[0]

                    # Apply log-linear correction (Hautus, 1995) to avoid zero rates
                    hit_rate = (detect_hits + 0.5) / (detect_hits + detect_misses + 1)
                    false_alarm_rate = (false_alarms + 0.5) / (false_alarms + correct_rejections + 1)

                    # Calculate d' and criterion
                    d_prime = stats.norm.ppf(hit_rate) - stats.norm.ppf(false_alarm_rate)
                    criterion = -0.5 * (stats.norm.ppf(hit_rate) + stats.norm.ppf(false_alarm_rate))

                    # Append the results to the list
                    results.append((study, subject, cond1, cond2, hit_rate, false_alarm_rate, d_prime, criterion))

    # Create a dataframe from the results list
    columns = [
        study_col,
        subject_col,
        condition1_col,
        condition2_col,
        "hit_rate",
        "false_alarm_rate",
        "dprime",
        "criterion",
    ]
    results_df = pd.DataFrame(results, columns=columns)

    return results_df
