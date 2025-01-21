"""
The script contains functions to analyze and plot the behavioral data for the ExPeCoN study.
The script reproduces the interaction between stimulus probability and the previous response
on the criterion in a signal detection theory (SDT) framework. The script also plots the response
time descriptives for the ExPeCoN study.

Also, it produces suppl. fig. 2

The ExPeCoN study investigates stimulus probabilities and the influence on perception and confidence in a
near-threshold somatosensory detection task in two paradigms that vary in their probability environment:

dataset 1 : stable environment, 144 trials in 5 blocks, 43 participants,
           stimulus probability is cued before blocks of 12 trials.
dataset 2: variable environment, 120 trials in 5 blocks, 40 participants,
           stimulus probability is cued before each trial.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2024-2025
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
                posthoc_interaction = pg.pairwise_tests(
                    data=subset_prev, dv='criterion', within='cue', subject='ID', padjust='bonf'
                ).round(3)
                print(f"Post hoc for Previous Response = {response}:\n", posthoc_interaction)

            # Example: Compare Prev_Response ('Yes' vs. 'No') within each level of Stim_Prob
            for prob in subset['cue'].unique():
                subset_prob = subset[subset['cue'] == prob]
                posthoc_interaction = pg.pairwise_tests(
                    data=subset_prob, dv='criterion', within='prevresp', subject='ID', padjust='bonf'
                ).round(3)
                print(f"Post hoc for Stimulus Probability = {prob}:\n", posthoc_interaction)

        else:

            # Pairwise tests for Stim_Prob
            posthoc_stim = pg.pairwise_tests(
                data=subset, dv='criterion', within='cue', subject='ID', padjust='bonf'
            ).round(3)
            print(posthoc_stim)

            # Pairwise tests for Prev_Response
            posthoc_prev = pg.pairwise_tests(
                data=subset, dv='criterion', within='prevresp', subject='ID', padjust='bonf'
            ).round(3)
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
                posthoc_interaction = pg.pairwise_tests(
                    data=subset_prev, dv='dprime', within='cue', subject='ID', padjust='bonf'
                ).round(3)
                print(f"Post hoc for Previous Response = {response}:\n", posthoc_interaction)

            # Example: Compare Prev_Response ('Yes' vs. 'No') within each level of Stim_Prob
            for prob in subset['cue'].unique():
                subset_prob = subset[subset['cue'] == prob]
                posthoc_interaction = pg.pairwise_tests(
                    data=subset_prob, dv='dprime', within='prevresp', subject='ID', padjust='bonf'
                ).round(3)
                print(f"Post hoc for Stimulus Probability = {prob}:\n", posthoc_interaction)

        else:

            # Pairwise tests for Stim_Prob
            posthoc_stim = pg.pairwise_tests(
                data=subset, dv='dprime', within='cue', subject='ID', padjust='bonf'
            ).round(3)
            print(posthoc_stim)

            # Pairwise tests for Prev_Response
            posthoc_prev = pg.pairwise_tests(
                data=subset, dv='dprime', within='prevresp', subject='ID', padjust='bonf'
            ).round(3)
            print(posthoc_prev)


def plot_response_times_behaviour_descriptives():
    """ Plot response time descriptives. """

    # custom color palette
    base_palette = sns.color_palette("colorblind", 5)  # 5 colors for the plot

    sns.set_palette(sns.color_palette(base_palette))
    
    # Set the path to save the figures
    save_dir = r"C:\Users\Carina\Desktop\PhD_thesis\figures_for_thesis"

    # Load the clean data for both studies
    df = pd.read_csv(Path(paths.data.behavior, "prepro_behav_data_expecon1_2.csv"))

    # drop columns with only NaNs
    df = df.dropna(axis=1, how="all")

    # remove rows with missing values
    df = df.dropna()
    # rename respt1 to rt
    df = df.rename(columns={"respt1": "rt"})

     # Plot response time distribution for each study and mark the median response time
    for study in [1, 2]:
        study_data = df[df["study"] == study]
        median_rt = study_data["rt"].median()
        plt.figure(figsize=(10, 6))
        sns.histplot(study_data["rt"], kde=True, color=base_palette[0])
        plt.axvline(median_rt, color="red", linestyle="--", label=f"Median RT: {median_rt:.2f} s")
        if study == 1:
            study_name = "Stable Environment"
        else:
            study_name = "Volatile Environment"
        plt.title(f"Response Time Distribution for {study_name}")
        plt.xlabel("Response Time (s)")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(save_dir + f"/rt_distribution_study{study}.png")

    # rename columns: from correct to accuracy
    df = df.rename(columns={"correct": "accuracy"})

    # Split data into correct and incorrect trials
    df_correct = df[df["accuracy"] == True]
    df_incorrect = df[df["accuracy"] == False]

    # Split data into signal detection theory (SDT) categories
    df['hits'] = (df["isyes"] == 1) & (df['accuracy'] == 1)
    df['crs'] = (df["isyes"] == 0) & (df['accuracy'] == 1)
    df['fas'] = (df["isyes"] == 0) & (df['accuracy'] == 0)
    df['miss'] = (df["isyes"] == 1) & (df['accuracy'] == 0)

    # Extract correct signal trials with high confidence
    df_correct_signal = df[(df["accuracy"] == True) & (df["isyes"] == 1) & (df['conf'] == 1)]

    # group data based on accuracy or confidence
    rt_acc = df.groupby(["study", "ID", "accuracy"])["rt"].mean().reset_index()
    rt_conf = df_correct.groupby(["study", "ID", "conf"])["rt"].mean().reset_index()
    rt_conf_incorrect = df_incorrect.groupby(["study", "ID", "conf"])["rt"].mean().reset_index()
    # Calculate mean response time for congruent and incongruent trials
    rt_congruent = df_correct_signal.groupby(["study", "ID", "congruency"])["rt"].mean().reset_index()
    rt_stimcongruent = df_correct_signal.groupby(["study", "ID", "congruency_stim"])["rt"].mean().reset_index()

    # plot boxplots for each grouping
    plot_boxplot(rt_acc, "Response Time by Study and Accuracy", "accuracy")
    plot_boxplot(rt_conf, "Response Time by Study and Confidence (Correct Trials)", "conf")
    plot_boxplot(rt_conf_incorrect, "Response Time by Study and Confidence (Incorrect Trials)", "conf")
    plot_boxplot(rt_congruent, "Response Time by Study and Congruency (Correct, High confidence signal trials)", "congruency")
    plot_boxplot(rt_stimcongruent, "Response Time by Study and Stimulus Congruency (Correct, High confidence signal trials)", "congruency_stim")
    

    # Plot response time for hits, crs, fas, and misses for each study
    for s in [1, 2]:
        sdt_hit = df[(df["study"] == s) & (df['hits'] == 1)].groupby("ID")['rt'].mean().reset_index()
        sdt_cr = df[(df["study"] == s) & (df['crs'] == 1)].groupby("ID")['rt'].mean().reset_index()
        sdt_fa = df[(df["study"] == s) & (df['fas'] == 1)].groupby("ID")['rt'].mean().reset_index()
        sdt_miss = df[(df["study"] == s) & (df['miss'] == 1)].groupby("ID")['rt'].mean().reset_index()
        sdt_all = pd.DataFrame({"ID": sdt_hit["ID"], "hit": sdt_hit["rt"], "correct rejection": sdt_cr["rt"], "false alarm": sdt_fa["rt"], "miss": sdt_miss["rt"]})
        # long format for plotting
        sdt_df_melt = sdt_all.melt(id_vars="ID", var_name="trial_type", value_name="response time (s)")

        plt.figure(figsize=(16, 6))
        sns.violinplot(x='trial_type', y='response time (s)', hue="trial_type", data=sdt_df_melt)
        if s == 1:
            study_name = "Stable Environment"
        else:
            study_name = "Volatile Environment"
        plt.title(f"{study_name}", fontdict={'fontsize': 16})
        plt.ylabel("Response Time (s)", fontdict={'fontsize': 14})
        plt.xlabel("Trial Type", fontdict={'fontsize': 14})
        # Save the plot
        plot_path = os.path.join(save_dir, f'sdt_rts_{s}.png')
        plt.savefig(plot_path)
        plt.show()
    
    # Stats
      # run a non parametric t-test to compare the response times between congruent and incongruent trials
    # use the pingouin package
    import pingouin as pg

    for study in [1, 2]:
        # test for differences in accuracy
        subset = rt_acc[rt_acc["study"] == study]
        correct = subset[subset["accuracy"] == True]["rt"]
        incorrect = subset[subset["accuracy"] == False]["rt"]
        # print the results
        print(f"Study {study}:")
        print("Accuracy contrast")
        print(pg.ttest(correct, incorrect, paired=True))

        subset = rt_congruent[rt_congruent["study"] == study]
        congruent = subset[subset["congruency"] == True]["rt"]
        incongruent = subset[subset["congruency"] == False]["rt"]
        # print the results
        print(f"Study {study}:")
        print("Congruency contrast")
        print(pg.ttest(congruent, incongruent, paired=True))
        
        # test for difference between hits and correct rejections
        sdt_hit = df[(df["study"] == s) & (df['hits'] == 1)].groupby("ID")['rt'].mean().reset_index()
        sdt_cr = df[(df["study"] == s) & (df['crs'] == 1)].groupby("ID")['rt'].mean().reset_index()
        sdt_fa = df[(df["study"] == s) & (df['fas'] == 1)].groupby("ID")['rt'].mean().reset_index()
        sdt_miss = df[(df["study"] == s) & (df['miss'] == 1)].groupby("ID")['rt'].mean().reset_index()

        # print the results
        print(f"Study {study}:")
        print("Hits vs. Correct Rejections")
        print(pg.ttest(sdt_hit['rt'], sdt_cr['rt'], paired=True))

        # test for difference between miss and false alarms
        # print the results
        print(f"Study {study}:")
        print("Miss vs. False Alarm")
        if s == 2:
        # drop rows where ID is 62 or 75 or 83 from sdt_miss 
            sdt_miss = sdt_miss[~sdt_miss["ID"].isin([62, 75, 83])] # no false alarms
        print(pg.ttest(sdt_miss['rt'], sdt_fa['rt'], paired=True))


def plot_boxplot(data, title, hue):
    
    plt.figure(figsize=(10, 6))

    # use violin plot
    sns.violinplot(x="study", y="rt", hue=hue, data=data)
    plt.xticks(ticks=[0, 1], labels=["Stable Environment", "Volatile Environment"])
    plt.ylabel("Response Time (s)")

    # Custom legend labels
    if hue == "ecg_phase":
        legend_labels = {0: 'Diastole', 1: 'Systole'}
    elif hue == "rsp_phase":
        legend_labels = {0: 'Expiration', 1: 'Inspiration'}
    elif hue == "accuracy":
        legend_labels = {'False': 'Incorrect', 'True': 'Correct'}
    elif hue == "congruency" or hue == "congruency_stim":
        legend_labels = {'False': 'Incongruent', 'True': 'Congruent'}
    elif hue == "conf":
        legend_labels = {0: 'Low Confidence', 1: 'High Confidence'}

    # Get the handles and labels from the current legend
    handles, labels = plt.gca().get_legend_handles_labels()

    if hue == "ecg_phase" or hue == "rsp_phase"  or hue == "conf":
        # Replace labels with custom labels
        labels = [legend_labels[int(float(label))] for label in labels]
    else:
        labels = [legend_labels[label] for label in labels]

    # Create a new legend with the custom labels
    plt.legend(handles, labels, title=hue)

    plot_path = os.path.join(save_dir, f'{hue}_rt.png')
    plt.savefig(plot_path)
    plt.show()


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
