# Plot correlations source_behavior
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# Set the font family to Arial
plt.rcParams["font.family"] = "Arial"

# Set font sizes globally
plt.rcParams["axes.labelsize"] = 14  # X and Y axis labels
plt.rcParams["xtick.labelsize"] = 12  # X-axis ticks
plt.rcParams["ytick.labelsize"] = 12  # Y-axis ticks


def plot_correlation():
    """Plot the correlation between pre-stimulus beta power and behavior."""
    # Read the CSV file into a DataFrame
    file_path1 = Path("E:\\expecon_ms", "data", "behav", "source_model_est_1.csv")
    file_path2 = Path("E:\\expecon_ms", "data", "behav", "source_model_est_2.csv")
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # Define the pairs of columns to plot
    column_pairs = [
        (df1, "beta_power_prob", "crit_change"),
        (df2, "beta_power_prob", "crit_change"),
        (df1, "beta_power_prev", "choice_bias"),
        (df2, "beta_power_prev", "choice_bias"),
    ]

    # Create the plots
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))

    # Flatten the axes array
    axes = axes.flatten()

    for ax, (df, x_col, y_col) in zip(axes, column_pairs):
        # Plot the regression plot
        sns.regplot(x=x_col, y=y_col, data=df, ax=ax)

        # Fit a linear model using statsmodels to get the regression coefficient
        X = sm.add_constant(df[x_col])  # Adds a constant term to the predictor
        model = sm.OLS(df[y_col], X).fit()
        coef = model.params[x_col]
        p_value = model.pvalues[x_col]
        # Determine p-value text
        # Determine p-value text

        p_value_text = f"P-val: {float(p_value):.4f}"

        # Add the coefficient to the plot
        ax.text(
            0.05,
            0.95,
            f"Coef: {coef:.2f}\n{p_value_text}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
        )

        # Customize the plot to remove grid lines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_visible(True)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")
        ax.grid(False)  # Remove grid lines

        # Customize x-axis labels
        if x_col == "beta_power_prev":
            ax.set_xlabel("Δ Previous Response Beta Power")
        elif x_col == "beta_power_prob":
            ax.set_xlabel("Δ Probability Beta Power")

        if y_col == "interaction":
            ax.set_ylabel("interaction estimate")
        elif y_col == "crit_change":
            ax.set_ylabel("probability estimate")
        elif y_col == "choice_bias":
            ax.set_ylabel("previous choice estimate")

    # Adjust layout
    plt.tight_layout()

    extensions = ["png", "svg"]

    for ext in extensions:
        plt.savefig(
            f"E:/expecon_ms/figs/manuscript_figures/figure6_correlation_mediation/source_behavior_correlations_control.{ext}",
            dpi=300,
            bbox_inches="tight",
        )

    # Display the plot
    plt.show()
