# Plot correlations source_behavior

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path

# Set the font family to Arial
plt.rcParams['font.family'] = 'Arial'

# Set font sizes globally
plt.rcParams['axes.titlesize'] = 20  # Titles on the axes
plt.rcParams['axes.labelsize'] = 16  # X and Y axis labels
plt.rcParams['xtick.labelsize'] = 12  # X-axis ticks
plt.rcParams['ytick.labelsize'] = 12  # Y-axis ticks

# Read the CSV file into a DataFrame
file_path = Path("E:", "expecon_ms", "data", "behav", "source_model_est_2.csv")
df = pd.read_csv(file_path)

# Function to calculate the correlation coefficient and p-value
def calculate_corr(x, y):
    corr_coef, p_value = stats.pearsonr(df[x], df[y])
    return corr_coef, p_value

# Define the plots
plots = [
    ('beta_power_prob', 'crit_change'),
    ('beta_power_prev', 'choice_bias')
]

def plot_correlation():

    # Create a 2x2 grid for subplots
    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    # Loop over the plots and create scatter plots with regression lines
    for i, (x, y) in enumerate(plots):
        col = i % 2
        ax = axes[col]

        # Scatter plot with regression line
        sns.regplot(x=x, y=y, data=df, ax=ax, scatter=True, line_kws={'color': 'red'})

        # Calculate correlation coefficient and p-value
        corr_coef, p_value = calculate_corr(x, y)

        # Add titles with correlation information
        #ax.set_title(f"Pearson r = {corr_coef:.2f}, p-value = {p_value:.2f}")
        delta_sign = "\u0394"
        ax.set_xlabel(f"{delta_sign} pre-stimulus beta power")
        ax.set_ylim(-0.4, 0.4)  # Set x-axis limit from -0.4 to 0.4
        ax.set_xlim(-0.1, 0.2)  # Set x-axis limit from -0.4 to 0.4
        if i == 0:
            ax.set_ylabel(f"{delta_sign} criterion")
            # Set x-axis limits for the first plot
        else:
            ax.set_ylabel("choice bias")


    # Adjust spacing between subplots
    plt.tight_layout()

    extensions = ['png', 'svg']

    for ext in extensions:
        plt.savefig(f"E:/expecon_ms/figs/manuscript_figures/figure7_mediation/source_behavior_correlations.{ext}", dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()