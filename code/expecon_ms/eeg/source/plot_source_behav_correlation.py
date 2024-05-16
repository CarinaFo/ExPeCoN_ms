# Plot correlations source_behavior
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import statsmodels.api as sm

# Set the font family to Arial
plt.rcParams['font.family'] = 'Arial'

# Set font sizes globally
plt.rcParams['axes.titlesize'] = 20  # Titles on the axes
plt.rcParams['axes.labelsize'] = 16  # X and Y axis labels
plt.rcParams['xtick.labelsize'] = 12  # X-axis ticks
plt.rcParams['ytick.labelsize'] = 12  # Y-axis ticks


def plot_correlation():

    """Plot the correlation between pre-stimulus beta power and behavior"""
    
    # Read the CSV file into a DataFrame
    file_path1 = Path("E:\\expecon_ms", "data", "behav", f"source_model_est_1.csv")
    file_path2 = Path("E:\\expecon_ms", "data", "behav", f"source_model_est_2.csv")
    df1 = pd.read_csv(file_path1)
    df2 = pd.read_csv(file_path2)

    # Define the pairs of columns to plot
    column_pairs = [
        (df1, 'beta_power_prob', 'interaction'),
        (df1, 'beta_power_prev', 'interaction'),
        (df2, 'beta_power_prob', 'crit_change'),
        (df2, 'beta_power_prev', 'choice_bias')
    ]

    # Create the plots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

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

        p_value_text = 'P-val: {:.4f}'.format(float(p_value))

        # Add the coefficient to the plot
        ax.text(0.05, 0.95, f'Coef: {coef:.2f}\n{p_value_text}', 
                transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Adjust layout
    plt.tight_layout()

    extensions = ['png', 'svg']

    for ext in extensions:
        plt.savefig(f"E:/expecon_ms/figs/manuscript_figures/figure6_correlation_mediation/source_behavior_correlations.{ext}", dpi=300, bbox_inches='tight')

    # Display the plot
    plt.show()