# post hoc power analysis
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from statsmodels.stats.power import TTestPower
import pandas as pd

# address reviewer 1 comments

# sample size calculation for paired t-test (pots hoc power analysis)
# Define parameters
cohen_d = 0.84 # Replace with your calculated effect size
alpha = 0.05   # Common significance level
power = 0.8    # Desired power

# Initialize power analysis for paired t-test
analysis = TTestPower()
sample_size = analysis.solve_power(effect_size=cohen_d, alpha=alpha, power=power, alternative='two-sided')

# Round up since sample size must be an integer
sample_size = round(sample_size)

print(f"Required sample size per group: {sample_size}")

# autocorrelation plot
# load behavioural data for both studies

# Load data
path = r"E:\expecon_ms\data\behav\behav_cleaned_for_eeg_expecon1_anonymized.csv"

df = pd.read_csv(path)

# Get unique participant IDs
participant_ids = df["ID"].unique()

# Define lag range
lags = 5
autocorr_matrix = []  # Store autocorrelation values for each participant

# Compute autocorrelation for each participant
for pid in participant_ids:
    participant_data = df[df["ID"] == pid]["sayyes"]
    acf_values = acf(participant_data, nlags=lags, fft=True)[1:lags+1]  # Exclude lag 0
    autocorr_matrix.append(acf_values)

# Convert to numpy array
autocorr_matrix = np.array(autocorr_matrix)

# Compute mean and standard error
mean_acf = np.mean(autocorr_matrix, axis=0)
std_acf = np.std(autocorr_matrix, axis=0) / np.sqrt(len(participant_ids))  # Standard Error of the Mean (SEM)
lag_range = np.arange(1, lags+1)

# Plot mean autocorrelation
plt.figure(figsize=(6, 4), dpi=120)
plt.bar(lag_range, mean_acf, yerr=std_acf, color="royalblue", alpha=0.7, edgecolor="black", width=0.5, capsize=5)

# Formatting
plt.axhline(y=0, linestyle="--", color="gray", linewidth=1)  # Zero line
plt.xlabel("Lag", fontsize=12, fontweight="bold")
plt.ylabel("Mean Autocorrelation", fontsize=12, fontweight="bold")
plt.title("Volatile environment", fontsize=14, fontweight="bold")
plt.xticks(lag_range)  # Show integer lags
plt.grid(axis="y", linestyle="--", alpha=0.6)

# Show plot
plt.show()
