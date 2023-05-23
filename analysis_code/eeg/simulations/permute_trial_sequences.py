import random
import pandas as pd
import numpy as np
from statsmodels.stats.stattools import durbin_watson

def generate_trial_sequence(num_blocks, trials_per_block, trials_per_miniblock, signal_trials_per_miniblock):
    trial_sequence = []

    for block in range(num_blocks):
        miniblock_order = [1] * (trials_per_block // trials_per_miniblock // 2) + [0] * (trials_per_block // trials_per_miniblock // 2)
        random.shuffle(miniblock_order)

        # Randomize the order of miniblocks within the big block
        random.shuffle(miniblock_order)

        for miniblock in range(trials_per_block // trials_per_miniblock):
            if miniblock < trials_per_block // trials_per_miniblock // 2:
                condition = 'High'
                signal_trials = [1] * signal_trials_per_miniblock + [0] * (trials_per_miniblock - signal_trials_per_miniblock)
            else:
                condition = 'Low'
                signal_trials = [1] * (signal_trials_per_miniblock // 3) + [0] * (trials_per_miniblock - (signal_trials_per_miniblock // 3))
            random.shuffle(signal_trials)

            for trial_type in signal_trials:
                trial_sequence.append((block+1, miniblock+1, condition, trial_type))

    trial_df = pd.DataFrame(trial_sequence, columns=['Block', 'Miniblock', 'Condition', 'TrialType'])
    return trial_df

def generate_permuted_sequences(num_permutations, num_blocks, trials_per_block, trials_per_miniblock, signal_trials_per_miniblock):
    permuted_sequences = []

    for _ in range(num_permutations):
        trial_df = generate_trial_sequence(num_blocks, trials_per_block, trials_per_miniblock, signal_trials_per_miniblock)
        permuted_sequences.append(trial_df)
    
    return permuted_sequences

def calculate_avg_conditional_probability(permuted_sequences):
    avg_conditional_prob_per_condition = {}

    for trial_df in permuted_sequences:
        for condition in trial_df['Condition'].unique():
            condition_df = trial_df[trial_df['Condition'] == condition]

            both_signal_count = len(condition_df[(condition_df['TrialType'] == 1) & (condition_df['TrialType'].shift(1) == 1)])
            prev_signal_count = len(condition_df[condition_df['TrialType'].shift(1) == 1])

            conditional_prob = both_signal_count / prev_signal_count

            if condition in avg_conditional_prob_per_condition:
                avg_conditional_prob_per_condition[condition].append(conditional_prob)
            else:
                avg_conditional_prob_per_condition[condition] = [conditional_prob]

    avg_conditional_prob_per_condition = {condition: np.mean(conditional_probs) for condition, conditional_probs in avg_conditional_prob_per_condition.items()}
    return avg_conditional_prob_per_condition

# Example usage
num_permutations = 10000
num_blocks = 5
trials_per_block = 144
trials_per_miniblock = 12
signal_trials_per_miniblock = 9

permuted_sequences = generate_permuted_sequences(num_permutations, num_blocks, trials_per_block, trials_per_miniblock, signal_trials_per_miniblock)

avg_conditional_prob_per_condition = calculate_avg_conditional_probability(permuted_sequences)

# Print the average conditional probability per condition
for condition, avg_conditional_prob in avg_conditional_prob_per_condition.items():
    print(f"Condition: {condition}")
    print(f"Average Conditional Probability: {avg_conditional_prob}")


#########################Autocorrelation####################

def calculate_avg_autocorrelation(permuted_sequences):
    avg_autocorr_per_condition = {}

    for trial_df in permuted_sequences:
        for condition in trial_df['Condition'].unique():
            condition_df = trial_df[trial_df['Condition'] == condition]
            autocorr_value = condition_df['TrialType'].autocorr()

            if condition in avg_autocorr_per_condition:
                avg_autocorr_per_condition[condition].append(autocorr_value)
            else:
                avg_autocorr_per_condition[condition] = [autocorr_value]

    avg_autocorr_per_condition = {condition: np.mean(autocorr_values) for condition, autocorr_values in avg_autocorr_per_condition.items()}
    return avg_autocorr_per_condition

avg_autocorr_per_condition = calculate_avg_autocorrelation(permuted_sequences)

