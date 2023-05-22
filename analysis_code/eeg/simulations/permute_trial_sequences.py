import random
import pandas as pd

def generate_trial_sequence(num_blocks, trials_per_block, trials_per_miniblock, signal_trials_per_miniblock):
    trial_sequence = []

    for block in range(num_blocks):
        miniblock_order = [1] * (trials_per_block // trials_per_miniblock // 2) + [0] * (trials_per_block // trials_per_miniblock // 2)
        random.shuffle(miniblock_order)

        for miniblock in range(trials_per_block // trials_per_miniblock):
            signal_trials = [1] * signal_trials_per_miniblock + [0] * (trials_per_miniblock - signal_trials_per_miniblock)
            random.shuffle(signal_trials)

            for trial_type in signal_trials:
                trial_sequence.append((block+1, miniblock+1, trial_type))

    trial_df = pd.DataFrame(trial_sequence, columns=['Block', 'Miniblock', 'TrialType'])
    return trial_df


def generate_permuted_sequences(num_permutations, num_blocks, trials_per_block, trials_per_miniblock, signal_trials_per_miniblock):
    permuted_sequences = []

    for _ in range(num_permutations):
        trial_df = generate_trial_sequence(num_blocks, trials_per_block, trials_per_miniblock, signal_trials_per_miniblock)
        permuted_sequences.append(trial_df)
    
    return permuted_sequences


# Example usage
num_permutations = 10000
num_blocks = 5
trials_per_block = 144
trials_per_miniblock = 12
signal_trials_per_miniblock = 3

permuted_sequences = generate_permuted_sequences(num_permutations, num_blocks, trials_per_block, trials_per_miniblock, signal_trials_per_miniblock)

# Print the first 10 trial sequences
for i, trial_df in enumerate(permuted_sequences[:10]):
    print(f"Permutation {i+1}:")
    print(trial_df)
    print()

