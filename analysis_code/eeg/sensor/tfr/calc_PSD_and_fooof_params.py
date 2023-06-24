################# Calculate PSD per condition and participant ##########################


import os
import sys

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns
import stats
from fooof import FOOOF, FOOOFGroup
import fooof
from fooof.bands import Bands

# add path to sys.path.append() if package isn't found
sys.path.append('D:\\expecon_ms\\analysis_code')

from behav import figure1

IDlist = ('007', '008', '009', '010', '011', '012', '013', '014', '015', '016',
          '017', '018', '019', '020', '021','022', '023', '024', '025', '026',
          '027', '028', '029', '030', '031', '032', '033', '034', '035', '036',
          '037', '038', '039', '040', '041', '042', '043', '044', '045', '046',
          '047', '048', '049')

dir_cleanepochs = "D:\\expecon_ms\\data\\eeg\\prepro_ica\\clean_epochs_corr"
behavpath = 'D:\\expecon_ms\\data\\behav\\behav_df\\'
laplace=0
induced_power=1

psd_all = []

reject_criteria=dict(eeg=200e-6)
flat_criteria=dict(eeg=1e-6)

for idx, subj in enumerate(IDlist):

        # print participant ID
        print('Analyzing ' + subj)

        # load cleaned epochs
        epochs = mne.read_epochs(f"{dir_cleanepochs}"
                                 f"/P{subj}_epochs_after_ica-epo.fif")

        ids_to_delete = [10, 12, 13, 18, 26, 30, 32, 32, 39, 40, 40, 30]
        blocks_to_delete = [6, 6, 4, 3, 4, 3, 2, 3, 3, 2, 5, 6]

        # Check if the participant ID is in the list of IDs to delete
        if pd.unique(epochs.metadata.ID) in ids_to_delete:

            # Get the corresponding blocks to delete for the current participant
            participant_blocks_to_delete = [block for id_, block in
                                            zip(ids_to_delete, blocks_to_delete)
                                            if id_ == pd.unique(epochs.metadata.ID)]
            
            # Drop the rows with the specified blocks from the dataframe
            epochs = epochs[~epochs.metadata.block.isin(participant_blocks_to_delete)]
            
        # remove trials with rts >= 2.5 (no response trials) and trials with rts < 0.1
        epochs = epochs[epochs.metadata.respt1 >= 0.1]
        epochs = epochs[epochs.metadata.respt1 != 2.5]

        if laplace:
            epochs = mne.preprocessing.compute_current_source_density(epochs)

        # load behavioral data
        data = pd.read_csv(f'{behavpath}//prepro_behav_data.csv')

        subj_data = data[data.ID == idx+7]

        # get drop log from epochs
        drop_log = epochs.drop_log

        search_string = 'IGNORED'

        indices = [index for index, tpl in enumerate(drop_log) if tpl and search_string not in tpl]

        # drop bad epochs (too late recordings)
        if indices:
            epochs.metadata = subj_data.reset_index().drop(indices)
        else:
            epochs.metadata = subj_data

        # drop bad epochs
        epochs.drop_bad(reject=reject_criteria, flat=flat_criteria)

        if induced_power:
            # subtract evoked response
            epochs = epochs.subtract_evoked()

        # calculate PSD per epoch (inherits metadata from epochs)
        psd = epochs.compute_psd(fmin=3, fmax=40, tmin=-0.4, tmax=0)

        psd_all.append(psd)

# save freqs
freqs = psd_all[0].freqs

# contrast conditions and average over epochs
psd_a_all = []
psd_b_all = []

for psd in psd_all:
        
        # get high and low probability trials 
        psd_a = psd[((psd.metadata.cue == 0.75))]
        psd_b = psd[((psd.metadata.cue == 0.25))]

        psd_a = psd[((psd.metadata.isyes == 1) & (psd.metadata.sayyes == 1))]
        psd_b = psd[((psd.metadata.isyes == 1) & (psd.metadata.sayyes == 0))]

        evoked_psd_a = psd_a.average()
        evoked_psd_b = psd_b.average()

        psd_a_all.append(evoked_psd_a)
        psd_b_all.append(evoked_psd_b)

# prepare for fooof (channel = C4)
psd_a = np.array([np.squeeze(psd[24,:]) for psd in psd_a_all])
psd_b = np.array([np.squeeze(psd[24,:]) for psd in psd_b_all])

# plot an example PSD

plt.plot(freqs, np.log10(psd_a[25]), label='Hit')
plt.plot(freqs, np.log10(psd_b[25]), label= 'Miss')
plt.legend()
plt.savefig('D:\\expecon_ms\\figs\\manuscript_figures\\Figure5\\example_psd.svg',
             dpi=300)

def calc_fooof(psd_array=psd_a, fmin=3, fmax=40, freqs=freqs):
    
    # Initialize a FOOOFGroup object, specifying some parameters
    fg = FOOOFGroup(peak_width_limits=[1, 8], min_peak_height=0.05)

    # Fit FOOOF model across the matrix of power spectra
    # pick C4 as channel of interest

    fg.fit(freqs, psd_array, [fmin, fmax])

    return fg

fg_dict = {'Hit': fg_hit, 'Miss': fg_miss}

off_all, exps_all, peaks_all = [], [],[]
cfs_all, errors_all, r2s_all = [], [], []

for  keys, fg in fg_dict.items():

    fg.print_results()

    fg.plot()

    # Extract aperiodic parameters
    off_all.append(fg.get_params('aperiodic_params', 'offset'))
    exps_all.append(fg.get_params('aperiodic_params', 'exponent'))

    # Extract peak parameters
    peaks_all.append(fg.get_params('peak_params'))
    cfs_all.append(fg.get_params('peak_params', 'CF'))

    # Extract goodness-of-fit metrics
    errors_all.append(fg.get_params('error'))
    r2s_all.append(fg.get_params('r_squared'))

    # Create and save out a report summarizing the results across the group of power spectra
    fg.save_report(file_name=f'D:\expecon_ms\\figs\manuscript_figures\Figure5\\fooof_group_results_{keys}.svg')

    # Save out FOOOF results for further analysis later
    fg.save(file_name=f'D:\expecon_ms\\figs\manuscript_figures\Figure5\\fooof_group_results_{keys}.csv', save_results=True)

# extract participant fooof
fm = fg_hit.get_fooof(ind=20, regenerate=True)
fm.plot(save_fig=True, file_name=f'D:\expecon_ms\\figs\manuscript_figures\Figure5\\fooof_example_participant.svg')

# compare R2 between conditions

scipy.stats.wilcoxon(r2s_all[0], r2s_all[1]) # n.s.
scipy.stats.wilcoxon(off_all[0], off_all[1]) # n.s.
scipy.stats.wilcoxon(exps_all[0], exps_all[1]) # n.s.

# extract alpha and beta peaks per participant and condition
# Import the Bands object, which is used to define frequency bands

peaks = np.empty((0, 3))
for f_res in fg_hit:  
    peaks = np.vstack((peaks, get_band_peak(f_res.peak_params, (17,35), select_highest=True)))

# Create a boolean mask indicating NaN values
nan_mask = np.isnan(peaks_high_beta[:,0])

print("Participants without peak:", sum(nan_mask))

# Drop NaN values from the array
fpeaks_high = peaks_high_beta[:,0][~nan_mask]
fpeaks_low = peaks_low_beta[:,0][~nan_mask]

# Extract power per peak
power_high = peaks_high_beta[:,1][~nan_mask]
power_low = peaks_low_beta[:,1][~nan_mask]

# compare alpha peak frequencies and power
scipy.stats.wilcoxon(fpeaks_high, fpeaks_low, nan_policy='omit') # n.s.
scipy.stats.wilcoxon(power_high, power_low, nan_policy='omit') # n.s.


# subtract 1/f slope and offset from power spectra


def extract_aperiodic(psd=None):

    """extract aperiodic parameters from fooof object (exponent and intercept) """
    # Compute aperiodic component for one PSD

    aperiodic_psd = []

    for offset, slope in zip(off_all[1], exps_all[1]):
        aperiodic_psd.append(offset - np.log10(freqs ** slope))

    aperiodic = np.array(aperiodic_psd)

    #now subtract aperiodic component from full PSD

    periodic = np.log10(psd)-aperiodic

    #plot 3 random flat PSDs

    random_indices = np.random.randint(42, size=3)

    for indices in random_indices:

        plt.plot(freqs, aperiodic[indices,])
        plt.show()

    for indices in random_indices:

        plt.plot(freqs, periodic[indices,])
        plt.show()

    return aperiodic, periodic


# Assuming the residual PSDs after subtracting 1/f are stored in the 'fooof_results' list
def compare_periodic_power(a_per=None, b_per=None, freqs=None):

    # Step 1: Define the frequency ranges for alpha and beta bands
    alpha_band = (8, 13)  # Alpha frequency range (7-13 Hz)
    beta_band = (18, 22)  # Beta frequency range (18-27 Hz)

    # Step 2: Calculate power in the specified frequency bands
    alpha_power_low = []
    beta_power_low = []
    alpha_power_high = []
    beta_power_high = []

    for l,h in zip(a_per, b_per):
        # Calculate power in alpha band
        alpha_indices = np.logical_and(freqs >= alpha_band[0], freqs <= alpha_band[1])
        alpha_power_low.append(np.trapz(l[alpha_indices], freqs[alpha_indices]))
        alpha_power_high.append(np.trapz(h[alpha_indices], freqs[alpha_indices]))

        # Calculate power in beta band
        beta_indices = np.logical_and(freqs >= beta_band[0], freqs <= beta_band[1])
        beta_power_low.append(np.trapz(l[beta_indices], freqs[beta_indices]))
        beta_power_high.append(np.trapz(h[beta_indices], freqs[beta_indices]))

    # Step 3: Calculate alpha/beta power ratio

    alpha_beta_ratio_low = np.array(alpha_power_low) / np.array(beta_power_low)
    alpha_beta_ratio_high = np.array(alpha_power_high) / np.array(beta_power_high)

    # check significance of alpha/beta ratio difference

    scipy.stats.wilcoxon(alpha_beta_ratio_low, alpha_beta_ratio_high) # n.s.
    scipy.stats.wilcoxon(alpha_power_low, alpha_power_high) # n.s.
    scipy.stats.wilcoxon(beta_power_low, beta_power_high) # n.s.
