################# Calculate PSD per condition and participant ##########################


import os
import numpy as np
import pandas as pd
import mne
from fooof import FOOOF
from fooof import FOOOF, FOOOFGroup
from fooof.bands import Bands

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
        psd = epochs.compute_psd(fmin=3, fmax=40, tmin=-0.5, tmax=0)

        psd_all.append(psd)

psd_a_all = []
psd_b_all = []

for psd in psd_all:
        
        # get high and low probability trials 
        psd_a = psd[((psd.metadata.cue == 0.75))]
        psd_b = psd[((psd.metadata.cue == 0.25))]

        evoked_psd_a = psd_a.average()
        evoked_psd_b = psd_b.average()

        psd_a_all.append(evoked_psd_a)
        psd_b_all.append(evoked_psd_b)

# prepare for fooof (channel = C4)
psd_a = np.array([np.squeeze(psd[24,:]) for psd in psd_a_all])
psd_b = np.array([np.squeeze(psd[24,:]) for psd in psd_b_all])

plt.plot(psd_all[0].freqs, np.log10(psd_a[25]), label='0.75 prob')
plt.plot(psd_all[0].freqs, np.log10(psd_b[25]), label= '0.25 prob')
plt.legend()


def calc_fooof(psd_array=psd_a):
    
    # Initialize a FOOOFGroup object, specifying some parameters
    fg = FOOOFGroup(peak_width_limits=[1, 8], min_peak_height=0.05)

    # Fit FOOOF model across the matrix of power spectra
    # pick C4 as channel of interest

    fg.fit(psd_a_all[0].freqs, psd_array, [3, 40])

    return fg

fg_dict = {'0.75_prob': fg_high, '0.25_prob': fg_low}

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
fm = fg_all[0].get_fooof(ind=20, regenerate=True)

# compare R2 between conditions

scipy.stats.wilcoxon(r2s_all[0], r2s_all[1]) # n.s.
scipy.stats.wilcoxon(off_all[0], off_all[1]) # n.s.
scipy.stats.wilcoxon(exps_all[0], exps_all[1]) # n.s.
# extract alpha and beta peaks per participant and condition
# Import the Bands object, which is used to define frequency bands


peaks = np.empty((0, 3))
for f_res in fg_low:  
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

