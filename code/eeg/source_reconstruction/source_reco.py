import os.path as op
import os
import numpy as np
import mne
from mne.datasets import fetch_fsaverage
from mne.time_frequency import csd_morlet
from mne.beamformer import make_dics, apply_dics_csd

#mne.viz.set_3d_backend("notebook")
print(__doc__)

# Reading the raw data and creating epochs:

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'

_oct = '6'

fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')

fwd = mne.read_forward_solution(fwd_dir)

os.chdir("D:\expecon_ms\data\eeg\prepro_ica\clean_epochs")

epochs = mne.read_epochs('P038_epochs_after_ica-epo.fif')

epochs_high = epochs[epochs.metadata.cue == 0.75]
epochs_low = epochs[epochs.metadata.cue == 0.25]

evokeds_high = epochs_high.average().crop(-0.5,0)
evokeds_low = epochs_low.average().crop(-0.5,0)

# We are interested in the beta band. Define a range of frequencies, using a
# log scale, from 12 to 30 Hz.

freqs = np.logspace(np.log10(15), np.log10(30), 9)

# Computing the cross-spectral density matrix for the beta frequency band, for
# different time intervals.
csd = csd_morlet(epochs, freqs, tmin=-1, tmax=1)
csd_a = csd_morlet(epochs_high, freqs, tmin=-0.5, tmax=0)
csd_b = csd_morlet(epochs_low, freqs, tmin=-0.5, tmax=0)
csd_baseline = csd_morlet(epochs, freqs, tmin=-1, tmax=-0.5)

info = epochs.info

# To compute the source power for a frequency band, rather than each frequency
# separately, we average the CSD objects across frequencies.
csd_a = csd_a.mean()
csd_b = csd_b.mean()
csd_baseline = csd_baseline.mean()

# Computing DICS spatial filters using the CSD that was computed on the entire
# timecourse.

filters = make_dics(info, fwd, csd, noise_csd=csd_baseline,
                    pick_ori='max-power', reduce_rank=True, real_filter=True)

# Applying DICS spatial filters separately to the CSD computed using the
# baseline and the CSD computed during the ERS activity.

source_power_a, freqs = apply_dics_csd(csd_a, filters)
source_power_b, freqs = apply_dics_csd(csd_b, filters)

noise_cov = mne.compute_covariance(
        epochs, tmin=-1, tmax=-0.5, method='auto', rank=None, verbose=True)

inv_op = mne.minimum_norm.make_inverse_operator(evokeds_high.info, fwd, noise_cov,
                                                    loose=1.0, fixed=False)

evokeds_high.set_eeg_reference(projection=True)  # needed for inverse modeling

conditiona_stc = mne.minimum_norm.apply_inverse(evokeds_high, inv_op, lambda2=0.05,
                                                                             method='eLORETA', pick_ori='normal')

inv_op = mne.minimum_norm.make_inverse_operator(evokeds_low.info, fwd, noise_cov,
                                                    loose=1.0, fixed=False)

evokeds_low.set_eeg_reference(projection=True)  # needed for inverse modeling

conditionb_stc = mne.minimum_norm.apply_inverse(evokeds_low, inv_op, lambda2=0.05,
                                                                             method='eLORETA', pick_ori='normal')

# Visualizing source power during ERS activity relative to the baseline power.

stc = source_power_a/source_power_b

stc_eLoreta = conditiona_stc / conditionb_stc

message = 'DICS source power in the 12-30 Hz frequency band'

brain = stc.plot(hemi='rh', views='axial', subjects_dir=subjects_dir,
                 subject=subject, time_label=message)


# References
# ----------
# .. footbibliography::
