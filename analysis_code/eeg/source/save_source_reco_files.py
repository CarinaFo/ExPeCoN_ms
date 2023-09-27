
import mne
import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.minimum_norm import write_inverse_operator
from mne.minimum_norm import read_inverse_operator
from mne.minimum_norm import apply_inverse_epochs

# mne includes a freesurfer average brain (I don't have individual MRIs for all subjects)

fs_dir = fetch_fsaverage(verbose=True)
subjects_dir = op.dirname(fs_dir)
subject = 'fsaverage'

# this extracts a certain brain area only

label_S1 = 'rh.BA3a'
fname_labelS1 = subjects_dir + '\\fsaverage\\label\\%s.label' % label_S1
labelS1 = mne.read_label(fname_labelS1)
label_S2 = 'rh.BA3b'
fname_labelS2 = subjects_dir + '\\fsaverage\\label\\%s.label' % label_S2
labelS2 = mne.read_label(fname_labelS2)

label_aparc = 'rh.aparc'
fname_labelaparc = subjects_dir + '\\fsaverage\\label\\%s.label' % label_aparc
labelap = mne.read_label(fname_labelaparc)

# from Minas code

_oct = '6'

trans_dir = op.join(subjects_dir, subject, 'bem', subject + '-trans.fif')
bem_sol_dir = op.join(subjects_dir, subject, 'bem', subject + '-5120-5120-5120-bem-sol.fif')
src_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')
inv_op_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-inv.fif')

# 2. ###############################################   source space   ##############################################

# read source space

src = mne.read_source_spaces(src_dir)
src.save('D:\\expecon_ms\\data\\eeg\\source\\fsaverage-6oct-src.fif', overwrite=True)

# 3. #########################################    forward solution     #############################################

# read forward solution

fwd = mne.read_forward_solution(fwd_dir)
mne.write_forward_solution('D:\\expecon_ms\\data\\eeg\\source\\fsaverage-6oct-fwd.fif',
                           fwd, overwrite=True)

leadfield = fwd['sol']['data']

print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

# 6. #######################################    Noise Covariance ##################################################

# Q: which fwd to use for EEG?

# prepare noise covariance (only works with epochs)
# should be computed over baseline time window

os.chdir("D:\expecon_ms\data\eeg\prepro_ica\clean_epochs")

epochs = mne.read_epochs('P035_epochs_after_ica-epo.fif')

noise_cov_reg = mne.compute_covariance(
    epochs, tmin=-1, tmax=-0.6, method='auto', rank=None, verbose=True)

noise_cov_reg.save('D:\\expecon_ms\\data\\eeg\\source\\noise_cov_reg.fif', overwrite=True)

inv_op = mne.minimum_norm.make_inverse_operator(epochs.info, fwd, noise_cov_reg, loose=0.2, depth=0.8)

mne.minimum_norm.write_inverse_operator('D:\\expecon_ms\\data\\eeg\\source\\fsaverage-6oct-inv.fif', inv_op)