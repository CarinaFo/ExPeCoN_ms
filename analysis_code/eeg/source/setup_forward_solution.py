# prepare files for source reconstruction in mne

# author: Carina Forster
# email: forster@cbs.mpg.de

# import packages
import mne
import os.path as op
from mne.datasets import fetch_fsaverage

# mne includes a freesurfer average brain (I don't have individual MRIs for all subjects, so I use a average brain (not ideal))
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
_oct = '6'  # 6mm between sources

# load files
trans_dir = op.join(subjects_dir, subject, 'bem', subject + '-trans.fif')
bem_sol_dir = op.join(subjects_dir, subject, 'bem', subject + '-5120-5120-5120-bem-sol.fif')
src_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-src.fif')
fwd_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-fwd.fif')
inv_op_dir = op.join(subjects_dir, subject, 'bem', subject + '-oct' + _oct + '-inv.fif')

# 2. ###############################################   source space   ##############################################

# read source space

src = mne.read_source_spaces(src_dir)
src.save('D:\\expecon_ms\\data\\eeg\\source\\fsaverage-6oct-src.fif', 
         overwrite=True)

#########################################    forward solution     #############################################

# read forward solution
fwd = mne.read_forward_solution(fwd_dir)
mne.write_forward_solution('D:\\expecon_ms\\data\\eeg\\source\\fsaverage-6oct-fwd.fif',
                           fwd, overwrite=True)

leadfield = fwd['sol']['data']

print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)