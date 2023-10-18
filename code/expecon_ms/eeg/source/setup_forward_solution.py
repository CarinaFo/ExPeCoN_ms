#!/usr/bin/python3
"""
Prepare files for source reconstruction in MNE.

Author: Carina Forster
Contact: forster@cbs.mpg.de
Years: 2023
"""
# %% Import
from pathlib import Path

import mne
from mne.datasets import fetch_fsaverage

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

# MNE includes a freesurfer average brain
# (We don't have individual MRIs for all subjects, so I use an average brain (not ideal))
fs_dir = Path(fetch_fsaverage(verbose=True))
subjects_dir = fs_dir.parent
subject = "fsaverage"

# this extracts a certain brain area only
label_s1 = "rh.BA3a"
fname_labels1 = subjects_dir / f"fsaverage/label/{label_s1}.label"
labels1 = mne.read_label(str(fname_labels1))
label_s2 = "rh.BA3b"
fname_labels2 = subjects_dir / f"fsaverage/label/{label_s2}.label"
labels2 = mne.read_label(str(fname_labels2))  # TODO(simon): this is not used

label_aparc = "rh.aparc"
fname_label_aparc = subjects_dir / f"fsaverage/label/{label_aparc}.label"
label_ap = mne.read_label(str(fname_label_aparc))  # TODO(simon): this is not used

# from Minas code
_oct = "6"  # 6mm between sources

# load files  # TODO(simon): some of the paths are not used
trans_dir = subjects_dir / f"{subject}/bem/{subject}-trans.fif"
fwd_dir = subjects_dir / f"{subject}/bem/{subject}-oct{_oct}-fwd.fif"
bem_sol_dir = subjects_dir / f"{subject}/bem/{subject}-5120-5120-5120-bem-sol.fif"
src_dir = subjects_dir / f"{subject}/bem/{subject}-oct{_oct}-src.fif"
inv_op_dir = subjects_dir / f"{subject}/bem/{subject}-oct{_oct}-inv.fif"


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":

    # %% Source space >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

    # read the source space
    src = mne.read_source_spaces(src_dir)
    src.save(fname=Path("./data/eeg/source/fsaverage-6oct-src.fif"), overwrite=True)

    # %% Forward solution  o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

    # read the forward solution
    fwd = mne.read_forward_solution(fwd_dir)
    mne.write_forward_solution(fname=Path("./data/eeg/source/fsaverage-6oct-fwd.fif"), fwd=fwd, overwrite=True)

    lead_field = fwd["sol"]["data"]

    print("Lead field size : {} sensors x {} dipoles".format(*lead_field.shape))

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
