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
# (We don't have individual MRIs for all subjects, so I use an average brain (not ideal, so let's stay humble))
fs_dir = Path(fetch_fsaverage(verbose=True))
subjects_dir = fs_dir.parent
subject = "fsaverage"

# from Minas code
_oct = "6"  # 6mm between sources

# load files
fwd_dir = subjects_dir / f"{subject}/bem/{subject}-oct{_oct}-fwd.fif"
src_dir = subjects_dir / f"{subject}/bem/{subject}-oct{_oct}-src.fif"

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
