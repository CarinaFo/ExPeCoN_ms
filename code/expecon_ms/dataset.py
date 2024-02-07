#!/usr/bin/python3
"""
Download study data of the ExPeCoN study from OSF.

Author: Simon M. Hofmann
Contact: simon.hofmann[at]pm.me
Years: 2023
"""

# %% Import
from __future__ import annotations

from pathlib import Path

from expecon_ms.configs import paths

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass


# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def get_expecon_data(data_dir: str | Path | None = None):
    """Download study data of the ExPeCoN study from OSF."""
    if data_dir is None:
        data_dir = Path(paths.data)
    # TODO: implement
    raise NotImplementedError("get_expecon_data is not implemented yet.")


# %% __main__  >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o

if __name__ == "__main__":
    pass

# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
