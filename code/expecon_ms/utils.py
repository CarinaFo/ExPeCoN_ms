#!/usr/bin/python3
"""
utils for expecon_ms.

Author: Carina Forster
Years: 2023
"""

# %% Import
import numpy as np

# %% Set global vars & paths >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o
pass

# %% Functions >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o


def zero_pad_or_mirror_data(data, zero_pad: bool):
    """
    Zero-pad or mirror data on both sides to avoid edge artifacts.

    Args:
    ----
    data: data array with the structure epochs x channels x time
    zero_pad: boolean, info: whether to zero pad or mirror the data

    Returns:
    -------
    array with mirrored data on both ends = data.shape[2]

    """
    if zero_pad:
        padded_list = []

        zero_pad = np.zeros(data.shape[2])
        # loop over epochs
        for epoch in range(data.shape[0]):
            ch_list = []
            # loop over channels
            for ch in range(data.shape[1]):
                # zero padded data at beginning and end
                ch_list.append(np.concatenate([zero_pad, data[epoch][ch], zero_pad]))
            padded_list.append(list(ch_list))
    else:
        padded_list = []

        # loop over epochs
        for epoch in range(data.shape[0]):
            ch_list = []
            # loop over channels
            for ch in range(data.shape[1]):
                # mirror data at beginning and end
                ch_list.append(np.concatenate([data[epoch][ch][::-1], data[epoch][ch], data[epoch][ch][::-1]]))
            padded_list.append(list(ch_list))

    return np.squeeze(np.array(padded_list))


# o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o >><< o END
