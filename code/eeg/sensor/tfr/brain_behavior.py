import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mne

# load c_diff, d_diff and prev_c

df = pd.read_csv("D:\\expecon_ms\\data\\behav\\sdt_para.csv")


method='multitaper'
cond_a = 'high'
cond_b = 'low'
channel_names = ['CP4', 'CP6', 'C4', 'C6']

# load EEG data
os.chdir(r"D:\\expecon_ms\\data\\eeg\\sensor\\induced_tfr\\tfr_multitaper\\laplace")

a = mne.time_frequency.read_tfrs(f"{method}_{cond_a}-tfr.h5")
b = mne.time_frequency.read_tfrs(f"{method}_{cond_b}-tfr.h5")

X = np.array([ax.crop(-0.5,0).pick_channels(channel_names).data - bx.crop(-0.5,0).pick_channels(channel_names).data for ax,bx in zip(a,b)])

alpha = [:6]
beta = [6:24]
gamma = [24:]

X_avg = np.mean(X[:,:,24:,:], axis=(1,2,3))

np.corrcoef(df.iloc[:,1], X_avg)


