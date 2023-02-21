import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import os
import pandas as pd
import scipy.stats as stats


## Data
behavpath = 'D:\\expecon_ms\\data\\behav'
savepath = 'D:\\expecon_ms\\figs\\behavior'
os.chdir(behavpath)
data = pd.read_csv("clean_bb.csv")

# Get the number of unique participants
n_subs = len(data.ID.unique())

# Prepare the data for hitrate calculation

signal = data[data.isyes == 1]
signal_grouped = signal.groupby(['ID', 'cue']).mean()['sayyes']
signal_grouped.head()

# now we plot the hitrate per participant between conditions: boxplot and single subject data points

# Define colors for the plot
blue = '#2a95ffff'
pink = '#ff2a2aff'
colors = [blue, pink]
medcolor = ['black','black']

# Get the lower and upper quartile values for the signal
low_condition_hr = signal_grouped.unstack()[0.25].reset_index()
high_condition_hr = signal_grouped.unstack()[0.75].reset_index()

# Noise\FA
noise = data[data.isyes==0]
noise_grouped = noise.groupby(['ID', 'cue']).mean()['sayyes']
noise_grouped.head()

low_condition_fa = noise_grouped.unstack()[0.25].reset_index()
high_condition_fa = noise_grouped.unstack()[0.75].reset_index()

#Criterion, DPrime, Confidence
hitrate_low = signal_grouped.unstack()[0.25]
farate_low = noise_grouped.unstack()[0.25]

d_prime_low = [stats.norm.ppf(h) - stats.norm.ppf(f) for h,f in zip(hitrate_low, farate_low)]
criterion_low = [-0.5 * (stats.norm.ppf(h) + stats.norm.ppf(f)) for h,f in zip(hitrate_low, farate_low)]

hitrate_high = signal_grouped.unstack()[0.75]
farate_high = noise_grouped.unstack()[0.75]

d_prime_high = [stats.norm.ppf(h) - stats.norm.ppf(f) for h,f in zip(hitrate_high, farate_high)]
criterion_high = [-0.5 * (stats.norm.ppf(h) + stats.norm.ppf(f)) for h,f in zip(hitrate_high, farate_high)]

data_grouped = data.groupby(['ID', 'cue']).mean()["conf"]
data_grouped.head()



low_condition_conf = data_grouped.unstack()[0.25].reset_index()
high_condition_conf = data_grouped.unstack()[0.75].reset_index()


## Figure
fig = plt.figure(figsize=(8,10),tight_layout=True)#original working was 10,12
gs = gridspec.GridSpec(6, 4)

schem_01_ax = fig.add_subplot(gs[0:2, 0:])
schem_01_ax.set_yticks([])
schem_01_ax.set_xticks([])

schem_02_ax = fig.add_subplot(gs[2:4,0:])
schem_02_ax.set_yticks([])
schem_02_ax.set_xticks([])

hr_ax = fig.add_subplot(gs[4,0])
# Plot individual data points 
for index in range(len(low_condition_hr)):
    hr_ax.plot(1, low_condition_hr.iloc[index,1],
            marker='', markersize=8, color=colors[0], markeredgecolor=colors[0], alpha=.5)
    hr_ax.plot(2, high_condition_hr.iloc[index,1],
            marker='', markersize=8, color=colors[1], markeredgecolor=colors[1], alpha=.5)
    hr_ax.plot([1,2], [low_condition_hr.iloc[index,1], high_condition_hr.iloc[index,1]],
            marker='', markersize=0, color='gray', alpha=.25)

hr_box = hr_ax.boxplot([low_condition_hr.iloc[:,1], high_condition_hr.iloc[:,1]], patch_artist=True)

# Set the face color and alpha for the boxes in the plot
for patch, color in zip(hr_box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Set the color for the medians in the plot
for patch, color in zip(hr_box['medians'], medcolor):
    patch.set_color(color)
    
hr_ax.set_ylabel('Hit rate', fontname="Arial", fontsize=14)
hr_ax.set_yticklabels(['0','0.5','1.0'], fontname="Arial", fontsize=12)
hr_ax.text(1.3,1,'***',verticalalignment='center',fontname='Arial',fontsize='18')


fa_ax = fig.add_subplot(gs[5,0])
for index in range(len(low_condition_fa)):
    fa_ax.plot(1, low_condition_fa.iloc[index,1],
            marker='', markersize=8, color=colors[0], markeredgecolor=colors[0], alpha=.5)
    fa_ax.plot(2, high_condition_fa.iloc[index,1],
            marker='', markersize=8, color=colors[1], markeredgecolor=colors[1], alpha=.5)
    fa_ax.plot([1,2], [low_condition_fa.iloc[index,1], high_condition_fa.iloc[index,1]],
            marker='', markersize=0, color='gray', alpha=.25)

fa_box = fa_ax.boxplot([low_condition_fa.iloc[:,1], high_condition_fa.iloc[:,1]], patch_artist=True)
# Set the face color and alpha for the boxes in the plot
for patch, color in zip(fa_box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Set the color for the medians in the plot
for patch, color in zip(fa_box['medians'], medcolor):
    patch.set_color(color)

fa_ax.set_ylabel('FA rate', fontname="Arial", fontsize=14)
fa_ax.set_yticklabels(['0','0.5','1.0'], fontname="Arial", fontsize=12)
fa_ax.text(1.3,1,'***',verticalalignment='center',fontname='Arial',fontsize='18')


crit_ax = fig.add_subplot(gs[4:,1])
crit_ax .set_ylabel('Criterion', fontname="Arial", fontsize=14)
crit_ax.text(1.3,1.5,'***',verticalalignment='center',fontname='Arial',fontsize='18')

crit_ax.set_ylim(-0.5,1.5)
crit_ax.set_yticks([-0.5,0.5,1.5])
crit_ax.set_yticklabels(['-0.5','0.5','1.5'], fontname="Arial", fontsize=12)



for index in range(len(criterion_low)):
    crit_ax.plot(1,criterion_low[index],
           marker='o',markersize=0,color=colors[0],markeredgecolor=colors[0],alpha=.5)
    crit_ax.plot(2,criterion_high[index],
           marker='o',markersize=0,color=colors[1],markeredgecolor=colors[1],alpha=.5)
    crit_ax.plot([1,2],[criterion_low[index],criterion_high[index]],
           marker='',markersize=0,color='gray',alpha=.25)
    
crit_box = crit_ax.boxplot([criterion_low, criterion_high],
                    patch_artist=True)    
    # Set the face color and alpha for the boxes in the plot
for patch, color in zip(crit_box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Set the color for the medians in the plot
for patch, color in zip(crit_box['medians'], medcolor):
    patch.set_color(color)

dprime_ax = fig.add_subplot(gs[4:,2])
dprime_ax .set_ylabel('Dprime', fontname="Arial", fontsize=14)
dprime_ax.text(1.4,3,'n.s',verticalalignment='center',fontname='Arial',fontsize='13')

dprime_ax.set_ylim(0,3)
dprime_ax.set_yticks([0,1.5,3])
dprime_ax.set_yticklabels(['0','1.5','3.0'], fontname="Arial", fontsize=12)

for index in range(len(d_prime_low)):
    dprime_ax.plot(1,d_prime_low[index],
           marker='o',markersize=0,color=colors[0],markeredgecolor=colors[0],alpha=.5)
    dprime_ax.plot(2,d_prime_high[index],
           marker='o',markersize=0,color=colors[1],markeredgecolor=colors[1],alpha=.5)
    dprime_ax.plot([1,2],[d_prime_low[index],d_prime_high[index]],
           marker='',markersize=0,color='gray',alpha=.25)
    
dprime_box = dprime_ax.boxplot([d_prime_low, d_prime_high],
                    patch_artist=True)    
    # Set the face color and alpha for the boxes in the plot
for patch, color in zip(dprime_box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Set the color for the medians in the plot
for patch, color in zip(dprime_box['medians'], medcolor):
    patch.set_color(color)



conf_ax = fig.add_subplot(gs[4:,3])
conf_ax .set_ylabel('Confidence', fontname="Arial", fontsize=14)
conf_ax.set_yticklabels(['0','0.5','1.0'], fontname="Arial", fontsize=12)
conf_ax.text(1.3,1,'***',verticalalignment='center',fontname='Arial',fontsize='18')

# Plot individual data points 
for index in range(len(low_condition_hr)):
    conf_ax.plot(1, low_condition_conf.iloc[index,1],
            marker='', markersize=8, color=colors[0], markeredgecolor=colors[0], alpha=.5)
    conf_ax.plot(2, high_condition_conf.iloc[index,1],
            marker='', markersize=8, color=colors[1], markeredgecolor=colors[1], alpha=.5)
    conf_ax.plot([1,2], [low_condition_conf.iloc[index,1], high_condition_conf.iloc[index,1]],
            marker='', markersize=0, color='gray', alpha=.25)

conf_box = conf_ax.boxplot([low_condition_conf.iloc[:,1], high_condition_conf.iloc[:,1]], patch_artist=True)

# Set the face color and alpha for the boxes in the plot
colors = ['white','black']
for patch, color in zip(conf_box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# Set the color for the medians in the plot
medcolor = ['black','white']
for patch, color in zip(conf_box['medians'], medcolor):
    patch.set_color(color)

for plots in [hr_ax,fa_ax,conf_ax]:
    plots.set_ylim(0,1)
    plots.set_yticks([0,0.5,1.0])

for plots in [hr_ax,fa_ax,crit_ax,dprime_ax,conf_ax]:
    plots.spines['top'].set_visible(False)
    plots.spines['right'].set_visible(False)
    plots.set_xticks([1,2])
    plots.set_xlim(0.5,2.5)
    plots.set_xticklabels(['',''])
    if plots != hr_ax:
        plots.set_xticklabels(['0.25','0.75'], fontname="Arial", fontsize=12)
        plots.set_xlabel('Stimulus (P)', fontname="Arial", fontsize=14)
    if plots == conf_ax:
        plots.set_xticklabels(['Congruent', 'Incongruent'], fontname="Arial", fontsize=12,rotation=30)
        plots.set_xlabel('')



fig.savefig(savepath + "\\figure1.svg",dpi=300, bbox_inches='tight', format='svg')
plt.show()