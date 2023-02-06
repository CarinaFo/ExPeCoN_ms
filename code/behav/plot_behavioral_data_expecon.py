# Plot behavioral data expecon study
# 
# last update: 06.02.2023
#
# this script produces data for Figure 1 Part II and III
# 
# written by Carina Forster 
# please report bugs: forster@cbs.mpg.de

# TODO:

# - change colors for congruency and accuracy plots

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns
import scipy.stats as stats

# Set the color palette for seaborn
sns.color_palette("colorblind")[9]

# Set the paths for the behavioral data and the save path
behavpath = r'D:\expecon\data\behav_brain'
savepath = r'D:\expecon_ms\figs\behavior'

# Load the behavioral data from the specified path
data = []
for root, dirs, files in os.walk(behavpath):
    for name in files:
        if 'behav_brain_expecon_sensor_laplace.csv' in name:
            data = pd.read_csv(os.path.join(root, name))

df = pd.read_csv(os.path.join(root, 'behav_brain_expecon_sensor_laplace.csv'))

# Clean up the dataframe by dropping unneeded columns
columns_to_drop = ["Unnamed: 0", 'X.1', 'X', 'Unnamed..0', 'Unnamed..0.1',
                   'Unnamed..0.2', 'Unnamed..0.3', 'alpha_trial', 'beta_trial',
                   'beta_scale_log', 'alpha_scale_log']
data_clean = df.drop(columns_to_drop, axis=1)
data = data_clean

# Change the block number for participant 7's block 3
data.loc[(144*2):(144*3), 'block'] = 4

# Drop participants 40 and 45
data = data.drop(data[data.ID == 40].index)
data = data.drop(data[data.ID == 45].index)

# Remove blocks with hitrates < 0.2 or > 0.8
data = data.drop(data[((data.ID == 10) & (data.block == 6))].index)
data = data.drop(data[((data.ID == 12) & (data.block == 6))].index)
data = data.drop(data[((data.ID == 26) & (data.block == 4))].index)
data = data.drop(data[((data.ID == 30) & (data.block == 3))].index)
data = data.drop(data[((data.ID == 32) & (data.block == 2))].index)
data = data.drop(data[((data.ID == 32) & (data.block == 3))].index)
data = data.drop(data[((data.ID == 39) & (data.block == 3))].index)

data.to_csv("clean_bb.csv")

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
low_condition = signal_grouped.unstack()[0.25].reset_index()
high_condition = signal_grouped.unstack()[0.75].reset_index()

# Create a figure and set the limits and labels for x and y axis
fig, ax = plt.subplots()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim(0.5, 2.5)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Low', 'High'])

ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels(['0', '.25', '0.5', '0.75', '1'], fontsize=12)

# Plot the box plot for both conditions
confbox = ax.boxplot([low_condition.iloc[:,1], high_condition.iloc[:,1]], patch_artist=True)

# Plot individual data points 
for index in range(len(low_condition)):
    ax.plot(1, low_condition.iloc[index,1],
            marker='o', markersize=8, color=colors[0], markeredgecolor=colors[0], alpha=.5)
    ax.plot(2, high_condition.iloc[index,1],
            marker='o', markersize=8, color=colors[1], markeredgecolor=colors[1], alpha=.5)
    ax.plot([1, 2], [low_condition.iloc[index,1], high_condition.iloc[index,1]],
            marker='', markersize=0, color='gray', alpha=.25)

# Set the face color and alpha for the boxes in the plot
for patch, color in zip(confbox['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)

# Set the color for the medians in the plot
for patch, color in zip(confbox['medians'], medcolor):
    patch.set_color(color)

# Set the x axis labels
ax.set_xticks([1, 2])
ax.set_xticklabels(['Low', 'High'], fontsize=14)
ax.set_ylabel('Hit rate', fontsize=14)

# Save the figure
fig.savefig(savepath + "\hitrate.svg", bbox_inches='tight', format='svg')
plt.show()

# Violinplot ? 

# Define colors for the plot
blue = '#2a95ffff'
pink = '#ff2a2aff'
alpha = 0.5
colors = [blue, pink]
medcolor = ['black','black']

# Plot the violin plot
sns.violinplot(data=[low_condition.iloc[:,1], high_condition.iloc[:,1]], palette=colors)

# Set the y-axis limit and ticks
plt.ylim(0,1)
plt.yticks([0,.25,.5,.75,1], ['0','.25','0.5','0.75','1'], fontsize=12)

# Set the x and y axis labels
plt.xlabel('Condition', fontsize=14)
plt.ylabel('Hit rate', fontsize=14)

# Save the plot
plt.savefig(savepath + "\\violin_hitrate.svg", bbox_inches='tight', format='svg')
plt.show()

# sign. difference in hitrates ? 

# non parametric

t,p = stats.wilcoxon(low_condition.iloc[:,1], high_condition.iloc[:,1])

print('hitrate diff significant? p_value (non parametric): ' + str(p))

# Check false alarm rate

#Prepare data

noise = data[data.isyes==0]
noise_grouped = noise.groupby(['ID', 'cue']).mean()['sayyes']
noise_grouped.head()

blue= '#2a95ffff'
pink = '#ff2a2aff'
alpha = 0.5
colors = [blue, pink]
medcolor = ['black','black']
#fontsize
fs=12

low_condition = noise_grouped.unstack()[0.25].reset_index()
high_condition = noise_grouped.unstack()[0.75].reset_index()

fig,ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlim(0.5,2.5)
ax.set_xticks([1,2])
ax.set_xticklabels(['Low','High'])

ax.set_ylim(0,0.6)
ax.set_yticks([0, 0.25, .5])
ax.set_yticklabels(['0', '.25', '0.5'],fontsize=12)

confbox = ax.boxplot([low_condition.iloc[:,1],high_condition.iloc[:,1]],
                    patch_artist=True)

for index in range(len(low_condition)):
    ax.plot(1,low_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[0],markeredgecolor=colors[0],alpha=alpha)
    ax.plot(2,high_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[1],markeredgecolor=colors[1],alpha=alpha)
    ax.plot([1,2],[low_condition.iloc[index,1],high_condition.iloc[index,1]],
           marker='',markersize=0,color='gray',alpha=alpha/2)
    
for patch, color in zip(confbox['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(alpha)
    
for patch, color in zip(confbox['medians'], medcolor):
    patch.set_color(color)
    
ax.set_xticks([1,2])
ax.set_xticklabels(['Low','High'],fontsize=fs)
ax.set_ylabel('False alarm rate', fontsize=fs)

fig.savefig(savepath + "\FArate.svg", bbox_inches='tight', format='svg')

# sign. difference in farates ? 

# non parametric
t,p = stats.wilcoxon(low_condition.iloc[:,1], high_condition.iloc[:,1])

print('farate diff significant? p_value (non parametric): ' + str(p))

# Now we plot SDT parameters

# calculate SDT from hitrates and FA rates

hitrate_low = signal_grouped.unstack()[0.25]
farate_low = noise_grouped.unstack()[0.25]

d_prime_low = [stats.norm.ppf(h) - stats.norm.ppf(f) for h,f in zip(hitrate_low, farate_low)]
criterion_low = [-0.5 * (stats.norm.ppf(h) + stats.norm.ppf(f)) for h,f in zip(hitrate_low, farate_low)]

hitrate_high = signal_grouped.unstack()[0.75]
farate_high = noise_grouped.unstack()[0.75]

d_prime_high = [stats.norm.ppf(h) - stats.norm.ppf(f) for h,f in zip(hitrate_high, farate_high)]
criterion_high = [-0.5 * (stats.norm.ppf(h) + stats.norm.ppf(f)) for h,f in zip(hitrate_high, farate_high)]

# ### plot the criterion

fig,ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax.set_xlim(0.5,2.5)
ax.set_xticks([1,2])
ax.set_xticklabels(['Low','High'])

#ax.set_ylim(0,1)
ax.set_yticks([-0.5,0,.5,1,1.5])
ax.set_yticklabels(['-0.5', '0.0','.5','1.0','1.5'],fontsize=12)

confbox = ax.boxplot([criterion_low, criterion_high],
                    patch_artist=True)

for index in range(len(criterion_low)):
    ax.plot(1,criterion_low[index],
           marker='o',markersize=8,color=colors[0],markeredgecolor=colors[0],alpha=.5)
    ax.plot(2,criterion_high[index],
           marker='o',markersize=8,color=colors[1],markeredgecolor=colors[1],alpha=.5)
    ax.plot([1,2],[criterion_low[index],criterion_high[index]],
           marker='',markersize=0,color='gray',alpha=.25)
    

for patch, color in zip(confbox['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
    
for patch, color in zip(confbox['medians'], medcolor):
    patch.set_color(color)
    
ax.set_xticks([1,2])
ax.set_xticklabels(['Low','High'],fontsize=12)
ax.set_ylabel('Criterion', fontsize=12)

fig.savefig(savepath + "\criterion.svg", bbox_inches='tight', format='svg')

# stats

# sign. difference in c ? 

# non parametric
t,p = stats.wilcoxon(criterion_high, criterion_low)

print('c diff significant? p_value (non parametric): ' + str(p))

# now we plot sensitivity (dprime)

fig,ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax.set_xlim(0.5,2.5)
ax.set_xticks([1,2])
ax.set_xticklabels(['Low','High'])

ax.set_ylim(0.0,3.0)
ax.set_yticks([0.0,1.5,3.0])
#ax.set_yticklabels([0.0', '.5','1.0','1.5'],fontsize=12)

confbox = ax.boxplot([d_prime_low,d_prime_high],
                    patch_artist=True)


for index in range(len(d_prime_low)):
    ax.plot(1,d_prime_low[index],
           marker='o',markersize=8,color=colors[0],markeredgecolor=colors[0],alpha=.5)
    ax.plot(2,d_prime_high[index],
           marker='o',markersize=8,color=colors[1],markeredgecolor=colors[1],alpha=.5)
    ax.plot([1,2],[d_prime_low[index],d_prime_high[index]],
           marker='',markersize=0,color='gray',alpha=.25)

for patch, color in zip(confbox['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
    
for patch, color in zip(confbox['medians'], medcolor):
    patch.set_color(color)
    
ax.set_xticks([1,2])
ax.set_xticklabels(['Low','High'],fontsize=12)
ax.set_ylabel('Sensitivity', fontsize=12)

fig.savefig(savepath + "\sensitivity.svg", bbox_inches='tight', format='svg')

# sign. difference in hitrates ? 

# non parametric
t,p = stats.wilcoxon(d_prime_high, d_prime_low)

print('dprime diff significant? p_value (non parametric): ' + str(p))

# now we look at confidence ratings

data_grouped = data.groupby(['ID', 'cue']).mean()["conf"]
data_grouped.head()

low_condition = data_grouped.unstack()[0.25].reset_index()
high_condition = data_grouped.unstack()[0.75].reset_index()

low_condition.head()

fig,ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax.set_xticklabels(['Low','High'])

ax.set_ylim(0,1)
ax.set_yticks([0,.5,1])
ax.set_yticklabels(['0.0', '.5','1.0'],fontsize=12)

confbox = ax.boxplot([low_condition.iloc[:,1],high_condition.iloc[:,1]],
                    patch_artist=True)

for index in range(len(low_condition)):
    ax.plot(1,low_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[0],markeredgecolor=colors[0],alpha=.5)
    ax.plot(2,high_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[1],markeredgecolor=colors[1],alpha=.5)
    ax.plot([1,2],[low_condition.iloc[index,1],high_condition.iloc[index,1]],
           marker='',markersize=0,color='gray',alpha=.25)

for patch, color in zip(confbox['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
    
for patch, color in zip(confbox['medians'], medcolor):
    patch.set_color(color)
    
ax.set_xticks([1,2])
ax.set_xticklabels(['Low','High'],fontsize=12)
ax.set_ylabel('mean confidence', fontsize=12)

fig.savefig(savepath + "\confidence_cue.svg", bbox_inches='tight', format='svg')

# non parametric

t,p = stats.wilcoxon(low_condition.iloc[:,1], high_condition.iloc[:,1])

print('confidence cue diff significant? p_value (non parametric): ' + str(p))

# ### now we look at congruency effects on confidence

data['congruency'] = ((data.cue == 0.25) & (data.sayyes == 0) | (data.cue == 0.75) & (data.sayyes == 1))

data_grouped = data.groupby(['ID', 'congruency']).mean()['conf']
con_condition = data_grouped.unstack()[1].reset_index()
incon_condition = data_grouped.unstack()[0].reset_index()

green = '#2aff95'
purple = '#2a2aff'

colors = [purple, green]


fig,ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

orange = (0.8352941176470589, 0.3686274509803922, 0.0)
blue = (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)

ax.set_xlim(0.5,2.5)
ax.set_xticks([1,2])
ax.set_xticklabels(['Congruent','Incongruent'])

ax.set_ylim(0,1)
ax.set_yticks([0,0.5,1.0])
#ax.set_yticklabels([0.0', '.5','1.0','1.5'],fontsize=12)

confbox = ax.boxplot([con_condition.iloc[:,1],incon_condition.iloc[:,1]],
                    patch_artist=True)

for index in range(len(con_condition)):
    ax.plot(1,con_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[0],markeredgecolor=colors[0], alpha=.5)
    ax.plot(2,incon_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[1],markeredgecolor=colors[1],alpha=.5)
    ax.plot([1,2],[con_condition.iloc[index,1],incon_condition.iloc[index,1]],
           marker='',markersize=0,color='gray',alpha=.25)
    

for patch, color in zip(confbox['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
    
for patch, color in zip(confbox['medians'], medcolor):
    patch.set_color(color)
    
ax.set_xticks([1,2])
ax.set_xticklabels(['Congruent','Incongruent'],fontsize=12)
ax.set_ylabel('Mean confidence', fontsize=12)

fig.savefig(savepath + "\confidence_congruency.svg", bbox_inches='tight', format='svg')

t,p = stats.wilcoxon(con_condition.iloc[:,1], incon_condition.iloc[:,1])

print('confidence congruency diff significant? p_value (non parametric): ' + str(p))

# Finally let's check overall accuraccy

data['correct'] = ((data.isyes == 1) & (data.sayyes == 1) | (data.isyes == 0) & (data.sayyes == 0))

data.head()

data_grouped = data.groupby(['ID', 'cue']).mean()['correct']

data_grouped

con_condition = data_grouped.unstack()[0.25].reset_index()
incon_condition = data_grouped.unstack()[0.75].reset_index()

con_condition

green = '#2aff95'
purple = '#2a2aff'

colors = [purple, green]


fig,ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

orange = (0.8352941176470589, 0.3686274509803922, 0.0)
blue = (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)

ax.set_xlim(0.5,2.5)
ax.set_xticks([1,2])
ax.set_xticklabels(["Low",'High'])

ax.set_ylim(0.5,1)
ax.set_yticks([0.5,0.75, 1])
#ax.set_yticklabels([0.0', '.5','1.0','1.5'],fontsize=12)

confbox = ax.boxplot([con_condition.iloc[:,1],incon_condition.iloc[:,1]],
                    patch_artist=True)

for index in range(len(con_condition)):
    ax.plot(1,con_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[0],markeredgecolor=colors[0], alpha=.5)
    ax.plot(2,incon_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[1],markeredgecolor=colors[1],alpha=.5)
    ax.plot([1,2],[con_condition.iloc[index,1],incon_condition.iloc[index,1]],
           marker='',markersize=0,color='gray',alpha=.25)
    
for patch, color in zip(confbox['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
    
for patch, color in zip(confbox['medians'], medcolor):
    patch.set_color(color)
    
ax.set_xticks([1,2])
ax.set_xticklabels(['Low','High'],fontsize=12)
ax.set_ylabel('Mean correct', fontsize=12)

fig.savefig(savepath + "\cue_correct.svg", bbox_inches='tight', format='svg')

t,p = stats.wilcoxon(con_condition.iloc[:,1], incon_condition.iloc[:,1])

print('accuracy cue diff significant? p_value (non parametric): ' + str(p))

# accuraccy depending on congruency?

data_grouped = data.groupby(['ID', 'congruency']).mean()['correct']

con_condition = data_grouped.unstack()[1].reset_index()
incon_condition = data_grouped.unstack()[0].reset_index()

green = '#2aff95'
purple = '#2a2aff'

colors = [purple, green]


fig,ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

orange = (0.8352941176470589, 0.3686274509803922, 0.0)
blue = (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)

ax.set_xlim(0.5,2.5)
ax.set_xticks([1,2])
ax.set_xticklabels(['Congruent','Incongruent'])

ax.set_ylim(0.4,1)
ax.set_yticks([0.5, 0.75, 1.0])
ax.set_yticklabels(['.5','0.75','1'],fontsize=12)

confbox = ax.boxplot([con_condition.iloc[:,1],incon_condition.iloc[:,1]],
                    patch_artist=True)

for index in range(len(con_condition)):
    ax.plot(1,con_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[0],markeredgecolor=colors[0], alpha=.5)
    ax.plot(2,incon_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[1],markeredgecolor=colors[1],alpha=.5)
    ax.plot([1,2],[con_condition.iloc[index,1],incon_condition.iloc[index,1]],
           marker='',markersize=0,color='gray',alpha=.25)
    

for patch, color in zip(confbox['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
    
for patch, color in zip(confbox['medians'], medcolor):
    patch.set_color(color)
    
ax.set_xticks([1,2])
ax.set_xticklabels(['Congruent','Incongruent'],fontsize=12)
ax.set_ylabel('Mean correct', fontsize=12)

fig.savefig(savepath + "\correct_congruency.svg", bbox_inches='tight', format='svg')

# non parametric
t,p = stats.wilcoxon(con_condition.iloc[:,1], incon_condition.iloc[:,1])

print('accuracy congruency diff significant? p_value (non parametric): ' + str(p))

# reaction times 

data_grouped = data.groupby(['ID', 'cue']).mean()['respt1']

data_grouped

con_condition = data_grouped.unstack()[0.25].reset_index()
incon_condition = data_grouped.unstack()[0.75].reset_index()

con_condition

green = '#2aff95'
purple = '#2a2aff'

colors = [purple, green]


fig,ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

orange = (0.8352941176470589, 0.3686274509803922, 0.0)
blue = (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)

ax.set_xlim(0.5,2.5)
ax.set_xticks([1,2])
ax.set_xticklabels(["Low",'High'])

#ax.set_yticklabels([0.0', '.5','1.0','1.5'],fontsize=12)

confbox = ax.boxplot([con_condition.iloc[:,1],incon_condition.iloc[:,1]],
                    patch_artist=True)

for index in range(len(con_condition)):
    ax.plot(1,con_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[0],markeredgecolor=colors[0], alpha=.5)
    ax.plot(2,incon_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[1],markeredgecolor=colors[1],alpha=.5)
    ax.plot([1,2],[con_condition.iloc[index,1],incon_condition.iloc[index,1]],
           marker='',markersize=0,color='gray',alpha=.25)
    

for patch, color in zip(confbox['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
    
for patch, color in zip(confbox['medians'], medcolor):
    patch.set_color(color)
    
ax.set_xticks([1,2])
ax.set_xticklabels(['Low','High'],fontsize=12)
ax.set_ylabel('Reaction times detection', fontsize=12)

fig.savefig(savepath + "\cue_rt1.svg", bbox_inches='tight', format='svg')

# non parametric
t,p = stats.wilcoxon(con_condition.iloc[:,1], incon_condition.iloc[:,1])

print('reaction time cue diff significant? p_value (non parametric): ' + str(p))

# congruency on rts

data_grouped = data.groupby(['ID', 'congruency']).mean()['respt1']

data_grouped

con_condition = data_grouped.unstack()[1].reset_index()
incon_condition = data_grouped.unstack()[0].reset_index()

con_condition

green = '#2aff95'
purple = '#2a2aff'

colors = [purple, green]


fig,ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

orange = (0.8352941176470589, 0.3686274509803922, 0.0)
blue = (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)

ax.set_xlim(0.5,2.5)
ax.set_xticks([1,2])
ax.set_xticklabels(["Low",'High'])


#ax.set_yticklabels([0.0', '.5','1.0','1.5'],fontsize=12)

confbox = ax.boxplot([con_condition.iloc[:,1],incon_condition.iloc[:,1]],
                    patch_artist=True)

for index in range(len(con_condition)):
    ax.plot(1,con_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[0],markeredgecolor=colors[0], alpha=.5)
    ax.plot(2,incon_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[1],markeredgecolor=colors[1],alpha=.5)
    ax.plot([1,2],[con_condition.iloc[index,1],incon_condition.iloc[index,1]],
           marker='',markersize=0,color='gray',alpha=.25)
    
for patch, color in zip(confbox['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
    
for patch, color in zip(confbox['medians'], medcolor):
    patch.set_color(color)
    
ax.set_xticks([1,2])
ax.set_xticklabels(['Congruent','Incongruent'],fontsize=12)
ax.set_ylabel('Reaction times detection', fontsize=12)

fig.savefig(savepath + "\cue_rt1.svg", bbox_inches='tight', format='svg')

# non parametric
t,p = stats.wilcoxon(con_condition.iloc[:,1], incon_condition.iloc[:,1])

print('reaction time congruency diff significant? p_value (non parametric): ' + str(p))

# accuraccy on rts

data_grouped = data.groupby(['ID', 'correct']).mean()['respt1']

data_grouped

con_condition = data_grouped.unstack()[1].reset_index()
incon_condition = data_grouped.unstack()[0].reset_index()

con_condition

green = '#2aff95'
purple = '#2a2aff'

colors = [purple, green]


fig,ax = plt.subplots()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

orange = (0.8352941176470589, 0.3686274509803922, 0.0)
blue = (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)

ax.set_xlim(0.5,2.5)
ax.set_xticks([1,2])
ax.set_xticklabels(["Low",'High'])


#ax.set_yticklabels([0.0', '.5','1.0','1.5'],fontsize=12)

confbox = ax.boxplot([con_condition.iloc[:,1],incon_condition.iloc[:,1]],
                    patch_artist=True)

for index in range(len(con_condition)):
    ax.plot(1,con_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[0],markeredgecolor=colors[0], alpha=.5)
    ax.plot(2,incon_condition.iloc[index,1],
           marker='o',markersize=8,color=colors[1],markeredgecolor=colors[1],alpha=.5)
    ax.plot([1,2],[con_condition.iloc[index,1],incon_condition.iloc[index,1]],
           marker='',markersize=0,color='gray',alpha=.25)
    

for patch, color in zip(confbox['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.5)
    
for patch, color in zip(confbox['medians'], medcolor):
    patch.set_color(color)
    
ax.set_xticks([1,2])
ax.set_xticklabels(['Correct','Incorrect'],fontsize=12)
ax.set_ylabel('Reaction times detection', fontsize=12)

fig.savefig(savepath + "\correct_rt1.svg", bbox_inches='tight', format='svg')

# non parametric
t,p = stats.wilcoxon(con_condition.iloc[:,1], incon_condition.iloc[:,1])

print('reaction time accuracy diff significant? p_value (non parametric): ' + str(p))

