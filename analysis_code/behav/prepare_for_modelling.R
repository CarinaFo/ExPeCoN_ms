#####################################ExPeCoN study#################################################


# Paradigm: 43 participants completed a behavioral and EEG study that consisted of 720 trials 
# (360 signal trials) where they had to indicate whether they felt a weak (near-threshold) somato-
# sensory stimulus and (yes/no) and give a confidence rating (guessing, sure)

# The main manipulation was a within condition that instructed particioants abbout stimulus 
# probabilites: in the high expectation condition (75 %) miniblocks of 12 trials contained 9 stimuli
# in the low expectation condition (25 %) 12 trials contained 3 signal trials (randomized between
# and within participants), to motivate participants to use those cues, we provided feedback after
# each miniblock, we had 5 blocks in total with 144 trials per block (6 low and 6 high expectation)

# we decided to cue the stimulus onset in order to analyze reaction times for catch trials(no signal)

# the behavioral data contains prestimulus alpha and beta power per trial (-300 to stimulation onset)
# estimated with morlet wavelets (4 cycles) averaged over significant channels, time and the beta 
# frequency (14-30 Hz) (3D cluster based permutation test)

# raw power values (log transform?)

# Participants set a higher criterion if they expect less stimuli (more conservative c)

# how is this change in c implemented in the brain? 

# see literature on alpha power => excitability => criterion: Busch, Samaha etc.

# Research question: does prestimulus power determine participants bias in a near-threshold
# somatosensory detection task? Does the low-level previous choice prior interact with the top-down
# prior?

# written by Carina Forster (2023)

# please report bugs:

# Email: forster@mpg.cbs.de

library(dplyr) # pandas style
library(tidyr) # spread function
library(data.table) # for shift function
library(lme4)
library(lmerTest) # pvalues for lmer models

####################################################################################################

# set working directory

# load brain behavior dataframe that contains single trial power in spec. frequency bands in sensor space
setwd("D:/expecon_ms")

# Identify the relative path from your current working directory to the file
relative_path <- file.path("data", "behav", "behav_df", "brain_behav.csv")

# Use the relative path to read the CSV file
power <- read.csv(relative_path)

# add prediction error per trial

power$PE_abs = abs(power$isyes - power$cue)

# https://philippmasur.de/2018/05/23/how-to-center-in-multilevel-models/

# Log transform and standardize the 'beta' variable within each participant (grouped by 'ID')
# cluster centering
#power <- power %>%
#  group_by(ID) %>%
#  mutate(
#    alpha_scale_log = scale(log10(alpha_150to0)),
#    beta_scale_log = scale(log10(beta_150to0))
 # )

# grand mean centering (we decided for this approach, see Stephani et al., 2021)
power$beta_scale_log <- scale(log10(power$beta_900to700))
power$alpha_scale_log <- scale(log10(power$alpha_900to700))

## detrend data within participants (see Stephani et al., 2021)

# copy dataset

power_copy <- power
power_copy$alpha <- NA
power_copy$beta <- NA

# take residuals from regression on trial_ix (calculate trend including intercept
# but do not subtract participant-specific intercepts)

for (s in c( unique(power$ID) )){
  dat_tmp2 <- power[power$ID==s,]
  
  power_copy$alpha_trial[power$ID==s] <- power$alpha_scale_log[power$ID==s] - 
    lm(alpha_scale_log ~ trial, data=dat_tmp2)$coefficients[2]*dat_tmp2$trial
  power_copy$beta_trial[power$ID==s] <- power$beta_scale_log[power$ID==s] - 
    lm(beta_scale_log~ trial, data=dat_tmp2)$coefficients[2]*dat_tmp2$trial
}

for (s in c( unique(power$ID) )){
  
  dat_tmp2 <- power_copy[power_copy$ID==s,]
  
  power_copy$alpha[power$ID==s] <- power_copy$alpha_trial[power_copy$ID==s] - 
    lm(alpha_trial ~ block, data=dat_tmp2)$coefficients[2]*dat_tmp2$block
  power_copy$beta[power$ID==s] <- power_copy$beta_trial[power_copy$ID==s] - 
    lm(beta_trial ~ block, data=dat_tmp2)$coefficients[2]*dat_tmp2$block
}

# check whether trend removal has worked (remove trend for trials within block and trend over blocks)

check = 1

if (check == 1){
  
  summary(lmer(alpha ~ trial + (1|ID), data=power_copy, REML=T))
  summary(lmer(alpha ~ block + (1|ID), data=power_copy, REML=T))
  
  summary(lmer(beta ~ trial + (1|ID), data=power_copy, REML=T))
  summary(lmer(beta ~ block + (1|ID), data=power_copy, REML=T))
  
}

# remove unnecessary variables 
power <- power_copy[, !(names(power_copy) %in% c("alpha_trial", "beta_trial", "alpha_scale_log", 
                                                 "beta_scale_log", "index", "alpha_150to0",
                                                 "beta_150to0", "sayyes_y", "X", "Unnamed..0.1",
                                                 "Unnamed..0"))]

# Identify the relative path from your current working directory to the file
relative_path <- file.path("data", "behav", "behav_df", "brain_behav_cleanpower_precue.csv")

write.csv(power, relative_path)