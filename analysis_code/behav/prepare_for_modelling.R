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

power = read.csv('D:\\expecon\\data\\behav_brain\\power_per_trial_alpha_beta_sensor_laplace.csv')
# TODO SIMON COMMENT: "I would suggest to use relative paths, assuming you start from the root of the repo."

# add variables for later analysis before data cleaning

#include previous trial variables (lag variables)

# previous choice

power$lag1r = shift(power$sayyes, n=1) 

# previous stimulus 

power$lag1 = shift(power$isyes, n=1)

# add accuracy variable

power$correct = power$isyes == power$sayyes

# previous trial accuracy

power$correct1 = shift(power$correct, n=1) 

# previous trial confidence

power$lag1c = shift(power$conf, n=1)

# add prediction error per trial

power$PE_abs = abs(power$isyes - power$cue)

# convert power values, log transform and standardize (see Stephani et. al, 2021)

power$beta_scale_log <- scale(log10(power$beta), center=T, scale=T)
power$alpha_scale_log <- scale(log10(power$alpha), center=T, scale=T)

# remove no response trials (max rt=2.5) 
# and trials that are faster than 200 ms for first order response 

power <- power %>%
  filter(power$respt1 < 2.5 & power$respt1 > 0.2 & power$respt2 < 2.5)

# save rt cleaned data

#write.csv(cue_power, "data_betapower_exclRT.csv")

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

write.csv(power_copy, 'D:\\expecon\\data\\behav_brain\\behav_brain_expecon_sensor_laplace.csv')
# TODO SIMON COMMENT: "Same as above."