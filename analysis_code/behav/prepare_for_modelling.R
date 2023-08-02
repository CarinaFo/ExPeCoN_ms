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
#library(purr)

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

# Columns to process
columns_to_process <- c("alpha_900to700", "beta_900to700", "alpha_300to100", "beta_300to100", "alpha_150to0", "beta_150to0")

# grand mean centering (we decided for this approach, see Stephani et al., 2021)
standardize_and_log <- function(x) {
  scale(log10(x))
}

# standardize and log transform
power <- power %>%
  mutate(across(all_of(columns_to_process), standardize_and_log))

# Remove attributes from the scaled columns
power[, columns_to_process] <- lapply(power[, columns_to_process], unclass)

# Convert the specified columns to numeric
power[c('block', 'trial')] <- lapply(power[c('block', 'trial')], as.numeric)

detrend_within_participants <- function(df, column) {
  for (s in unique(df$ID)) {
    dat_tmp2 <- df[df$ID == s,]
    
    trend_col <- paste0(column)
    df[[trend_col]][df$ID == s] <- df[[trend_col]][df$ID == s] -
      lm(as.formula(paste0(trend_col, " ~ trial")), data = dat_tmp2)$coefficients[2] * dat_tmp2$trial
  }
  
  for (s in unique(df$ID)) {
    dat_tmp2 <- df[df$ID == s,]
    
    trial_col <- paste0(column)
    df[[trial_col]][df$ID == s] <- df[[trial_col]][df$ID == s] -
      lm(as.formula(paste0(trial_col, " ~ block")), data = dat_tmp2)$coefficients[2] * dat_tmp2$block
  }
  
  return(df)
}


# Loop over the columns and apply detrending
for (col in columns_to_process) {
  power <- detrend_within_participants(power, col)
}

# check whether trend removal has worked (remove trend for trials within block and trend over blocks)

check = 1

if (check == 1){
  
  summary(lmer(alpha_900to700 ~ trial + (1|ID), data=power, REML=T))
  summary(lmer(alpha_900to700 ~ block + (1|ID), data=power, REML=T))
  
  summary(lmer(beta_900to700 ~ trial + (1|ID), data=power, REML=T))
  summary(lmer(beta_900to700 ~ block + (1|ID), data=power, REML=T))
  
}

# remove unnecessary variables 
power <- power[, !(names(power) %in% c("index", "sayyes_y", "X", "Unnamed..0.1",
                                                 "Unnamed..0", "sayyes_y"))]

# Identify the relative path from your current working directory to the file
relative_path <- file.path("data", "behav", "behav_df", "brain_behav_cleanpower.csv")

write.csv(power, relative_path)