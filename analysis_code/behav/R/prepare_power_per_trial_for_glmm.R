#####################################ExPeCoN study#################################################
### prepare single trial power for generalized linear mixed effects modelling ######################

# written by Carina Forster (2023)

# please report bugs:

# Email: forster@mpg.cbs.de

library(dplyr) # pandas style
library(tidyr) # spread function
library(data.table) # for shift function
library(lme4)
library(lmerTest) # pvalues for lmer models

# which study do you want to analyze?

expecon <- 2

# check wether the removal of the trend worked?

check_trend_removal = 1

####################################################################################################

# set working directory
# Example 3: if-else if loop

if (expecon == 1) {
  
  # load brain behavior dataframe that contains single trial power in spec. frequency bands in sensor space
  setwd("D:/expecon_ms")
  # Identify the relative path from your current working directory to the file
  relative_path <- file.path("data", "behav", "behav_df", "brain_behav.csv")
  # Use the relative path to read the CSV file
  power <- read.csv(relative_path)
  # add prediction error per trial
  power$PE_abs = abs(power$isyes - power$cue)
  # Columns to process
  columns_to_process <- c("pre_alpha", "pre_beta")
  
} else {
  
  setwd("D:/expecon_2")
  # Identify the relative path from your current working directory to the file
  relative_path <- file.path("behav", "brain_behav_expecon2.csv")
  # Use the relative path to read the CSV file
  power <- read.csv(relative_path)
  # add prediction error per trial
  power$PE_abs = abs(power$isyes - power$cue)
  # expecon 2
  columns_to_process <- c("pre_alpha", "pre_beta")
}

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

if (expecon == 1 && check_trend_removal == 1) {
  summary(lmer(pre_alpha ~ trial + (1|ID), data=power, REML=T))
} else if (expecon == 2 && check_trend_removal == 1) {
  summary(lmer(pre_beta ~ block + (1|ID), data=power, REML=T))
}


if (expecon == 1) {
  # remove unnecessary variables 
  power <- power[, !(names(power) %in% c("index", "sayyes_y", "X", "Unnamed..0.1",
                                         "Unnamed..0", "sayyes_y", "level_0"))]
  relative_path <- file.path("data", "behav", "behav_df", "brain_behav_cleanpower.csv")
} else {
  # remove unnecessary variables 
  power <- power[, !(names(power) %in% c("index", "sayyes_y"))]
  relative_path <- file.path("behav", "brain_behav_cleanpower.csv")
}

write.csv(power, relative_path)