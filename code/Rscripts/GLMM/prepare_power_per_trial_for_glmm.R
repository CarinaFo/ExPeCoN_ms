#####################################ExPeCoN study#################################################
### prepare single trial power for generalized linear mixed effects modelling ######################

# written by Carina Forster (2023)

# please report bugs:

# Email: forster@mpg.cbs.de

library(dplyr) # pandas style
library(tidyr) # spread function
library(lme4)
library(lmerTest) # pvalues for lmer models

# which study do you want to analyze?
expecon <- 1 # 1 or 2 
source <- 1 # 1 or 0 

# check wether the removal of the trend worked?
check_trend_removal = 1

####################################################################################################
# set working directory
setwd("E:/expecon_ms")

if (expecon == 1) {
  if (source == 1) {
    # Identify the relative path from your current working directory to the file
    relative_path <- file.path("data", "behav", "brain_behav_source_700ms1.csv")
    # Use the relative path to read the CSV file
    power <- read.csv(relative_path)
    # add prediction error per trial
    power$PE_abs = abs(power$isyes - power$cue)
    # expecon 2
    columns_to_process <- c("beta_source_prob", "beta_source_prev")
  }else {
  # Identify the relative path from your current working directory to the file
  relative_path <- file.path("data", "behav", "brain_behav_1.csv")
  # Use the relative path to read the CSV file
  power <- read.csv(relative_path)
  # add prediction error per trial
  power$PE_abs = abs(power$isyes - power$cue)
  # Columns to process
  columns_to_process <- c("pre_alpha", "pre_beta")}
  
} else {
  if (source == 1) {
    # Identify the relative path from your current working directory to the file
    relative_path <- file.path("data", "behav", "brain_behav_source_2.csv")
    # Use the relative path to read the CSV file
    power <- read.csv(relative_path)
    # add prediction error per trial
    power$PE_abs = abs(power$isyes - power$cue)
    # expecon 2
    columns_to_process <- c("beta_source_prob", "beta_source_prev")
  }else{
  # Identify the relative path from your current working directory to the file
  relative_path <- file.path("data", "behav", "brain_behav_2.csv")
  # Use the relative path to read the CSV file
  power <- read.csv(relative_path)
  # add prediction error per trial
  power$PE_abs = abs(power$isyes - power$cue)
  # expecon 2
  columns_to_process <- c("pre_alpha", "pre_beta")}
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

if (expecon == 2 && check_trend_removal == 1 && source == 1) {
  summary(lmer(beta_source_prob ~ trial + (1|ID), data=power, REML=T))
} else if (expecon == 2 && check_trend_removal == 1 && source == 1) {
  summary(lmer(beta_source_prev ~ block + (1|ID), data=power, REML=T))
}

if (expecon == 1 && check_trend_removal == 1 && source == 1) {
  summary(lmer(beta_source_prob ~ trial + (1|ID), data=power, REML=T))
} else if (expecon == 1 && check_trend_removal == 1) {
  summary(lmer(beta_source_prev ~ block + (1|ID), data=power, REML=T))
}

# define filename for cleaned power
filename_cleaned_power = paste("brain_behav_cleaned_", expecon, ".csv", sep="")

if (expecon == 1) {
  if (source == 1) {
    # define filename for cleaned power
    filename_cleaned_power = paste("brain_behav_cleaned_source_", expecon, ".csv", sep="")
    # remove unnecessary variables 
    power <- power[, !(names(power) %in% c("index", "sayyes_y",'X', 'Unnamed..0.1',
                                           'Unnamed..0', "level_0"))]
    relative_path <- file.path("data", "behav", filename_cleaned_power)
  }else{
  # remove unnecessary variables 
  power <- power[, !(names(power) %in% c("index", "sayyes_y", "X", "Unnamed..0.1",
                                         "Unnamed..0", "sayyes_y", "level_0"))]
  relative_path <- file.path("data", "behav", filename_cleaned_power)}
} else {
  if (source == 1) {
    # define filename for cleaned power
    filename_cleaned_power = paste("brain_behav_cleaned_source_tvals", expecon, ".csv", sep="")
    # remove unnecessary variables 
    power <- power[, !(names(power) %in% c("index", "sayyes_y",'X', 'Unnamed..0.2', 'Unnamed..0.1',
                                           'Unnamed..0'))]
    relative_path <- file.path("data", "behav", filename_cleaned_power)
  }else{
    # remove unnecessary variables 
    power <- power[, !(names(power) %in% c("index", "sayyes_y"))]
    relative_path <- file.path("data", "behav", filename_cleaned_power)
  }
}

write.csv(power, relative_path)