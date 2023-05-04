# generalized linear mixed models to estimate signal detection theory parameters
# including previous choice effects

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

# Participants set a higher criterion if they expect less stimuli (more conservative c)

# written by Carina Forster

# please report bugs:

# Email: forster@mpg.cbs.de

library(lme4) # mixed models 
library(brms) # bayesian mixed models
library(lmerTest) # no p values without this package for linear mixed mdoels
library(dplyr)# pandas style
library(mediation)
library(tidyr)
library(data.table) # for shift function

####################################################################################################

# set working directory and load behavioral dataframe

behav = read.csv("D:\\expecon_ms\\data\\behav\\clean_bb.csv")

behav$lag1r = shift(behav$sayyes, n=1) 

###############manual SDT calculation##############

# without cue

sdt <- behav %>%
  mutate(
    type = "hit",
    type = ifelse(isyes == 1 & sayyes == 0, "miss", type),
    type = ifelse(isyes == 0 & sayyes == 0, "cr", type), # Correct rejection
    type = ifelse(isyes == 0 & sayyes == 1, "fa", type) # False alarm
  )

sdt <- sdt %>%
  group_by(type) %>%
  summarise(count = n()) %>%
  spread(type, count) # Format data to one row per person

sdt[is.na(sdt)] <- 0

sdt_overall_nocue <- sdt %>%
  mutate(
    hitrate = hit / (hit+miss),
    falsealarmrate = fa/ (fa+cr),
    zhr = qnorm((hit+0.5) / (hit + miss+1)),
    zfa = qnorm((fa+0.5) / (fa + cr+1)),
    dprime = zhr - zfa,
    crit = -(zhr + zfa)/2
  )


sdt <- power %>%
  mutate(
    type = "hit",
    type = ifelse(isyes == 1 & sayyes == 0, "miss", type),
    type = ifelse(isyes == 0 & sayyes == 0, "cr", type), # Correct rejection
    type = ifelse(isyes == 0 & sayyes == 1, "fa", type) # False alarm
  )

sdt <- sdt %>%
  group_by(type,cue) %>%
  summarise(count = n()) %>%
  spread(type, count) # Format data to one row per person

sdt[is.na(sdt)] <- 0

sdt_overall <- sdt %>%
  mutate(
    hitrate = hit / (hit+miss),
    falsealarmrate = fa/ (fa+cr),
    zhr = qnorm((hit+0.5) / (hit + miss+1)),
    zfa = qnorm((fa+0.5) / (fa + cr+1)),
    dprime = zhr - zfa,
    crit = -(zhr + zfa)/2
  )

#SDT per subject

sdt <- behav %>%
  mutate(
    type = "hit",
    type = ifelse(isyes == 1 & sayyes == 0, "miss", type),
    type = ifelse(isyes == 0 & sayyes == 0, "cr", type), # Correct rejection
    type = ifelse(isyes == 0 & sayyes == 1, "fa", type) # False alarm
  )

sdt <- sdt %>%
  group_by(type,cue,ID) %>%
  summarise(count = n()) %>%
  spread(type, count) # Format data to one row per person

sdt[is.na(sdt)] <- 0

sdt_persub <- sdt %>%
  mutate(
    hitrate = hit / (hit+miss),
    falsealarmrate = fa/ (fa+cr),
    zhr = qnorm((hit+0.5) / (hit + miss+1)),
    zfa = qnorm((fa+0.5) / (fa + cr+1)),
    dprime = zhr - zfa,
    crit = -(zhr + zfa)/2
  )

#per block 

sdt <- behav %>%
  mutate(
    type = "hit",
    type = ifelse(isyes == 1 & sayyes == 0, "miss", type),
    type = ifelse(isyes == 0 & sayyes == 0, "cr", type), # Correct rejection
    type = ifelse(isyes == 0 & sayyes == 1, "fa", type) # False alarm
  )

sdt <- sdt %>%
  group_by(ID, type, block,cue) %>%
  summarise(count = n()) %>%
  spread(type, count) # Format data to one row per person

sdt[is.na(sdt)] <- 0

sdt_perblock <- sdt %>%
  mutate(
    hitrate = hit / (hit+miss),
    falsealarmrate = fa/ (fa+cr),
    zhr = qnorm((hit+0.5) / (hit + miss+1)),
    zfa = qnorm((fa+0.5) / (fa + cr+1)),
    dprime = zhr - zfa,
    crit = -0.5*(zhr + zfa)
  )

################################ linear mixed modelling ###########################################

# make factors:

behav$ID = as.factor(behav$ID)
behav$isyes = as.factor(behav$isyes)
behav$cue = as.factor(behav$cue)

# fit sdt model

cue_model = glmer(sayyes ~ isyes*cue + (isyes*cue|ID), data=behav, 
                  family=binomial(link='probit'),
                  control=glmerControl(optimizer="bobyqa",
                                       optCtrl=list(maxfun=2e5)))

save(cue_model, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\sdt_cue.rda")

# fit sdt model including previous choice parameter

cue_lag_model = glmer(sayyes ~ isyes*cue + lag1r + (isyes*cue+lag1r|ID), 
                      data=behav, family=binomial(link='probit'),
                      control=glmerControl(optimizer="bobyqa",
                      optCtrl=list(maxfun=2e5)))

save(cue_lag_model, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\sdt_cue_prevchoice.rda")

# cue is still a significant predictor but less strong effect

cue_lag_int_model = glmer(sayyes ~ isyes*cue + lag1r*cue + (isyes*cue+lag1r*cue|ID), 
                      data=behav, family=binomial(link='probit'),
                      control=glmerControl(optimizer="bobyqa",
                                           optCtrl=list(maxfun=2e5)))
