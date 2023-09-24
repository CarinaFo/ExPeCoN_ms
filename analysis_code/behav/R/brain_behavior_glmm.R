# generalized linear mixed models to estimate signal detection theory parameters
# including previous choice effects: Brain-behavior modelling

#####################################ExPeCoN study#################################################


# Paradigm 1: mini-block design:
# 43 participants completed a behavioral and EEG study that consisted of 720 trials in total, 
# divided in 5 blocks
# participants had to indicate whether they felt a weak (near-threshold) somato-
# sensory stimulus (yes/no) and give a confidence rating (sure/unsure) on each trial
# experimental manipulation: stimulus probability: high vs. low stimulus probability (cue was
# valid for 12 trials (mini-blocks))

# Author: Carina Forster
# Email: forster@mpg.cbs.de

# libraries
library(lme4) # mixed models
library(mediation)
library(lmerTest) # no p values without this package for linear mixed mdoels
library(dplyr)# pandas style
library(tidyr)
library(data.table) # for shift function
library(sjPlot)
library(ggplot2)
library(htmlTable)
library(emmeans)
library(gridExtra) # for subplots
library(performance)

# don't forget to give credit to the amazing authors of those packages
#citation("emmeans")

# Set the font family and size
par(family = "Arial", cex = 1.2)

# which dataset to analyze (1 => mini-block, 2 => trial-by-trial design)

expecon <- 1

####################################brain behav#####################################################

if (expecon == 1) {
  
  # expecon 1
  setwd("D:/expecon_ms")
  
  brain_behav_path <- file.path("data", "behav", "behav_df", "brain_behav_cleanpower.csv")
  
  behav = read.csv(brain_behav_path)
  
} else {
  
  # expecon 2 behavioral data
  setwd("D:/expecon_2")
  
  brain_behav_path <- file.path("behav", "brain_behav_cleanpower.csv")
  
  behav = read.csv(brain_behav_path)
  
}
################################prepare variables for linear mixed modelling #######################

# make factors for categorical variables:
behav$ID = as.factor(behav$ID) # subject ID
behav$isyes = as.factor(behav$isyes) # stimulus
behav$cue = as.factor(behav$cue) # probability for a signal
behav$prevresp = as.factor(behav$prevresp) # previous response
behav$previsyes = as.factor(behav$previsyes) # previous stimulus
behav$prevconf = as.factor(behav$prevconf) # previous confidence
behav$correct = as.factor(behav$correct) # performance
behav$prevcue = as.factor(behav$prevcue) # previous probability

# Remove NaN trials for model comparision (models neeed to have same amount of data)
behav <- na.omit(behav) 

########################### Brain behavior GLMMs ###################################################

# replace stimulus probability regressor with beta or alpha power per trial (neural correlate of stimulus probability)

summary(lmer(alpha_close_stimonset ~ cue + (cue|ID), data=behav, 
             control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5))))

summary(lmer(beta_close_stimonset ~ cue + (cue|ID), 
             data=behav, control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5))))

# expecon 2

summary(lmer(alpha_close_stimonset ~ cue + (cue|ID), data=behav, 
             control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5))))

summary(lmer(beta_close_stimonset ~ cue + (cue|ID), data=behav, 
             control=lmerControl(optimizer="bobyqa",
              optCtrl=list(maxfun=2e5))))

# alpha power per trial as regressor
alpha_int_glm_expecon2 <- glmer(sayyes ~ alpha_close_stimonset + isyes + prevresp + 
                                  alpha_close_stimonset*isyes + 
                                  alpha_close_stimonset*prevresp + 
                                  (isyes + prevresp + alpha_close_stimonset| ID),
                                data = behav, family=binomial(link='probit'), 
                                control=glmerControl(optimizer="bobyqa",
                                                     optCtrl=list(maxfun=2e5)))

# save model
if (expecon == 1) {
  cue_model_path = file.path("data", "behav", "mixed_models", "alpha_int_glm_expecon1.rda")
  saveRDS(alpha_int_glm_expecon1, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
} else {
  cue_model_path = file.path("data", "behav", "mixed_models", "alpha_int_glm_expecon2.rda")
  saveRDS(alpha_int_glm_expecon2, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
}

# beta power per trial as regressor
beta_int_glm_expecon2 <- glmer(sayyes ~ beta_close_stimonset + isyes + prevresp + 
                                 beta_close_stimonset*isyes + 
                                 beta_close_stimonset*prevresp + (isyes + prevresp| ID),
                               data = behav, family=binomial(link='probit'), 
                               control=glmerControl(optimizer="bobyqa",
                                                    optCtrl=list(maxfun=2e5)))

# Post hoc tests for behavior interaction
emm_model <- emmeans(beta_int_glm_expecon1, "prevresp", by = "beta_close_stimonset")
con <- contrast(emm_model)
con


if (expecon == 1) {
  cue_model_path = file.path("data", "behav", "mixed_models", "beta_int_glm_expecon1.rda")
  saveRDS(beta_int_glm_expecon1, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
} else {
  cue_model_path = file.path("data", "behav", "mixed_models", "beta_int_glm_expecon2.rda")
  saveRDS(alpha_int_glm_expecon2, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
}
############################### plot model estimates ###############################################

est_alpha_expecon1 = plot_model(alpha_int_glm_expecon1, type='est', 
                                title='yes response ~',
                                sort.est = TRUE, transform='plogis', show.values =TRUE, 
                                value.offset = 0.3, colors='black')

est_beta_expecon1 = plot_model(beta_int_glm_expecon1, type='est', 
                               title='yes response ~',
                               sort.est = TRUE, transform='plogis', show.values =TRUE, 
                               value.offset = 0.3, colors='black')

est_alpha_expecon2 = plot_model(alpha_int_glm_expecon2, type='est', 
                                title='yes response ~',
                                sort.est = TRUE, transform='plogis', show.values =TRUE, 
                                value.offset = 0.3, colors='black')

est_beta_expecon2 = plot_model(beta_int_glm_expecon2, type='est', 
                               title='yes response ~',
                               sort.est = TRUE, transform='plogis', show.values =TRUE, 
                               value.offset = 0.3, colors='black')

# mean plus minus one sd for continious variables

intg = arrangeGrob(est_alpha_expecon1, est_alpha_expecon2, est_beta_expecon1, est_beta_expecon2, 
                   nrow=2)

ggsave('D:\expecon_ms\figs\manuscript_figures\figure7_brain_behav\brain_behavior_est.svg',
       dpi = 300, height = 8, width = 10, plot=intg)

# plot interactions between prestimulus power and previous choice 

al_signal_int1 = plot_model(alpha_int_glm_expecon1, type='int', mdrt.values = "meansd")
al_signal_int2 = plot_model(alpha_int_glm_expecon2, type='int', mdrt.values = "meansd") 


cue_signal_int1 = plot_model(beta_int_glm_expecon1, type='int', mdrt.values = "meansd")
cue_signal_int2 = plot_model(beta_int_glm_expecon2, type='int', mdrt.values = "meansd") 

# arange plots in a grid
g = arrangeGrob(cue_signal_int1[[1]], cue_signal_int1[[2]], cue_signal_int2[[1]], cue_signal_int2[[2]], 
                nrow = 2)

# save figure
ggsave('model_brain_behav_int.svg', dpi = 300, height = 8, width = 10, plot=g)

######################## mediation #############################################################

# https://towardsdatascience.com/doing-and-reporting-your-first-mediation-analysis-in-r-2fe423b92171

total_effect <- glmer(sayyes ~ cue + prevresp + (cue + prevresp|ID), data = behav,  
                      family=binomial(link='probit'))

fit.mediator <- glmer(alpha_close_stimonset ~ cue*prevresp + (1|ID), data = behav) # only works with
# glmer model not sure why not with lmer

fit.dv <- glmer(sayyes ~ cue*isyes + alpha_close_stimonset + prevresp*cue + (1|ID),
                data = behav,
                family=binomial(link='probit'))

results_beta_expecon1 = mediate(fit.mediator, fit.dv, treat='cue', mediator='alpha_close_stimonset', boot=F)
