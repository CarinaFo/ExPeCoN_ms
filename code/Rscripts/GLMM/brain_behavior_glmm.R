#####################################ExPeCoN study#################################################
# generalized linear mixed models to estimate signal detection theory parameters
# including previous choice effects: Brain-behavior modelling

# author: Carina Forster
# email: forster@mpg.cbs.de

# libraries
library(lme4) # mixed models
library(lmerTest) # no p values without this package for linear mixed mdoels
library(dplyr)# pandas style
library(tidyr)
library(data.table) # for shift function
library(htmlTable)
library(emmeans)
library(performance)
library(brms)

# don't forget to give credit to the amazing authors of those packages
#citation("emmeans")

# Set the font family and size
par(family = "Arial", cex = 1.2)

# which dataset to analyze (1 => mini-block, 2 => trial-by-trial design)

expecon <- 1

####################################brain behav#####################################################
setwd("E:/expecon_ms")

filename = paste("brain_behav_cleaned_", expecon, ".csv", sep="")
brain_behav_path <- file.path("data", "behav", filename)

behav = read.csv(brain_behav_path)
  
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
behav$congruency <- as.integer(as.logical(behav$congruency))
behav$congruency_stim <- as.integer(as.logical(behav$congruency_stim))

# Remove NaN trials for model comparision (models neeed to have same amount of data)
behav <- na.omit(behav) 

########################### Brain behavior GLMMs ###################################################

# replace stimulus probability regressor with beta or alpha power per trial (neural correlate of stimulus probability)

summary(lmer(pre_alpha ~ cue + (cue|ID), data=behav, 
             control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5))))

summary(lmer(pre_beta ~ cue + (cue|ID), 
             data=behav, control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5))))

summary(lmer(pre_beta ~ isyes + (isyes|ID), data=behav))

# the cue sign. predicts beta but not alpha

### does prestimulus power predict detection, while controlling for previous choice
alpha_glm <- glmer(sayyes ~ pre_alpha + isyes + prevresp + 
                         pre_alpha*isyes +
                         (isyes + prevresp + pre_alpha|ID),
                       data = behav, family=binomial(link='probit'), 
                       control=glmerControl(optimizer="bobyqa",
                                            optCtrl=list(maxfun=2e5)))
summary(alpha_glm)

beta_glm <- glmer(sayyes ~ pre_beta + isyes + prevresp + 
                     pre_beta*isyes +
                     (isyes + prevresp|ID),
                   data = behav, family=binomial(link='probit'), 
                   control=glmerControl(optimizer="bobyqa",
                                        optCtrl=list(maxfun=2e5)))
summary(beta_glm)


# save models to disk
filename = paste("alpha_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(alpha_glm, cue_model_path)
cue_prev_model <- readRDS(cue_model_path)


filename = paste("beta_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(beta_glm, cue_model_path)
cue_prev_model <- readRDS(cue_model_path)

##### no we fit the interaction between prestimulus power and previous choice

# alpha interaction
alpha_int_glm <- glmer(sayyes ~ pre_alpha + isyes + prevresp + 
                                pre_alpha*isyes +
                                pre_alpha*prevresp + 
                                (isyes + prevresp + pre_alpha|ID),
                                data = behav, family=binomial(link='probit'), 
                                control=glmerControl(optimizer="bobyqa",
                                                     optCtrl=list(maxfun=2e5)))
summary(alpha_int_glm)

# beta interaction
beta_int_glm <- glmer(sayyes ~ pre_beta + isyes + prevresp + 
                        pre_beta*isyes + 
                        pre_beta*prevresp + 
                        (isyes + prevresp| ID),
                      data = behav, family=binomial(link='probit'), 
                      control=glmerControl(optimizer="bobyqa",
                                           optCtrl=list(maxfun=2e5)))

summary(beta_int_glm)

# Post hoc tests for behavior interaction
emm_model <- emmeans(beta_int_glm, "prevresp", by = "pre_beta")
con <- contrast(emm_model)
con

# save models to disk
filename = paste("alpha_int_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(alpha_int_glm, cue_model_path)
cue_prev_model <- readRDS(cue_model_path)


filename = paste("beta_int_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(beta_int_glm, cue_model_path)
cue_prev_model <- readRDS(cue_model_path)

##########################congruency###############################################################

# does beta power per trial predict congruent responses in both probability conditions?

con_beta = glmer(congruency ~ pre_beta * cue + isyes + (cue|ID), data=behav, 
                 family=binomial(link='probit'), 
                 control=glmerControl(optimizer="bobyqa",
                                      optCtrl=list(maxfun=2e5)))
summary(con_beta)

# Post hoc tests for behavior interaction
emm_model <- emmeans(con_beta, "cue", by = "pre_beta")
con <- contrast(emm_model)
con

con_plot = plot_model(con_beta, type='int', mdrt.values = "meansd")
ggsave('congruency_model_1.svg', dpi = 300, height = 8, width = 10, plot=con_plot)