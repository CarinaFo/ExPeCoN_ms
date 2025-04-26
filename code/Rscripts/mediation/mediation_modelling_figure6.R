# this script fits a mediation model to brain behaviour data from Forster et al., 2025
# script produces vaues for figure 6 (the graphical model was created in inkscape)

# author: Carina Forster
# email: forster@mpg.cbs.de

# libraries
library(lme4) # mixed models
library(ggplot2)
library(mediation)
library(dplyr)# pandas style
library(tidyr)
library(emmeans)
library(ggeffects)
library(sjPlot)
library(modelsummary)
library(performance) # for Nakagawa conditional/marginal R2
library(partR2) # for part R2 values
library(glmmTMB)
library(bmlm) # bayesian mediation model

# don't forget to give credit to the amazing authors of those packages
#citation("emmeans")

# Set the font family and size
par(family = "Arial", cex = 1.2)

# skip scientific notation
options(scipen=999)

###################load data#######################################
setwd("E:/expecon_ms")
study1 = "brain_behav_cleaned_1.csv"
study2 = "brain_behav_cleaned_2.csv"

brain_behav_path_1 <- file.path("data", "behav", study1)
brain_behav_path_2 <- file.path("data", "behav", study2)

behav = read.csv(brain_behav_path_1)
################################ prep for modelling ###########################################

# make factors for categorical variables:
behav$ID = as.factor(behav$ID) # subject ID

# Remove NaN trials for model comparision (models neeed to have same amount of data)
behav$level_0 <- NULL
behav <- na.omit(behav) 

#rename beta power variable
behav$beta <- behav$beta_source_prob
behav$beta_prev <- behav$beta_source_prev

hist(behav$beta)

# kick out very high or low beta values
behav <- behav[!(behav$beta > 3 | behav$beta < -3), ]
behav <- behav[!(behav$beta_prev > 3 | behav$beta_prev < -3), ]
############################## mediation model ####################################################

# does beta power mediate probability and/or previous response?

# Check the version of a specific package (e.g., "ggplot2")
packageVersion("mediation")

################################prepare variables for linear mixed modelling #######################

# regressor variables can not be a factor for mediation analysis

######################## mediation #############################################################

# https://towardsdatascience.com/doing-and-reporting-your-first-mediation-analysis-in-r-2fe423b92171

####################################### volatile env.##############################################
any(is.na(behav)) ## returns FALSE

# without p-values, model for mediation function
med.model_beta_prob <- lme4::lmer(beta ~ cue + 
                                     (1 + cue|ID), 
                                   data = behav,
                                   control=lmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)))
summary(med.model_beta_prob)


med.model_beta_prev <- lme4::lmer(beta_prev ~  prevresp + 
                                    (1 + prevresp|ID), 
                             data = behav,
                             control=lmerControl(optimizer="bobyqa",
                                                 optCtrl=list(maxfun=2e5))) # significant 
summary(med.model_beta_prev)

# fit outcome model: do the mediator (beta) and the IV (stimulus probability cue) predict the
# detection response? included stimulus and previous choice at a given trial as covariates,
# but no interaction between prev. resp and cue
out.model_beta_prob <- glmer(sayyes ~ beta + cue + isyes +
                          (1 + cue + isyes|ID),
                        data = behav,
                        control=glmerControl(optimizer="bobyqa",
                                             optCtrl=list(maxfun=2e5)),
                        family=binomial(link='probit'))

summary(out.model_beta_prob)

out.model_beta_prev <- glmer(sayyes ~ beta_prev + prevresp + isyes +
                          (1 + isyes + prevresp|ID),
                        data = behav,
                        control=glmerControl(optimizer="bobyqa",
                                             optCtrl=list(maxfun=2e5)),
                        family=binomial(link='probit'))

summary(out.model_beta_prev)


# save models as tables for manuscript
#https://modelsummary.com/articles/modelsummary.html

filename_med = paste("mediation_expecon_beta", expecon, ".docx", sep="_")
output_file_path_med <- file.path("data", "behav", "mediation", filename_med)

models = list("probability" = out.model_beta_prob, "previous_response" = out.model_beta_prev)

modelsummary::modelsummary(models, estimate  = "{estimate} [{conf.low}, {conf.high}], {stars}", 
                           statistic = NULL,  output = output_file_path_med)

mediation_cue_beta_prob <- mediate(med.model_beta_prob, out.model_beta_prob, treat='cue', 
                              mediator='beta')

summary(mediation_cue_beta_prob)

mediation_cue_beta_prev <- mediate(med.model_beta_prev, out.model_beta_prev, treat='prevresp', 
                              mediator='beta_prev')

summary(mediation_cue_beta_prev)