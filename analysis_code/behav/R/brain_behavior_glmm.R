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
library(sjPlot)
library(ggplot2)
library(htmlTable)
library(emmeans)
library(gridExtra) # for subplots
library(performance)
library(brms)

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

summary(lmer(pre_alpha ~ cue + (cue|ID), data=behav, 
             control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5))))

summary(lmer(pre_beta ~ cue + (cue|ID), 
             data=behav, control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5))))

summary(lmer(pre_beta ~ isyes, data=behav))

# alpha power per trial as regressor
alpha_int_glm <- glmer(sayyes ~ pre_alpha + isyes + prevresp + 
                                   pre_alpha*prevresp +
                                  (isyes + prevresp + pre_alpha|ID),
                                data = behav, family=binomial(link='probit'), 
                                control=glmerControl(optimizer="bobyqa",
                                                     optCtrl=list(maxfun=2e5)))
summary(alpha_int_glm)

# beta power per trial as regressor
beta_int_glm <- glmer(sayyes ~ pre_beta+isyes + prevresp + 
                        pre_beta*isyes + 
                        pre_beta*prevresp + (isyes + prevresp| ID),
                      data = behav, family=binomial(link='probit'), 
                      control=glmerControl(optimizer="bobyqa",
                                           optCtrl=list(maxfun=2e5)))

summary(beta_int_glm)

# Post hoc tests for behavior interaction
emm_model <- emmeans(beta_int_glm, "prevresp", by = "pre_beta")
con <- contrast(emm_model)
con

# save model
if (expecon == 1) {
  cue_model_path = file.path("data", "behav", "mixed_models", "alpha_int_glm_expecon1.rda")
  saveRDS(alpha_int_glm, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
} else {
  cue_model_path = file.path("behav", "mixed_models", "alpha_int_glm_expecon2.rda")
  saveRDS(alpha_int_glm, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
}


if (expecon == 1) {
  cue_model_path = file.path("data", "behav", "mixed_models", "beta_int_glm_expecon1.rda")
  saveRDS(beta_int_glm, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
} else {
  cue_model_path = file.path("behav", "mixed_models", "beta_int_glm_expecon2.rda")
  saveRDS(alpha_int_glm, cue_model_path)
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

est_beta_expecon2 = plot_model(beta_int_glm, type='est', 
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


cue_signal_int1 = plot_model(beta_int_glm, type='int', mdrt.values = "meansd")
cue_signal_int2 = plot_model(beta_int_glm_expecon2, type='int', mdrt.values = "meansd") 

# arange plots in a grid
g = arrangeGrob(cue_signal_int1[[1]], cue_signal_int1[[2]], cue_signal_int2[[1]], cue_signal_int2[[2]], 
                nrow = 2)

# save figure
ggsave('model_brain_behav_int.svg', dpi = 300, height = 8, width = 10, plot=g)

######################## mediation #############################################################

# https://towardsdatascience.com/doing-and-reporting-your-first-mediation-analysis-in-r-2fe423b92171

behav$beta <- behav$pre_beta

total_effect <- glmer(sayyes ~ cue + prevresp + (cue + prevresp|ID), data = behav,  
                      family=binomial(link='probit'))

# fit mediator
mediatorbeta  <- lme4::lmer(beta ~ cue + (-1|ID), data = behav)
summary(mediatorbeta)

mediatoralpha <- lme4::lmer(pre_alpha ~ cue + (-1 + cue|ID), data = behav)
summary(fit.mediator_alpha)

# only works with glmer model not sure why not with lmer
# fit whole model
dvalpha <- glmer(sayyes ~ cue+ pre_alpha + (cue|ID),data = behav,
                      family=binomial(link='probit'))

dvbeta <- glmer(sayyes ~ cue + beta + (1|ID),data = behav,
                     family=binomial(link='probit'))

# mediation
results_alpha_expecon2 <- mediate(mediator_alpha,  dv_alpha, treat='cue', mediator='pre_alpha')

results_beta_expecon2 <- mediate(mediator_beta, dv_beta, treat='cue', mediator='beta')

summary(results_alpha_expecon2)
summary(results_beta_expecon2)

############################ mediation with nlme ##################################################

# Copy the column_to_copy into a new column named copied_column
behav$y <- behav$sayyes
behav$x <- behav$cue
behav$m <- behav$pre_beta

# Making data longer by putting m and y into a single column, z
datalong <- pivot_longer(data = behav,
                         cols = c("m", "y"), # variables we want to combine
                         names_to = "dv", # column that has the variable names
                         values_to = "z") # column with m and y values

#adding the double indicators 
datalong$dy <- ifelse(datalong$dv == "y", 1, 0)
datalong$dm <- ifelse(datalong$dv == "m", 1, 0)
datalong$dvnum <- ifelse(datalong$dv == "m", 1, 0)

#look at updated data set
head(datalong, 10)

library(nlme) #for multilevel models

#lme mediation model
model_lme <- lme(fixed = z ~  
                   dm + dm:cue + dm:prevresp + # + #m as outcome
                   dy + dy:pre_beta + dy:cue, #+ dy:prevresp, #y as outcome
                 random = ~  dm:cue + dy:pre_beta + dy:cue | ID, 
                 weights = varIdent(form = ~ 1 | dvnum), #separate sigma^{2}_{e} for each outcome
                 data = datalong,
                 na.action = na.exclude,
                 control = lmeControl(opt = "optim", maxIter = 200, msMaxIter = 200, niterEM = 50, msMaxEval = 400))

summary(model_lme)

# fit model with brms
# for documentation:
#https://thechangelab.stanford.edu/intensive-longitudinal-analysis/t-specifying-1-1-1-mediation-models-r/

#xm <- bf(fwkdiscw ~ -1 + timec + fwkstrcw + (-1 + fwkstrcw |p| id))

# mediator is the outcome (p allows to calculate covariance between a and b path)
mediation_model <- bf(pre_beta ~ cue + (-1 + cue |p|ID))

# my <- bf(freldiscw ~  -1 + timec + fwkstrcw + fwkdiscw + (-1 + fwkstrcw + fwkdiscw |p| id))

# dv is the outcome
main_effect_model <- bf(sayyes ~ isyes + cue + pre_beta + isyes*cue + 
                          (isyes + cue + pre_beta|p| ID),
                       family=bernoulli(link='probit'))

library(bmlm) #for Bayesian 1-1-1 mediation

fit_brm_expecon_1_noprevresp  <- brm(mediation_model + main_effect_model + set_rescor(FALSE),
                          data = behav,
                          iter= 2000,
                          cores = getOption("mc.cores", 12))

brms_mediation_path = file.path("D:", "expecon_ms", "data", "behav", "mixed_models", "brms", "mediation", "fit_expecon1_inclprevresp")

saveRDS(fit_brm_expecon_1, file = brms_mediation_path)

print(fit_brm_expecon_1_noprevresp,digits = 4)

# compute indirect effect
cortocov <- function (r, var1, var2) {
  cov=r*((var1*var2)^0.5)
  return(cov)
}

## use the posterior_samples() function to pull out the posterior distributions for all model parameters
## these are all possible effect sizes for each parameter
med_post <- posterior_samples(fit_brm_expecon_1_noprevresp)

# plug in SD and correlation corresponding to a and b paths for cortocov function

med_post$covab <- cortocov(
  # vector of posterior samples corresponding to correlation of a and b paths
  med_post$cor_ID__prebeta_cue__sayyes_cue,
  
  # vector of posterior samples corresponding to SD of a path --> convert to variance
  med_post$sd_ID__prebeta_cue^2,
  
  # vector of posterior samples corresponding to SD of b path --> convert to variance
  med_post$sd_ID__sayyes_cue^2)

round(mean(med_post$covab), digits = 3) # rounds to .03

indirect_effect <- 
  med_post$b_prebeta_cue*  # a path
  med_post$b_sayyes_cue +  # b path
  med_post$covab   # cov(a, b) as calculated above

# indirect effect if 95 % CI does not include 0
round(quantile(indirect_effect, probs = c(.025, .5, .975)), digits = 2) 

round(mean(indirect_effect), digits = 3)