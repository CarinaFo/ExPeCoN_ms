#####################################ExPeCoN study#################################################
# generalized linear mixed models to estimate signal detection theory parameters
# including previous choice effects

# Paradigm 1: block design: stable environment
# 43 participants completed a behavioral and EEG study that consisted of 720 trials in total, 
# divided in 5 blocks, participants had to indicate whether they felt a weak (near-threshold) somato-
# sensory stimulus (yes/no) and give a confidence rating (sure/unsure) on each trial
# experimental manipulation: stimulus probability: high vs. low stimulus probability
# Paradigm 2: trial-by-trial design: volatile environment
# same trial structure as paradigm 1 but probability cues before each trial

# Author: Carina Forster
# Email: forster@mpg.cbs.de

# libraries
library(lme4) # mixed models
library(lmerTest) # no p values without this package for linear mixed mdoels
library(dplyr)# pandas style
library(tidyr)
library(data.table) # for shift function
library(htmlTable)
library(emmeans)
library(performance)

# don't forget to give credit to the amazing authors of those packages
#citation("emmeans")

# Set the font family and size
par(family = "Arial", cex = 1.2)

# which dataset to analyze (1 => mini-block, 2 => trial-by-trial design)

expecon <- 1

####################################################################################################
# set base directory
setwd("E:/expecon_ms")

if (expecon == 1) {
  
  # set working directory and load clean, behavioral dataframe
  behav_path <- file.path("data", "behav", "prepro_behav_data.csv")
  
  behav = read.csv(behav_path)
  
} else {
  
  behav_path <- file.path("data", "behav", "prepro_behav_data_expecon2.csv")
  
  behav = read.csv(behav_path)
  
  # ID to exclude
  ID_to_exclude <- 13
  
  # Excluding the ID from the dataframe
  behav <- behav[behav$ID != ID_to_exclude, ]
  
}
################################ linear mixed modelling ###########################################

# make factors for categorical variables:
behav$ID = as.factor(behav$ID) # subject ID
behav$isyes = as.factor(behav$isyes) # stimulus
behav$cue = as.factor(behav$cue) # probability for a signal
behav$prevresp = as.factor(behav$prevresp) # previous response
behav$previsyes = as.factor(behav$previsyes) # previous stimulus
behav$prevconf = as.factor(behav$prevconf) # previous confidence
behav$correct = as.factor(behav$correct) # performance
behav$prevcue = as.factor(behav$prevcue) # previous probability

# remove NaN column
behav <- subset(behav, select = -sayyes_y)
# Remove NaN trials for model comparision (models neeed to have same amount of data)
behav <- na.omit(behav) 
################################GLMMs##############################################################

# fit sdt model
simple_sdt_model = glmer(sayyes ~ isyes + (isyes|ID), data=behav, 
                         family=binomial(link='probit'),
                         control=glmerControl(optimizer="bobyqa",
                                              optCtrl=list(maxfun=2e5)),
)

# should replicate SDT parameters from manual calculation
# intercept => criterion
# stimulus regressor => dprime
summary(simple_sdt_model)

# now add the stimulus probability as a regressor: exclude interaction random effects: singularity
cue_model = glmer(sayyes ~ isyes+cue+isyes*cue + (isyes+cue|ID), data=behav, 
                  family=binomial(link='probit'),
                  control=glmerControl(optimizer="bobyqa",
                                       optCtrl=list(maxfun=2e5)),
)

# check model performance
check_collinearity(cue_model) # VIF should be < 3
check_convergence(cue_model)

summary(cue_model)

# Post hoc tests for interactions (estimated marginal means)
emm_model <- emmeans(cue_model, "isyes", by = "cue")
con = contrast(emm_model) # fdr corrected for 2 tests
con

# save and load model from and to disk
filename = paste("cue_model_expecon", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(cue_model, cue_model_path)
cue_model <- readRDS(cue_model_path)
########################### add previous choice predictor###########################################
cue_prev_model = glmer(sayyes ~ isyes + cue + prevresp + isyes*cue +
                         + (isyes + cue + prevresp|ID), data=behav, 
                       family=binomial(link='probit'),
                       control=glmerControl(optimizer="bobyqa",
                                            optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_model)

check_collinearity(cue_prev_model)
check_convergence(cue_prev_model)

# save the model to disk
filename = paste("cue_prev_model_expecon", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(cue_prev_model, cue_model_path)
cue_prev_model <- readRDS(cue_model_path)

# extract random effects
ranef <- ranef(cue_prev_model)
cue_ran = ranef$ID[,3] # probability cue
prev_choice_ran = ranef$ID[,4] # previous choice

# the stronger the effect of previous choice, the weaker the effect of the probability cue
cor.test(cue_ran, prev_choice_ran)

# alternator: sign. neg. correlation between cue beta weight and prev choice beta weight
cor.test(cue_ran[prev_choice_ran<0], prev_choice_ran[prev_choice_ran<0])

# repeater: no sign. correlation between cue and previous choice for repeater
cor.test(cue_ran[prev_choice_ran>0], prev_choice_ran[prev_choice_ran>0])

plot_data = data.frame(beta_probcue = cue_ran, 
                       beta_previouschoice = prev_choice_ran)

ggplot(plot_data, aes(beta_probcue, beta_previouschoice)) +
  geom_point() + 
  geom_smooth(method = "lm", se = FALSE) +  # Add a linear trend line
  labs(x = "probability condition beta weight", y = "previous choice beta weight")

################ including interaction between cue and previous choice#############################

cue_prev_int_model = glmer(sayyes ~ isyes + cue + prevresp + prevresp*cue
                           + cue*isyes +
                             (isyes + cue + prevresp|ID), data=behav, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model)

check_collinearity(cue_prev_int_model)
check_convergence(cue_prev_int_model)

# save to disk
filename = paste("cue_prev_int_model", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(cue_prev_int_model, cue_model_path)
cue_prev_int_model <- readRDS(cue_model_path)

# Post hoc tests for behavior interaction
emm_model <- emmeans(cue_prev_int_model, "prevresp", by = "cue")
con <- contrast(emm_model)
con
############################################### Model comparision ##################################

# Likelihood ratio tests
anova(cue_model, cue_prev_model)
anova(cue_prev_int_model, cue_prev_model)

# difference in AIC and BIC

diff_aic_1 = AIC(cue_prev_model) - AIC(cue_model)
diff_bic_1 = BIC(cue_prev_model) - BIC(cue_model)
print(diff_aic_1)
print(diff_bic_1)

diff_aic_2 = AIC(cue_prev_int_model) - AIC(cue_prev_model)
diff_bic_2 = BIC(cue_prev_int_model) - BIC(cue_prev_model)

print(diff_aic_2)
print(diff_bic_2)

# save table to html
table1 = sjPlot::tab_model(cue_model, cue_prev_model, cue_prev_int_model, 
                           show.aic=TRUE, show.loglik=TRUE)

filename = paste("expecon", expecon, ".html", sep="_")
output_file_path <- file.path("figs", "manuscript_figures", "Tables", filename)
htmlTable(table1, file = output_file_path)
###########################separate models for signal and noise trials#############################

signal = filter(behav, isyes==1)
noise = filter(behav, isyes==0)

cue_prev_int_model_signal = glmer(sayyes ~ prevresp + cue + prevresp*cue +
                                    (prevresp + cue|ID), data=signal, 
                                  family=binomial(link='probit'),
                                  control=glmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_signal)

check_collinearity(cue_prev_int_model_signal)
check_convergence(cue_prev_int_model_signal)

#only fitted for expecon 1
cue_model_path = file.path("data", "behav", "mixed_models", "cue_prev_int_model_signal_1.rda")
saveRDS(cue_prev_int_model_signal, cue_model_path)
cue_prev_int_model_signal <- readRDS(cue_model_path)

# noise model
cue_prev_int_model_noise = glmer(sayyes ~ prevresp + cue + prevresp*cue +
                                   (prevresp + cue|ID), data=noise, 
                                 family=binomial(link='probit'),
                                 control=glmerControl(optimizer="bobyqa",
                                                      optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_noise)

check_collinearity(cue_prev_int_model_noise)
check_convergence(cue_prev_int_model_noise)

cue_model_path = file.path("data", "behav", "mixed_models", "cue_prev_int_model_noise_1.rda")
saveRDS(cue_prev_int_model_noise, cue_model_path)
cue_prev_int_model_noise <- readRDS(cue_model_path)

##################separate model for confident/unconfident previous response#######
acc_conf_model = glmer(correct ~ conf + (conf|ID), data=behav, 
                       family=binomial(link='probit'),
                       control=glmerControl(optimizer="bobyqa",
                                            optCtrl=list(maxfun=2e5)))

summary(acc_conf_model)

conf = filter(behav, prevconf==1)
unconf = filter(behav, prevconf==0)

cue_prev_int_model_conf = glmer(sayyes ~ isyes + prevresp + cue + isyes*cue+prevresp*cue +
                                  (isyes + prevresp + cue|ID), data=conf, 
                                family=binomial(link='probit'),
                                control=glmerControl(optimizer="bobyqa",
                                                     optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_conf)
check_convergence(cue_prev_int_model_conf)

summary(cue_prev_int_model_conf)

cue_model_path = file.path("data", "behav", "mixed_models", "cue_prev_int_model_conf_1.rda")
saveRDS(cue_prev_int_model_conf, cue_model_path)
cue_prev_int_model_conf <- readRDS(cue_model_path)


cue_prev_int_model_unconf = glmer(sayyes ~ isyes + prevresp + cue + isyes*cue+prevresp*cue +
                                    (isyes + prevresp + cue|ID), 
                                  data=unconf, 
                                  family=binomial(link='probit'),
                                  control=glmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_unconf)
check_convergence(cue_prev_int_model_unconf)

summary(cue_prev_int_model_unconf)

# save and load model from and to disk
cue_model_path = file.path("data", "behav", "mixed_models", "cue_prev_int_model_unconf_1.rda")
saveRDS(cue_prev_int_model_unconf, cue_model_path)
cue_prev_int_model_unconf <- readRDS(cue_model_path)

# save model summary as table
table2 = sjPlot::tab_model(cue_prev_int_model_signal, cue_prev_int_model_noise,
                           cue_prev_int_model_conf, cue_prev_int_model_unconf, 
                           show.aic=TRUE, show.loglik=TRUE)

filename = paste("expecon_interaction_control_models", expecon, ".html", sep="_")
output_file_path <- file.path("figs", "manuscript_figures", "Tables", filename)
htmlTable(table2, file = output_file_path)
############################## Unused functions (not code reviewed)###################################################


# test for autocorrelation between trials?
high = filter(behav, cue==0.75)
low = filter(behav, cue==0.25)

high$ID = as.factor(high$ID)
high$cue = as.factor(high$cue)
high$isyes = as.factor(high$isyes)
high$prevsayyes = as.factor(high$prevsayyes)

auto_model = glmer(isyes ~ previsyes*cue+(previsyes*cue|ID), data=behav, 
                   family=binomial(link='probit'),
)

summary(auto_model)

# Calculate autocorrelation
autocorrelation <- acf(behav$isyes, plot = FALSE)

# Check for significant autocorrelation
significant_lags <- which(abs(autocorrelation$acf) > 2 / sqrt(length(data)))

if (length(significant_lags) > 0) {
  print("Significant autocorrelation detected at lag(s):")
  print(significant_lags)
} else {
  print("No significant autocorrelation detected.")
}

# No significant autocorrelation between trials

########################### confidence as dependent variable ########################################

# correct trials only
behav$correct = as.numeric(behav$correct)

correct_trial_only = filter(behav, correct==2)

conf_cue_prev_model = glmer(conf ~ isyes + cue + prevconf + isyes*cue + prevconf*cue +
                              (isyes + cue + prevconf|ID), data=correct_trial_only, 
                            family=binomial(link='probit'),
                            control=glmerControl(optimizer="bobyqa",
                                                 optCtrl=list(maxfun=2e5)),
)

summary(conf_cue_prev_model)
# congruency effect: higher confidence in signal trials in high prob. condition

plot_model(conf_cue_prev_model, 'int')

# Post hoc tests for behavior interaction
emm_model <- emmeans(conf_cue_prev_model, "cue", by = "isyes")
con <- contrast(emm_model)
con

emm_model <- emmeans(conf_cue_prev_model, "prevconf", by = "cue")
con <- contrast(emm_model)
con


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

expecon <- 2

####################################################################################################

if (expecon == 1) {
  
  # expecon 1
  setwd("D:/expecon_ms")
  
  # set working directory and load clean, behavioral dataframe
  behav_path <- file.path("data", "behav", "behav_df", "prepro_behav_data.csv")
  
  behav = read.csv(behav_path)
  
} else {
  
  # expecon 2 behavioral data
  setwd("D:/expecon_2/behav")
  
  behav_path <- file.path("prepro_behav_data.csv")
  
  behav = read.csv(behav_path)
  
  # ID to exclude
  ID_to_exclude <- 13
  
  # Excluding the ID from the dataframe
  behav <- behav[behav$ID != ID_to_exclude, ]
  
}

################################ linear mixed modelling ###########################################

# make factors for categorical variables:
behav$ID = as.factor(behav$ID) # subject ID
behav$isyes = as.factor(behav$isyes) # stimulus
behav$cue = as.factor(behav$cue) # probability for a signal
behav$prevresp = as.factor(behav$prevresp) # previous response
behav$previsyes = as.factor(behav$previsyes) # previous stimulus
behav$prevconf = as.factor(behav$prevconf) # previous confidence
behav$correct = as.factor(behav$correct) # performance
behav$prevcue = as.factor(behav$prevcue) # previous probability

# remove NaN column
behav <- subset(behav, select = -sayyes_y)
# Remove NaN trials for model comparision (models neeed to have same amount of data)
behav <- na.omit(behav) 
################################GLMMs##############################################################

# fit sdt model
simple_sdt_model = glmer(sayyes ~ isyes + (isyes|ID), data=behav, 
                                      family=binomial(link='probit'),
                                      control=glmerControl(optimizer="bobyqa",
                                                           optCtrl=list(maxfun=2e5)),
)

# should replicate SDT parameters from manual calculation
# intercept => criterion
# stimulus regressor => dprime
summary(simple_sdt_model)

# now add the stimulus probability as a regressor
cue_model = glmer(sayyes ~ isyes+cue+isyes*cue + (isyes+cue|ID), data=behav, 
                  family=binomial(link='probit'),
                  control=glmerControl(optimizer="bobyqa",
                  optCtrl=list(maxfun=2e5)),
                  )

# check model performance
check_collinearity(cue_model) # VIF should be < 3
check_convergence(cue_model)

summary(cue_model)

# Post hoc tests for interactions (estimated marginal means)
emm_model <- emmeans(cue_model, "isyes", by = "cue")
con = contrast(emm_model) # fdr corrected for 2 tests
con

# save and load model from and to disk

if (expecon == 1) {
  
  cue_model_path = file.path("data", "behav", "mixed_models", "cue_model_expecon1.rda")
  saveRDS(cue_model, cue_model_path)
  cue_model <- readRDS(cue_model_path)
  
} else {
  
  cue_model_path = file.path("data", "behav", "mixed_models", "cue_model_expecon2.rda")
  saveRDS(cue_model, cue_model_path)
  cue_model <- readRDS(cue_model_path)
  
}

########################### add previous choice predictor###########################################

cue_prev_model = glmer(sayyes ~ isyes + cue + prevresp + isyes*cue +
                           + (isyes + cue + prevresp|ID), data=behav, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_model)

check_collinearity(cue_prev_model)
check_convergence(cue_prev_model)


if (expecon == 1) {
  cue_model_path = file.path("data", "behav", "mixed_models", "cue_prev_model_expecon1.rda")
  saveRDS(cue_prev_model, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
} else {
  cue_model_path = file.path("data", "behav", "mixed_models", "cue_prev_model_expecon2.rda")
  saveRDS(cue_prev_model, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
}

ranef <- ranef(cue_prev_model)
cue_ran = ranef$ID[,3]
prev_choice_ran = ranef$ID[,4]

# alternator: correlation between cue beta weight and prev choice beta weight
cor.test(cue_ran[prev_choice_ran<0], prev_choice_ran[prev_choice_ran<0])

# repeater
cor.test(cue_ran[prev_choice_ran>0], prev_choice_ran[prev_choice_ran>0])

plot_data = data.frame(beta_probcue = cue_ran[prev_choice_ran>0], 
                       beta_previouschoice = prev_choice_ran[prev_choice_ran>0])

ggplot(plot_data, aes(beta_probcue, beta_previouschoice)) +
  geom_point() + 
  geom_smooth(method = "lm", se = FALSE) +  # Add a linear trend line
  labs(x = "probability condition beta weight", y = "previous choice beta weight")

# cue is still a significant predictor but less strong effect

################ including interaction between cue and previous choice#############################

cue_prev_int_model = glmer(sayyes ~ isyes + cue + prevresp + prevresp*cue
                                        + cue*isyes +
                                        (isyes + cue + prevresp|ID), data=behav, 
                                      family=binomial(link='probit'),
                                      control=glmerControl(optimizer="bobyqa",
                                      optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model)

check_collinearity(cue_prev_int_model)
check_convergence(cue_prev_int_model)

if (expecon == 1) {
  cue_model_path = file.path("data", "behav", "mixed_models", "cue_prev_int_model_expecon1.rda")
  saveRDS(cue_prev_int_model, cue_model_path)
  cue_prev_int_model <- readRDS(cue_model_path)
} else {
  cue_model_path = file.path("data", "behav", "mixed_models", "cue_prev_int_model_expecon2.rda")
  saveRDS(cue_prev_int_model, cue_model_path)
  cue_prev_int_model <- readRDS(cue_model_path)
}

# Post hoc tests for behavior interaction
emm_model <- emmeans(cue_prev_int_model, "prevresp", by = "cue")
con <- contrast(emm_model)
con
############################################### Model comparision ##################################

# Likelihood ratio tests
anova(cue_model, cue_prev_model)
anova(cue_prev_int_model, cue_prev_model)

# difference in AIC and BIC

diff_aic_1 = AIC(cue_prev_model) - AIC(cue_model)
diff_bic_1 = BIC(cue_prev_model) - BIC(cue_model)
print(diff_aic_1)
print(diff_bic_1)

diff_aic_2 = AIC(cue_prev_int_model) - AIC(cue_prev_model)
diff_bic_2 = BIC(cue_prev_int_model) - BIC(cue_prev_model)

print(diff_aic_2)
print(diff_bic_2)

# save table to html
table1 = sjPlot::tab_model(cue_model, cue_prev_model, cue_prev_int_model, 
                           show.aic=TRUE, show.loglik=TRUE)
# mini block design only
table2 = sjPlot::tab_model(cue_prev_int_model_signal, cue_prev_int_model_noise,
                           cue_prev_int_model_conf, cue_prev_int_model_unconf, 
                           show.aic=TRUE, show.loglik=TRUE)

output_file_path <- file.path("figs", "manuscript_figures", "Tables", "expecon_2.html")
htmlTable(table1, file = output_file_path)
################################Plot estimates and interactions#####################################

theme_set(theme_sjplot())
# expecon 1
setwd("D:/expecon_ms")

save_path = file.path("figs", "manuscript_figures", "figure7_brain_behav")
setwd(save_path)

est_cue1 = plot_model(cue_model, type='est', 
                      title='yes response ~',
                      sort.est = TRUE, transform='plogis', show.values =TRUE, 
                      value.offset = 0.3, colors='black')

est_cue2 = plot_model(cue_model2, type='est', 
                      title='yes response ~',
                      sort.est = TRUE, transform='plogis', show.values =TRUE, 
                      value.offset = 0.3, colors='black')

est_cue_prev1 = plot_model(cue_prev_model, type='est',
                           sort.est = TRUE, transform='plogis', show.values =TRUE, 
                           value.offset = 0.3, colors='black')

est_cue_prev2 = plot_model(cue_prev_model2, type='est', 
                           sort.est = TRUE, transform='plogis', show.values =TRUE, 
                           value.offset = 0.3, colors='black')


est_cue_prev_int1 = plot_model(cue_prev_int_model, type='est',
                               sort.est = TRUE, transform='plogis', show.values =TRUE, 
                               value.offset = 0.3, colors='black')+
ylab("Probabilities")


est_cue_prev_int2 = plot_model(cue_prev_int_model2, type='est', 
                               sort.est = TRUE, transform='plogis', show.values =TRUE, 
                               value.offset = 0.3, colors='black') +
  ylab("Probabilities")

# arange plots in a grid
g = arrangeGrob(est_cue1, est_cue2, est_cue_prev1, est_cue_prev2, est_cue_prev_int1,
             est_cue_prev_int2, nrow = 3)

# save figure
ggsave('model_estimates.svg', dpi = 300, height = 8, width = 10, plot=g)


est_alpha_expecon1 = plot_model(alpha_int_glm, type='est', 
                      title='yes response ~',
                      sort.est = TRUE, transform='plogis', show.values =TRUE, 
                      value.offset = 0.3, colors='black')

est_beta_expecon1 = plot_model(beta_int_glm, type='est', 
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

ggsave('model_brain_behavior.svg', dpi = 300, height = 8, width = 10, plot=intg)

cue_signal_int1 = plot_model(beta_int_glm, type='int', mdrt.values = "meansd")
cue_signal_int2 = plot_model(beta_int_glm_expecon2, type='int', mdrt.values = "meansd") 

# arange plots in a grid
g = arrangeGrob(cue_signal_int1[[1]], cue_signal_int1[[2]], cue_signal_int2[[1]], 
                cue_signal_int2[[2]], 
                nrow = 2)

# save figure
ggsave('model_brain_behav_int.svg', dpi = 300, height = 8, width = 10, plot=g)

cue_signal_prev_int1 = plot_model(cue_prev_int_model, type='pred',
                                  terms = c("prevresp", "cue"))

cue_signal_prev_int2 = plot_model(cue_prev_int_model, type='pred', terms = c("prevresp", "cue"))

# change order of variables after model fitting
plot_model(beta_int_glm, type = "pred", 
           terms = c("isyes", "beta_close_stimonset"))

###########################separate models for signal and noise trials#############################

signal = filter(behav, isyes==1)
noise = filter(behav, isyes==0)

cue_prev_int_model_signal = glmer(sayyes ~ prevresp + cue + prevresp*cue +
                             (prevresp + cue|ID), data=signal, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_signal)

check_collinearity(cue_prev_int_model_signal)
check_convergence(cue_prev_int_model_signal)


if (expecon == 1) {
  cue_model_path = file.path("data", "behav", "mixed_models", 
                             "cue_prev_int_model_signal_expecon1.rda")
  saveRDS(cue_prev_int_model_signal, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
} else {
  cue_model_path = file.path("data", "behav", "mixed_models", 
                             "cue_prev_int_model_signal_expecon2.rda")
  saveRDS(cue_prev_int_model_signal, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
}

# noise model

cue_prev_int_model_noise = glmer(sayyes ~ prevresp + cue + prevresp*cue +
                                    (prevresp + cue|ID), data=noise, 
                                  family=binomial(link='probit'),
                                  control=glmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_noise)

emmeans::emmeans(cue_prev_int_model_noise, ~ cue * prevsayyes)

check_collinearity(cue_prev_int_model_noise)
check_convergence(cue_prev_int_model_noise)

if (expecon == 1) {
  cue_model_path = file.path("data", "behav", "mixed_models", 
                             "cue_prev_int_model_noise_expecon1.rda")
  saveRDS(cue_prev_int_model_noise, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
} else {
  cue_model_path = file.path("data", "behav", "mixed_models", 
                             "cue_prev_int_model_noise_expecon2.rda")
  saveRDS(cue_prev_int_model_noise, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
}
##################separate model for confident/unconfident previous response#######
acc_conf_model = glmer(correct ~ conf + (conf|ID), data=behav, 
                       family=binomial(link='probit'),
                       control=glmerControl(optimizer="bobyqa",
                                            optCtrl=list(maxfun=2e5)))

summary(acc_conf_model)

conf = filter(behav, prevconf==1)
unconf = filter(behav, prevconf==0)

cue_prev_int_model_conf = glmer(sayyes ~ isyes + prevresp + cue + isyes*cue+prevresp*cue +
                             (isyes + prevresp + cue|ID), data=conf, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_conf)
check_convergence(cue_prev_int_model_conf)

summary(cue_prev_int_model_conf)

if (expecon == 1) {
  cue_model_path = file.path("data", "behav", "mixed_models", 
                             "cue_prev_int_model_conf_expecon1.rda")
  saveRDS(cue_prev_int_model_conf, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
} else {
  cue_model_path = file.path("data", "behav", "mixed_models", 
                             "cue_prev_int_model_conf_expecon2.rda")
  saveRDS(cue_prev_int_model_conf, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
}

cue_prev_int_model_unconf = glmer(sayyes ~ isyes + prevresp + cue + isyes*cue+prevresp*cue +
                             (isyes + prevresp + cue|ID), 
                             data=unconf, 
                             family=binomial(link='probit'),
                             control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_unconf)
check_convergence(cue_prev_int_model_unconf)

summary(cue_prev_int_model_unconf)

# save and load model from and to disk

if (expecon == 1) {
  cue_model_path = file.path("data", "behav", "mixed_models", 
                             "cue_prev_int_model_unconf_expecon1.rda")
  saveRDS(cue_prev_int_model_unconf, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
} else {
  cue_model_path = file.path("data", "behav", "mixed_models", 
                             "cue_prev_int_model_unconf_expecon2.rda")
  saveRDS(cue_prev_int_model_unconf, cue_model_path)
  cue_prev_model <- readRDS(cue_model_path)
}
############################## Unused functions ###################################################
# not code reviewed

# test for autocorrelation between trials?
high = filter(behav, cue==0.75)
low = filter(behav, cue==0.25)

high$ID = as.factor(high$ID)
high$cue = as.factor(high$cue)
high$isyes = as.factor(high$isyes)
high$prevsayyes = as.factor(high$prevsayyes)

auto_model = glmer(isyes ~ previsyes*cue+(previsyes*cue|ID), data=behav, 
                   family=binomial(link='probit'),
)

summary(auto_model)

# Calculate autocorrelation
autocorrelation <- acf(behav$isyes, plot = FALSE)

# Check for significant autocorrelation
significant_lags <- which(abs(autocorrelation$acf) > 2 / sqrt(length(data)))

if (length(significant_lags) > 0) {
  print("Significant autocorrelation detected at lag(s):")
  print(significant_lags)
} else {
  print("No significant autocorrelation detected.")
}

# No significant autocorrelation between trials

########################### confidence as dependent variable #######################################

# correct trials only
behav$correct = as.numeric(behav$correct)

correct_trial_only = filter(behav, correct==2)

conf_cue_prev_model = glmer(conf ~ isyes + cue + prevconf + isyes*cue + prevconf*cue +
                                 (isyes + cue + prevconf|ID), data=correct_trial_only, 
                               family=binomial(link='probit'),
                               control=glmerControl(optimizer="bobyqa",
                                                    optCtrl=list(maxfun=2e5)),
)

summary(conf_cue_prev_model)
# congruency effect: higher confidence in signal trials in high prob. condition

plot_model(conf_cue_prev_model, 'int')

# Post hoc tests for behavior interaction
emm_model <- emmeans(conf_cue_prev_model, "cue", by = "isyes")
con <- contrast(emm_model)
con

emm_model <- emmeans(conf_cue_prev_model, "prevconf", by = "cue")
con <- contrast(emm_model)
con
