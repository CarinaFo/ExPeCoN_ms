# fit generalized linear mixed effects models with brms

library(brms)
library(rstan)
library(parallel)
library(tidybayes)
library(car)
library(svglite) # save svg file
library(dplyr)
library(sjPlot)
library(htmlTable)

# how many cores for brms model fitting?
num_cores <- detectCores()

# make rstan quicker
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

# https://github.com/stan-dev/rstan/wiki/Configuring-C---Toolchain-for-Windows

# Set the font family and size

par(family = "Arial", cex = 1.2)

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
####################################################################################################

# make factors:
behav$ID = as.factor(behav$ID)
behav$isyes = as.factor(behav$isyes)
behav$cue = as.factor(behav$cue)
behav$prevresp = as.factor(behav$prevresp)
behav$previsyes = as.factor(behav$previsyes)
behav$prevconf = as.factor(behav$prevconf)
behav$correct = as.factor(behav$correct)
behav$prevcue = as.factor(behav$prevcue)

wd_path <- file.path("data", "behav", "mixed_models", "brms")
setwd(wd_path)

##################################### prev choice  models ##########################################

cue_model = brm(sayyes ~ isyes+cue+isyes*cue + (isyes+cue+isyes*cue|ID), data=behav, 
                family=bernoulli(link='probit'), 
                cores = getOption("mc.cores", 12))

summary(cue_model)

# posterior predictive checks
pp_check(cue_model)

# extract variables
get_variables(cue_model)

# Compute marginal effects for all parameters
marginal_effects_model <- marginal_effects(cue_model)

# Plot parameter estimates
me = plot(marginal_effects_model)
me_int = me$`isyes:cue`

# save figure
fig_save_path = file.path("figs", "manuscript_figures", "Figure1", "sdt_int_cue_brms_model.svg")
ggsave(fig_save_path, plot = me_int, device = "svg")

# save and read RDS model
saveRDS(object = cue_model, file = "cue_model_brms_firstsixtrialsonly.rds")
cue_model = readRDS("cue_model_brms.rds")

####################add previous choice############################################################

cue_prev_model = brm(sayyes ~ prevsayyes+isyes+cue+isyes*cue + (prevsayyes+isyes+cue+isyes*cue|ID),
                     data=behav, family=bernoulli(link='probit'), 
                     cores = getOption("mc.cores", 12))

summary(cue_prev_model)

# posterior predictive
pp_check(cue_prev_model)

# save and read model
saveRDS(object = cue_prev_model, file = "cue_prev_model_brms_firstsixtrialsonly.rds")
cue_prev_model = readRDS(file = "cue_prev_model_brms.rds")


############################interaction model######################################################

cue_prev_int_model = brm(sayyes ~ prevsayyes+isyes+beta+isyes*beta+beta*prevsayyes +
                           (prevsayyes+isyes+beta+isyes*beta+beta*prevsayyes|ID),
                     data=behav, family=bernoulli(link='probit'), 
                     cores = getOption("mc.cores", 12))

summary(cue_prev_int_model)

saveRDS(object = cue_prev_int_model_cue150to100, file = "cue_prev_int_model_cue150to0_brms.rds")
cue_prev_int_model = readRDS(file = "cue_prev_int_model_brms.rds")

########################### including prestimulus  power############################################

# replacing the cue predictor with prestimulus beta power
cue_prev_int_model_cue150to100 = brm(sayyes ~ isyes + beta_150to0 + prevsayyes + 
                                       beta_150to0*prevsayyes
                                       + beta_150to0*isyes +
                                         (isyes + beta_150to0 + prevsayyes + 
                                            beta_150to0*prevsayyes
                                          + beta_150to0*isyes|ID), data=behav, 
                                          family=bernoulli(link='probit'), 
                                          cores = getOption("mc.cores", 12)
)

# replacing previous choice with prestimulus beta power
cue_prev_int_model_prev300to100 = brm(sayyes ~ isyes + cue + beta_300to100 + cue*beta_300to100
                                       + cue*isyes +
                                         (isyes + cue + beta_300to100 + cue*beta_300to100
                                          + cue*isyes|ID), data=behav, 
                                          family=bernoulli(link='probit'), 
                                          cores = getOption("mc.cores", 12)
)

saveRDS(object = cue_prev_int_model_prev300to100, file = "cue_prev_int_model_prev300to100_brms.rds")
cue_prev_int_model_prev300to100 = readRDS(file = "cue_prev_int_model_prev300to100.rds")

# Obtain fixed effects estimates
fixed_effects <- fixef(cue_prev_int_model)

# Create a new data frame with the predictor variables and their corresponding values
new_data <- data.frame(cue = behav$cue,
                       prevsayyes = behav$prevsayyes)

# Predict the response variable using the fixed effects estimates and new data
predicted_values <- predict(cue_prev_int_model, newdata = new_data, type = "response")

# extract variables
get_variables(cue_prev_int_model)

# Compute marginal effects for all parameters
marginal_effects_model <- marginal_effects(cue_prev_int_model)

# Plot parameter estimates
me = plot(marginal_effects_model)
me_int = me$`prevsayyes:cue`

# save figure
ggsave(fig_save_path, plot = me_int, device = "svg")

# model comparision

# R2
bayes_R2(cue_model)
bayes_R2(cue_prev_model)
bayes_R2(cue_prev_int_model)

# looo
waic_comp = loo(cue_model, cue_prev_model, cue_prev_int_model, 'waic')

# save supplementary table 1

# save model output as table to html file
table1 = sjPlot::tab_model(cue_model, cue_prev_model, cue_prev_int_model)
output_file_path <- file.path("figs", "manuscript_figures", "Tables", "table1_supplm.html")
htmlTable(table1, file =  output_file_path)
#########################################signal vs. noise##########################################
# Signal vs. noise 

signal = filter(behav, isyes==1)
noise = filter(behav, isyes==0)

cue_prev_int_model_signal = brm(sayyes ~ prevsayyes+cue+cue*prevsayyes +
                           (prevsayyes+cue+cue*prevsayyes|ID),
                         data=signal, family=bernoulli(link='probit'), 
                         cores = getOption("mc.cores", 12))

summary(cue_prev_int_model_signal)

saveRDS(object = cue_prev_int_model_signal, file = "cue_prev_int_model_signal_brms.rds")

cue_prev_int_model_noise = brm(sayyes ~ prevsayyes+cue+cue*prevsayyes +
                                  (prevsayyes+cue+cue*prevsayyes|ID),
                                data=noise, family=bernoulli(link='probit'), 
                               cores = getOption("mc.cores", 12))

summary(cue_prev_int_model_noise)

saveRDS(object = cue_prev_int_model_noise, file = "cue_prev_int_model_noise_brms.rds")

############################# Confident vs. unconfident trials ####################################

conf = filter(behav, prevconf==1)
unconf = filter(behav, prevconf==0)

cue_prev_int_model_conf = brm(sayyes ~ isyes + prevsayyes + cue + isyes*cue+prevsayyes*cue +
                                  (isyes + prevsayyes + cue + isyes*cue+prevsayyes*cue|ID),
                                data=conf, family=bernoulli(link='probit'), 
                              cores = getOption("mc.cores", 12))

summary(cue_prev_int_model_conf)

saveRDS(object = cue_prev_int_model_conf, file = "cue_prev_int_model_conf_brms.rds")
cue_prev_int_model_conf = readRDS(file = "cue_prev_int_model_conf_brms.rds")

cue_prev_int_model_unconf = brm(sayyes ~ isyes + prevsayyes + cue + isyes*cue+prevsayyes*cue +
                                (isyes + prevsayyes + cue + isyes*cue+prevsayyes*cue|ID),
                              data=unconf, family=bernoulli(link='probit'), 
                              cores = getOption("mc.cores", 12))

summary(cue_prev_int_model_unconf)

saveRDS(object = cue_prev_int_model_unconf, file = "cue_prev_int_model_unconf_brms.rds")
cue_prev_int_model_unconf = readRDS(file = "cue_prev_int_model_unconf_brms.rds")

######################### Brain behavior ###########################################################


# replace stimulus probability regressor with beta or alpha power per trial (neural correlate of stimulus probability)

summary(brm(alpha_close_stimonset ~ cue + (cue|ID), data=behav))

summary(brm(beta_close_stimonset ~ cue + (cue|ID), data=behav, cores = getOption("mc.cores", 12)))

# expecon 2

summary(lmer(alpha_close_stimonset ~ cue + (cue|ID), data=behav, control=lmerControl(optimizer="bobyqa",
                                                                                     optCtrl=list(maxfun=2e5))))

summary(lmer(beta_close_stimonset ~ cue + (cue|ID), data=behav, control=lmerControl(optimizer="bobyqa",
                                                                                    optCtrl=list(maxfun=2e5))))

# expecon 1 model fitting
alpha_int_glm_expecon1 <- brm(sayyes ~ alpha_close_stimonset + isyes + prevresp + alpha_close_stimonset*isyes + 
                                  alpha_close_stimonset*prevresp + 
                                (alpha_close_stimonset + isyes + prevresp 
                                + alpha_close_stimonset*isyes + 
                                alpha_close_stimonset*prevresp| ID),
                                data = behav, family=bernoulli(link='probit'), 
                                cores = getOption("mc.cores", 12))

# save and load model from and to disk
brain_behav_path = file.path("data", "behav", "brain_behav","alpha_int_glm_expecon1_brms.rda")

saveRDS(alpha_int_glm_expecon1, brain_behav_path)

alpha_int_glm_expecon1 <- readRDS(brain_behav_path)

beta_int_glm_expecon1 <- brm(sayyes ~ beta_close_stimonset + isyes + prevresp + beta_close_stimonset*isyes + 
                                 beta_close_stimonset*prevresp + (beta_close_stimonset + isyes + prevresp + beta_close_stimonset*isyes + 
                                                                    beta_close_stimonset*prevresp | ID),
                               data = behav, family=bernoulli(link='probit'), 
                               cores = getOption("mc.cores", 12))

# save and load model from and to disk
brain_behav_path = file.path("data", "behav", "brain_behav","beta_int_glm_expecon1_brms.rda")

saveRDS(beta_int_glm_expecon1, brain_behav_path)

beta_int_glm_expecon1 <- readRDS(brain_behav_path)

# expecon 2 model fitting

alpha_int_glm_expecon2 <- brm(sayyes ~ alpha_close_stimonset + isyes + prevresp 
                              + alpha_close_stimonset*isyes + 
                                  alpha_close_stimonset*prevresp + 
                                (alpha_close_stimonset + isyes + 
                                   prevresp + alpha_close_stimonset*isyes + 
                                alpha_close_stimonset*prevresp| ID),
                                data = behav, family=bernoulli(link='probit'), 
                                cores = getOption("mc.cores", 12))

# save and load model from and to disk
brain_behav_path = file.path("data", "behav", "brain_behav","alpha_int_glm_expecon2.rda")

saveRDS(alpha_int_glm_expecon2, brain_behav_path)

alpha_int_glm_expecon2 <- readRDS(brain_behav_path)

beta_int_glm_expecon2 <- brm(sayyes ~ beta_close_stimonset + isyes + prevresp + 
                               beta_close_stimonset*isyes + 
                                 beta_close_stimonset*prevresp + 
                               (beta_close_stimonset + isyes + prevresp +
                                  beta_close_stimonset*isyes + 
                              beta_close_stimonset*prevresp | ID),
                               data = behav, family=bernoulli(link='probit'), 
                             cores = getOption("mc.cores", 12))

# save and load model from and to disk
brain_behav_path = file.path("data", "behav", "brain_behav","beta_int_glm_expecon2.rda")

saveRDS(beta_int_glm_expecon2, brain_behav_path)

beta_int_glm_expecon2 <- readRDS(brain_behav_path)

# plot brms outputs
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


int1 = plot_model(beta_int_glm_expecon1, type='int', mdrt.values = "meansd")
int2 = plot_model(beta_int_glm_expecon2, type='int', mdrt.values = "meansd")

####################################### mediation #################################################

# https://m-clark.github.io/models-by-example/bayesian-mixed-mediation.html

# is beta mediating the effect of the cue on the detection response? 

# Model 1: Predictor-Mediator Relationship
model1 <- brm(beta_close_stimonset ~ cue + (cue|ID), data = behav, family=bernoulli(link='probit'))

# Model 2: Mediator-Outcome Relationship
model2 <- brm(outcome ~ mediator + (1|random_group), data = your_data, family = gaussian())

# Model 3: Predictor-Outcome Relationship
model3 <- brm(outcome ~ predictor + (1|random_group), data = your_data, family = gaussian())
