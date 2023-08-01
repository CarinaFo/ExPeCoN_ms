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

####################################################################################################

# set working directory and load clean, behavioral dataframe
behav_path <- file.path("data", "behav", "behav_df", "prepro_behav_data.csv")

behav = read.csv(behav_path)

# brain behav path
brain_behav_path <- file.path("data", "behav", "behav_df", "behav_brain_cleanpower.csv")

behav = read.csv(brain_behav_path)

####################################################################################################

# make factors:

behav$ID = as.factor(behav$ID)
behav$isyes = as.factor(behav$isyes)
behav$cue = as.factor(behav$cue)
behav$prevsayyes = as.factor(behav$prevsayyes)
behav$previsyes = as.factor(behav$previsyes)
behav$prevconf = as.factor(behav$prevconf)
behav$correct = as.factor(behav$correct)

# Remove the first row (assures equal amount of data for all models)
behav <- behav[-1, ]

setwd("D:\\expecon_ms\\data\\behav\\mixed_models\\brms\\")

##################################### prev choice  models ##########################################

cue_model = brm(sayyes ~ isyes+cue+isyes*cue + (isyes+cue+isyes*cue|ID), data=behav, 
                family=bernoulli(link='probit'), 
                cores = getOption("mc.cores", 12))

# posterior predictive
pp_check(cue_model)

# extract variables
get_variables(cue_model)

# Compute marginal effects for all parameters
marginal_effects_model <- marginal_effects(cue_model)

# Plot parameter estimates
me = plot(marginal_effects_model)
me_int = me$`isyes:cue`

ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\sdt_int_cue_brms_model.svg", 
       plot = me_int, device = "svg")

# save and read RDS model
saveRDS(object = cue_model, file = "cue_model_brms_firstsixtrialsonly.rds")
summary(cue_model)

cue_model = readRDS("cue_model_brms.rds")

cue_prev_model = brm(sayyes ~ prevsayyes+isyes+cue+isyes*cue + (prevsayyes+isyes+cue+isyes*cue|ID),
                     data=behav, family=bernoulli(link='probit'), 
                     cores = getOption("mc.cores", 12))

# posterior predictive
pp_check(cue_prev_model)

saveRDS(object = cue_prev_model, file = "cue_prev_model_brms_firstsixtrialsonly.rds")

cue_prev_model = readRDS(file = "cue_prev_model_brms.rds")

summary(cue_prev_model)

cue_prev_int_model = brm(sayyes ~ prevsayyes+isyes+beta+isyes*beta+beta*prevsayyes +
                           (prevsayyes+isyes+beta+isyes*beta+beta*prevsayyes|ID),
                     data=behav, family=bernoulli(link='probit'), 
                     cores = getOption("mc.cores", 12))

saveRDS(object = cue_prev_int_model, file = "cue_prev_int_model_brms_beta.rds")

cue_prev_int_model = readRDS(file = "cue_prev_int_model_brms.rds")

# Obtain fixed effects estimates
fixed_effects <- fixef(cue_prev_int_model)

# Create a new data frame with the predictor variables and their corresponding values
new_data <- data.frame(cue = behav$cue,
                       prevsayyes = behav$prevsayyes)

# Predict the response variable using the fixed effects estimates and new data
predicted_values <- predict(cue_prev_int_model, newdata = new_data, type = "response")

# Create new variables representing the mediated pathway
mediator <- your_data$mediator_variable
outcome <- your_data$response_var

# Fit the mediation models
med_model <- lm(mediator ~ predictor1 + predictor2, data = your_data)
out_model <- lm(outcome ~ mediator + predictor1 + predictor2, data = your_data)

# Run the mediation analysis
mediation_result <- mediate(med_model, out_model, treat = "mediator", mediator = "outcome")

# extract variables
get_variables(cue_prev_int_model)

# Compute marginal effects for all parameters
marginal_effects_model <- marginal_effects(cue_prev_int_model)

# Plot parameter estimates
me = plot(marginal_effects_model)
me_int = me$`prevsayyes:cue`

ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\int_cue_brms_model.svg", 
       plot = me_int, device = "svg")

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
output_file <- "D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\table1_supplm.html"
htmlTable(table1, file = output_file)

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
