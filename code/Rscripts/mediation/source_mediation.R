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
library(performance)
library(r2mlm) # for R2 (power estimates and model comparision for mixed models)

# don't forget to give credit to the amazing authors of those packages
#citation("emmeans")

# Set the font family and size
par(family = "Arial", cex = 1.2)

# skip scientific notation
options(scipen=999)

###################load csv including data from both studies#######################################
# Set base directory
setwd("E:/expecon_ms")

# Define expecon variable
expecon = 1
freq_band = 'alpha'

# Construct the file path using the expecon variable
behav_path = file.path("data", "behav", paste0("brain_behav_cleaned_source_", freq_band, "_", expecon, ".csv"))

# Read the CSV file
behav = read.csv(behav_path)

pow_variables = c(paste0(freq_band, '_source_prob'), paste0(freq_band, '_source_prev'))
################################ prep for modelling ###########################################

# make factors for categorical variables:
behav$ID = as.factor(behav$ID) # subject ID

# don't create factor variables for mediation model

behav$isyes = as.factor(behav$isyes) # stimulus
# dummy recode
behav$cue <- ifelse(behav$cue == 0.25, 0, 1)
behav$cue = as.factor(behav$cue) # probability for a signal
behav$prevresp = as.factor(behav$prevresp) # previous response
behav$previsyes = as.factor(behav$previsyes) # previous stimulus
behav$prevconf = as.factor(behav$prevconf) # previous confidence
behav$correct = as.factor(behav$correct) # performance
behav$prevcue = as.factor(behav$prevcue) # previous probability

#recode accuracy
behav$correct = as.numeric(as.factor(behav$correct))

behav$correct[behav$correct == 1] <- 0
behav$correct[behav$correct == 2] <- 1

# Remove NaN trials for model comparision (models neeed to have same amount of data)
behav <- na.omit(behav) 

# remove outlier trials based on pre-stim power
behav <- behav[!(behav[[pow_variables[1]]] > 3 | behav[[pow_variables[1]]] < -3), ]
behav <- behav[!(behav[[pow_variables[2]]] > 3 | behav[[pow_variables[2]]] < -3), ]

hist(behav[[pow_variables[1]]])
hist(behav[[pow_variables[2]]])
##################################GLMMers###########################################################

summary(glm(isyes ~ previsyes, data=behav, family=binomial(link='probit')))

null_model_beta = lmer(beta_source_prob ~ 1 + (1|ID), data=behav)

beta_by_cue_fixed = lmer(beta_source_prob ~ cue + (1|ID), data=behav)

beta_by_cue = lmer(beta_source_prob ~ cue + (cue|ID), data=behav, 
                   control=lmerControl(optimizer="bobyqa",
                    optCtrl=list(maxfun=2e5)))

beta_by_prevchoice = lmer(beta_source_prev ~ prevresp + (prevresp|ID), data=behav)

alpha_by_cue = lmer(alpha_source_prob ~ cue + (cue|ID), data=behav)
summary(alpha_by_cue)

alpha_by_prevchoice = lmer(alpha_source_prev ~ prevresp + (prevresp|ID), data=behav)

# get R2
r2mlm(null_model_beta)
r2mlm(beta_by_cue_fixed)
r2mlm(alpha_by_cue)
r2mlm(beta_by_cue)

# does the cue predict beta source prob?
sdt_glm_prevresp = glmer(sayyes ~ isyes*cue + prevresp*cue +
                                       (isyes+cue*prevresp|ID), data=behav, 
                                     family=binomial(link='probit'),
                                     control=glmerControl(optimizer="bobyqa",
                                                          optCtrl=list(maxfun=2e5)))

summary(sdt_glm_prevresp)

check_collinearity(sdt_glm_prevresp)
check_convergence(sdt_glm_prevresp)

# extract random effects for each parameter (only works if the parameter was fit as a random effect)
ranef = ranef(sdt_glm_prevresp)
choice_bias = ranef$ID$prevresp1
crit_change = ranef$ID$cue0.75
# singularity warning
prev_cue_int = ranef$ID$`cue0.75:prevresp1`
  
# Group by ID
df_grouped <- behav %>%
  group_by(ID) %>%
  # Calculate means for cue == 1 and cue == 0 separately
  summarise(mean_cue_1 = mean(beta_source_prob[cue == 0.75], na.rm = TRUE),
            mean_cue_0 = mean(beta_source_prob[cue == 0.25], na.rm = TRUE)) %>%
  # Calculate the difference between the means
  mutate(difference = mean_cue_0 - mean_cue_1)

# Save differences in a list
beta_pow_change_prob <- df_grouped$difference

# Group by ID
df_grouped <- behav %>%
  group_by(ID) %>%
  # Calculate means for cue == 1 and cue == 0 separately
  summarise(mean_cue_1 = mean(beta_source_prev[prevresp == 1], na.rm = TRUE),
            mean_cue_0 = mean(beta_source_prev[prevresp == 0], na.rm = TRUE)) %>%
  # Calculate the difference between the means
  mutate(difference = mean_cue_0 - mean_cue_1)

# calculate criterion change with SDT as sanity check

# Function to calculate hit rate and false alarm rate with Hautus correction
calc_rates_hautus <- function(df) {
  # Count the number of hits and false alarms
  n_stimulus = sum(df$isyes == 1)
  n_no_stimulus = sum(df$isyes == 0)
  
  hits = sum(df$isyes == 1 & df$sayyes == 1)
  false_alarms = sum(df$isyes == 0 & df$sayyes == 1)
  
  # Hit rate with Hautus correction
  hit_rate = (hits + 0.5) / (n_stimulus + 1)
  
  # False alarm rate with Hautus correction
  false_alarm_rate = (false_alarms + 0.5) / (n_no_stimulus + 1)
  
  return(data.frame(
    hit_rate = hit_rate,
    false_alarm_rate = false_alarm_rate
  ))
}

# Group by ID and cue and apply the calc_rates function
behav_grouped <- behav %>%
  group_by(ID, cue) %>%
  do(calc_rates_hautus(.))

# Calculate the SDT criterion
behav_grouped <- behav_grouped %>%
  mutate(criterion = -0.5 * (qnorm(hit_rate) + qnorm(false_alarm_rate)))

crit_diff = 
  behav_grouped$criterion[behav_grouped$cue == 0.25] - 
  behav_grouped$criterion[behav_grouped$cue == 0.75]

# Save differences in a list
beta_pow_change_prev <- df_grouped$difference

cor.test(beta_pow_change_prev, choice_bias)
cor.test(beta_pow_change_prob, crit_diff) # from manual calculation
cor.test(beta_pow_change_prob, crit_change) # model estimates
cor.test(beta_pow_change_prev, prev_cue_int)
cor.test(beta_pow_change_prob, prev_cue_int)

# control analysis:
cor.test(beta_pow_change_prob, choice_bias)
cor.test(beta_pow_change_prev, crit_change)

# Combine the lists into a data frame
df <- data.frame(beta_pow_change_prev, beta_pow_change_prob, choice_bias, crit_change, prev_cue_int)

# Rename the columns (optional)
colnames(df) <- c("beta_power_prev", "beta_power_prob", "choice_bias", "crit_change", "interaction")

# save csv
filename = paste("source_model_est_", expecon, ".csv", sep="")
save_path = file.path("data", "behav", filename)

# Save the data frame to a CSV file
write.csv(df, save_path, row.names = FALSE)
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
med.model_beta_prob <- lme4::lmer(beta_source_prob ~ cue + prevresp +
                                     (1 + prevresp|ID), 
                                   data = behav,
                                   control=lmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)))
summary(med.model_beta_prob)


med.model_beta_prev <- lme4::lmer(beta_source_prev ~  prevresp + cue +
                                    (1 + prevresp + cue|ID), 
                             data = behav,
                             control=lmerControl(optimizer="bobyqa",
                                                 optCtrl=list(maxfun=2e5))) # significant 
summary(med.model_beta_prev)

# fit outcome model: do the mediator (beta) and the IV (stimulus probability cue) predict the
# detection response? included stimulus and previous choice at a given trial as covariates,
# but no interaction between prev. resp and cue
out.model_beta_prob <- glmer(sayyes ~ beta_source_prob + cue + isyes + prevresp +
                          (1 + cue + isyes + prevresp|ID),
                        data = behav,
                        control=glmerControl(optimizer="bobyqa",
                                             optCtrl=list(maxfun=2e5)),
                        family=binomial(link='probit'))

summary(out.model_beta_prob)

out.model_beta_prev <- glmer(sayyes ~ beta_source_prev + prevresp + cue + isyes +
                          (1 + isyes+prevresp+cue|ID),
                        data = behav,
                        control=glmerControl(optimizer="bobyqa",
                                             optCtrl=list(maxfun=2e5)),
                        family=binomial(link='probit'))

summary(out.model_beta_prev)

# save models as tables for manuscript
#https://modelsummary.com/articles/modelsummary.html

filename_med = paste("mediation_expecon", expecon, ".docx", sep="_")
output_file_path_med<- file.path("figs", "manuscript_figures", "Tables", filename_med)

models = list("probability" = out.model_beta_prob, "previous_response" = out.model_beta_prev)

modelsummary::modelsummary(models, estimate  = "{estimate} [{conf.low}, {conf.high}], {stars}", 
                           statistic = NULL,  output = output_file_path_med)

mediation_cue_beta_prob <- mediate(med.model_beta_prob, out.model_beta_prob, treat='cue', 
                              mediator='beta_source_prob')

summary(mediation_cue_beta_prob)

mediation_cue_beta_prev <- mediate(med.model_beta_prev, out.model_beta_prev, treat='prevresp', 
                              mediator='beta_source_prev')

summary(mediation_cue_beta_prev)

############################confidence modelling ##################################################
# not included in manuscript

conf.model_beta <- lme4::lmer(beta_source_prob ~  isyes*sayyes + cue + prevresp + prevconf +
                                   (isyes + sayyes + prevresp + prevconf|ID), 
                                 data = behav,
                                 control=lmerControl(optimizer="bobyqa",
                                                     optCtrl=list(maxfun=2e5))) # significant 
summary(conf.model_beta)

conf_full_model_beta = glmer(conf ~ isyes*sayyes*beta_source_prob + cue + prevresp + prevconf +
                                 (isyes + sayyes + prevresp + prevconf|ID), data=behav, 
                               family=binomial(link='probit'),
                               control=glmerControl(optimizer="bobyqa",
                                                    optCtrl=list(maxfun=2e5)))

check_collinearity(conf_full_model_beta)
check_convergence(conf_full_model_beta)

summary(conf_full_model_beta)
plot_model(conf_full_model_beta)

mediation_cue_beta_conf <- mediate(conf_full_model_beta, conf.model_beta, treat='cue', 
                                   mediator='beta_source_prob')

summary(mediation_cue_beta_conf)