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
library(broom)

# don't forget to give credit to the amazing authors of those packages
#citation("emmeans")

# Set the font family and size
par(family = "Arial", cex = 1.2)

####################################################################################################
# set base directory
setwd("E:/expecon_ms")

# which dataset to analyze (1 => mini-block, 2 => trial-by-trial design)
expecon <- 1

if (expecon == 1) {
  
  # set working directory and load clean, behavioral dataframe
  behav_path <- file.path("data", "behav", "prepro_behav_data_expecon1.csv")
  
  behav_1 = read.csv(behav_path)
  
} else {
  
  behav_path <- file.path("data", "behav", "prepro_behav_data_expecon2.csv")
  
  behav_2 = read.csv(behav_path)
  
  # ID to exclude
  ID_to_exclude <- 13
  
  # Excluding the ID from the dataframe
  behav_2 <- behav[behav$ID != ID_to_exclude, ]
  
}

# to combine datasets, make sure they have the same amount of columns
behav_1 = subset(behav_1, select = -c(X,Unnamed..0,index,sayyes_y,surprise, sex, prevconf_resp,
                                      conf_resp))
behav_2 = subset(behav_2, select = -c(X,Unnamed..0.1,Unnamed..0, sayyes_y, gender, ITI))

# add sublock variable to study 2
subblock_list <- rep(1, nrow(behav_2))
behav_2$subblock <- subblock_list

# add variable for the 2 studies to the big dataframe
environment_list1 <- rep(1, nrow(behav_1))
environment_list2 <-rep(2, nrow(behav_2))

behav_1$study <- environment_list1
behav_2$study <- environment_list2

# change IDs for the second study (add 43 to each ID)
behav_2$ID <- behav_2$ID + 43

# combine both dataframes
behav = rbind(behav_1, behav_2)

behav_path = file.path("data", "behav", "prepro_behav_data_expecon1_2.csv")
write.csv(behav, behav_path)

###################load csv including data from both studies#######################################
# set base directory
setwd("E:/expecon_ms")

behav_path = file.path("data", "behav", "prepro_behav_data_expecon1_2.csv")
behav = read.csv(behav_path)

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

#recode accuracy
behav$correct = as.numeric(as.factor(behav$correct))

behav$correct[behav$correct == 1] <- 0
behav$correct[behav$correct == 2] <- 1

# Remove NaN trials for model comparision (models neeed to have same amount of data)
behav <- na.omit(behav) 
################################descriptive analysis###############################################

# Calculate accuracy for each cue condition and stimulus and each study and plot
mean_acc <- behav %>%
  group_by(ID, cue, isyes, study) %>%
  summarize(acc = mean(correct))

mean_acc$cue = as.factor(mean_acc$cue)
mean_acc$stimulus = as.factor(mean_acc$isyes)

ggplot(mean_acc, aes(x = cue, y = acc, fill = stimulus)) +
  geom_boxplot(position = position_dodge(0.8)) +
  facet_grid(~ study) +
  labs(x = "Cue Condition", y = "accuracy'")

# stimulus and response must be coded as 0 and 1 
calc_sdt_params <- function(response, stimulus) {
  # Calculate hit rate, false alarm rate, and overall mean
  hit_rate <- sum(response == 1 & stimulus == 1) / sum(stimulus == 1)
  false_alarm_rate <- sum(response == 1 & stimulus == 0) / sum(stimulus == 0)
  overall_mean <- (sum(response == 1) + sum(stimulus == 1)) / length(response)
  
  # Apply Hautus correction to prevent undefined values
  hit_rate = ifelse(hit_rate == 1, 1 - 1 / (2 * sum(stimulus == 1)), hit_rate)
  hit_rate = ifelse(hit_rate == 0, 1 / (2 * sum(stimulus == 1)), hit_rate)
  false_alarm_rate = ifelse(false_alarm_rate == 1, 1 - 1 / (2 * sum(stimulus == 0)), false_alarm_rate)
  false_alarm_rate = ifelse(false_alarm_rate == 0, 1 / (2 * sum(stimulus == 0)), false_alarm_rate)
  
  
  # Calculate Z-scores
  z_hit <- qnorm(hit_rate)
  z_false_alarm <- qnorm(false_alarm_rate)
  z_overall_mean <- qnorm(overall_mean)
  
  # Calculate d' and criterion
  d_prime <- z_hit - z_false_alarm
  criterion <- -0.5 * (z_hit + z_false_alarm)
  
  # Return the results as a list
  return(list(d_prime = d_prime, criterion = criterion))
}

# calc_sdt_params function is at the end of the script

# Calculate d' and criterion for each participant, previous response, and cue condition
results <- behav %>%
  group_by(ID, prevresp, cue, study) %>%
  summarize(d_prime = calc_sdt_params(sayyes, isyes)$d_prime,
            criterion = calc_sdt_params(sayyes, isyes)$criterion)

# convert to factor
results$prevresp = as.factor(results$prevresp)
results$cue = as.factor(results$cue)
results$study = as.factor(results$study)

# Plot boxplots for d' and criterion based on cue condition, previous response, and study
plot_d_prime <- ggplot(results, aes(x = cue, y = d_prime, fill = prevresp)) +
  geom_boxplot(position = position_dodge(0.8)) +
  facet_grid(~ study) +
  labs(title = "Boxplot of d' by Cue Condition, Previous Response, and Study",
       x = "Cue Condition", y = "d'") +
  scale_fill_manual(values = c(col_prevno, col_prevyes))

plot_criterion <- ggplot(results, aes(x = cue, y = criterion, fill = prevresp)) +
  geom_boxplot(position = position_dodge(0.8)) +
  facet_grid(~ study) +
  labs(title = "Boxplot of Criterion by Cue Condition, Previous Response, and Study",
       x = "Cue Condition", y = "Criterion") +
  scale_fill_manual(values = c(col_prevno, col_prevyes))

# Display the plots
plot_d_prime
plot_criterion

# plot mean over all participants
# plot mean reaction time for correct and error trials based on confidence level

# Calculate mean rts for each participant, accuracy, and confidence
mean_rts <- behav %>%
  group_by(ID, correct, conf, study) %>%
  summarize(mean_rts = mean(respt1))

# plot boxplots
rt_boxplot <- ggplot(mean_rts, aes(x = as.factor(correct), y = mean_rts, fill = as.factor(conf))) +
  geom_boxplot() +
  facet_wrap(~ study) +
  scale_fill_manual(values = c(col_correct, col_incorrect)) +
  labs(title = "Mean response time Based on accuracy and confidence", x = "Accuracy", 
       y = "Mean RTs", fill = "Confidence")

# Calculate mean confidence for each participant, accuracy, and stimulus
mean_confidence <- behav %>%
  group_by(ID, correct, isyes, study) %>%
  summarize(mean_rating = mean(conf))

# Plot boxplots
confidence_boxplot <- ggplot(mean_confidence, aes(x = as.factor(correct), y = mean_rating, 
                                                  fill = as.factor(isyes))) +
  geom_boxplot() +
  facet_wrap(~ study) +
  scale_fill_manual(values = c(col_correct, col_incorrect)) +
  labs(title = "Mean Confidence Based on Accuracy and Stimulus", x = "Accuracy", 
       y = "Mean Confidence", fill = "Stimulus")

confidence_boxplot
rt_boxplot

##################################GLMMers###########################################################

behav = filter(behav, study == expecon)

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
cue_model = glmer(sayyes ~ isyes+cue + isyes*cue + (isyes+cue|ID), data=behav, 
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
cue_model_path = file.path("data", "behav", "mixed_models", "behavior", filename)
saveRDS(cue_model, cue_model_path)
cue_model <- readRDS(cue_model_path)

########################### add previous choice predictor###########################################
cue_prev_model = glmer(sayyes ~ isyes + cue + prevresp + prevresp*isyes
                         + (cue+prevresp*isyes|ID), data=behav_1, 
                       family=binomial(link='probit'),
                       control=glmerControl(optimizer="bobyqa",
                                            optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_model)

check_collinearity(cue_prev_model)
check_convergence(cue_prev_model)

# Post hoc tests for behavior interaction
emm_model <- emmeans(cue_prev_model, "isyes", by = "prevresp", infer=TRUE)
con <- contrast(emm_model)
con


# save the model to disk
filename = paste("cue_prev_model_expecon", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", "behavior", filename)
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

cue_prev_int_model = glmer(sayyes ~ isyes + cue + prevresp + prevresp*cue + 
                           + cue*isyes +
                             (isyes + cue*prevresp|ID), data=behav, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model)
report(cue_prev_int_model)

check_collinearity(cue_prev_int_model)
check_convergence(cue_prev_int_model)

# save to disk
filename = paste("cue_prev_int_model", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", "behavior", filename)
saveRDS(cue_prev_int_model, cue_model_path)
cue_prev_int_model <- readRDS(cue_model_path)

# Post hoc tests for behavior interaction
emm_model <- emmeans(cue_prev_int_model, "prevresp", by = "cue", infer=TRUE)
con <- contrast(emm_model)
con

# Post hoc tests for interaction for both studies
emm_model <- emmeans(cue_prev_int_model, "cue", by = c("prevresp", "study"), infer=TRUE)
con <- contrast(emm_model)
con
###########################check interaction in study 2 if you condition on prev cue #############

# Create a column that indexes wether the current cue had the same cue in the trial before (==1)
behav <- behav %>%
  mutate(same_as_previous = ifelse(cue == lag(cue), 1, 0))

same_only = filter(behav, same_as_previous==1)
diff_only = filter(behav, same_as_previous==0)

cue_prev_int_model_same = glmer(sayyes ~ isyes + cue + prevresp + prevresp*cue
                           + cue*isyes +
                             (isyes + cue + prevresp|ID), data=same_only, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_same)

cue_prev_int_model_diff = glmer(sayyes ~ isyes + cue + prevresp + prevresp*cue
                                + cue*isyes +
                                  (isyes + cue + prevresp|ID), data=diff_only, 
                                family=binomial(link='probit'),
                                control=glmerControl(optimizer="bobyqa",
                                                     optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_diff)
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
# crop the file and save as png or pdf

filename = paste("expecon", expecon, ".html", sep="_")
output_file_path <- file.path("figs", "manuscript_figures", "Tables", filename)

sjPlot::tab_model(simple_sdt_model, cue_model, cue_prev_model, cue_prev_int_model, 
                           show.aic=TRUE, show.loglik=TRUE,
                           file=output_file_path)
###########################separate models for signal and noise trials#############################

signal = filter(behav, isyes==1)
noise = filter(behav, isyes==0)

cue_prev_int_model_signal = glmer(sayyes ~ prevresp + cue + prevresp*cue +
                                    (prevresp+cue|ID), data=signal, 
                                  family=binomial(link='probit'),
                                  control=glmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_signal)

check_collinearity(cue_prev_int_model_signal)
check_convergence(cue_prev_int_model_signal)

#only fitted for expecon 1
cue_model_path = file.path("data", "behav", "mixed_models", "behavior", "cue_prev_int_model_signal_1.rda")
saveRDS(cue_prev_int_model_signal, cue_model_path)
cue_prev_int_model_signal <- readRDS(cue_model_path)

# noise model
cue_prev_int_model_noise = glmer(sayyes ~ prevresp + cue + prevresp*cue +
                                   (prevresp*cue|ID), data=noise, 
                                 family=binomial(link='probit'),
                                 control=glmerControl(optimizer="bobyqa",
                                                      optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_noise)

check_collinearity(cue_prev_int_model_noise)
check_convergence(cue_prev_int_model_noise)

cue_model_path = file.path("data", "behav", "mixed_models", "behavior", "cue_prev_int_model_noise_1.rda")
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
                                  (isyes + prevresp+cue|ID), data=conf, 
                                family=binomial(link='probit'),
                                control=glmerControl(optimizer="bobyqa",
                                                     optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_conf)
check_convergence(cue_prev_int_model_conf)

summary(cue_prev_int_model_conf)

cue_model_path = file.path("data", "behav", "mixed_models", "behavior", "cue_prev_int_model_conf_1.rda")
saveRDS(cue_prev_int_model_conf, cue_model_path)
cue_prev_int_model_conf <- readRDS(cue_model_path)


cue_prev_int_model_unconf = glmer(sayyes ~ isyes + prevresp + cue + isyes*cue+prevresp*cue +
                                    (isyes + prevresp*cue|ID), 
                                  data=unconf, 
                                  family=binomial(link='probit'),
                                  control=glmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_unconf)
check_convergence(cue_prev_int_model_unconf)

summary(cue_prev_int_model_unconf)

# save and load model from and to disk
cue_model_path = file.path("data", "behav", "mixed_models", "behavior", "cue_prev_int_model_unconf_1.rda")
saveRDS(cue_prev_int_model_unconf, cue_model_path)
cue_prev_int_model_unconf <- readRDS(cue_model_path)

# save model summary as table
filename = paste("expecon_interaction_control_models", expecon, ".html", sep="_")
output_file_path <- file.path("figs", "manuscript_figures", "Tables", filename)

sjPlot::tab_model(cue_prev_int_model_signal, cue_prev_int_model_noise,
                           cue_prev_int_model_conf, cue_prev_int_model_unconf, 
                           show.aic=TRUE, show.loglik=TRUE, file = output_file_path)