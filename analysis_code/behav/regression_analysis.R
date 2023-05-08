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
library(sjPlot)
library(svglite)
library(ggplot2)

####################################################################################################

# set working directory and load behavioral dataframe

behav = read.csv("D:\\expecon_ms\\data\\behav\\behav_df\\prepro_behav_data.csv")

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


sdt <- behav %>%
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
behav$prevsayyes = as.factor(behav$prevsayyes)
behav$prevconf = as.factor(behav$prevconf)

# fit sdt model

cue_model = glmer(sayyes ~ isyes*cue + (isyes*cue|ID), data=behav, 
                  family=binomial(link='probit'),
                  control=glmerControl(optimizer="bobyqa",
                                       optCtrl=list(maxfun=2e5)))

save(cue_model, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\sdt_cue.rda")

# cue is still a significant predictor but less strong effect

# including interaction between cue and previous choice

# do we see the same effect in confident responses and unconfident responses?

confident_trials_only = filter(behav, conf==1)
unconfident_trials_only = filter(behav, conf==0)

# yes, significant interaction in both confident and unconfident trials but stronger in confident 
# responses (however less unconfident trials)

cue_lag_int_model = glmer(sayyes ~ isyes*cue + prevsayyes*cue + (isyes*cue+prevsayyes*cue|ID), 
                      data=unconfident_trials_only, family=binomial(link='probit'),
                      control=glmerControl(optimizer="bobyqa",
                                           optCtrl=list(maxfun=2e5)))

# still significant interaction, prevconf is not significantly predicting det response

save(cue_lag_int_model, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\sdt_cue_prevchoice_int.rda")


# include confidence in previous trial as regressor

cue_lag_int_conf_model = glmer(sayyes ~ isyes*cue + prevsayyes*cue + prevconf + (isyes*cue+prevsayyes*cue
                                                                            +prevconf|ID), 
                          data=behav, family=binomial(link='probit'),
                          control=glmerControl(optimizer="bobyqa",
                                               optCtrl=list(maxfun=2e5)))

# still significant interaction, prevconf is not significantly predicting det response

save(cue_lag_int_conf_model, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_lag_int_conf_model.rda")

cue_prevconf = glmer(sayyes ~ isyes*cue + prevconf*cue + (isyes*cue+prevconf*cue|ID), 
                              data=behav, family=binomial(link='probit'),
                              control=glmerControl(optimizer="bobyqa",
                                                   optCtrl=list(maxfun=2e5)))

# cue is still significant, interaction between cue and previous confidence rating not

save(cue_prevconf, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prevconf.rda")

###########################################Metacognition############################################

resp_conf = glmer(conf ~ sayyes + (sayyes|ID), data=behav, family=binomial(link='probit'))

save(resp_conf, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\resp_conf.rda")

# higher confidence in no responses

conf_acc = glmer(conf ~ correct + (correct|ID), data=behav, family=binomial(link='probit'))

# higher confidence in correct trials

save(conf_acc, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\conf_acc.rda")

conf_con = glmer(conf ~ congruency + (congruency|ID), data=behav, family=binomial(link='probit'))

#higher confidence in congruent trials

conf_surprise = glmer(conf ~ congruency + (congruency|ID), data=behav, family=binomial(link='probit'))
# higher surprise, less confident

# plot the model parameters
# Set the font family and size

par(family = "Arial", cex = 1.2)

est = sjPlot::plot_model(cue_lag_int_model, type='est')
int = sjPlot::plot_model(cue_lag_int_model, type='int')

# Extract coefficients
coeffs <- as.data.frame(summary(cue_lag_int_model)$coefficients)

# Plot coefficients
p1 <- ggplot(coeffs, aes(x = rownames(coeffs), y = Estimate)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 3.5, color = "#2E8B57") +
  geom_errorbar(aes(ymin = Estimate - 1.96 * sdt, ymax = Estimate + 1.96 * sdt),  width = 0.3, size=1, color = "#2E8B57") +
  coord_flip() +
  labs(x = "", y = "Effect Size") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.y = element_text(size = 12),
        axis.line.x = element_blank(),
        panel.border = element_blank())
p1

# Create an effects object for the interaction
eff <- effect("cue0.75:lagsayyes1", cue_lag_int_model)

# Plot the predicted probabilities
plot(eff, type="response", rug=FALSE)

ggsave('D:\\expecon_ms\\figs\\behavior\\regression\\prev_sdt_estimates.svg', p1, device='svg', width=10, height=10)
ggsave('D:\\expecon_ms\\figs\\behavior\\regression\\prev_sdt_int.svg', int[[1]], device='svg')
ggsave('D:\\expecon_ms\\figs\\behavior\\regression\\prev_sdt_int_2.svg', int[[2]], device='svg')
