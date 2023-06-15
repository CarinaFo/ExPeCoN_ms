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
library(MuMIn)
library(finalfit)
library(htmlTable)

####################################################################################################

# set working directory and load behavioral dataframe

#behav = read.csv("D:\\expecon_ms\\data\\behav\\behav_df\\prepro_behav_data.csv")
behav = read.csv("D:\\expecon_ms\\figs\\brain_behavior\\brain_behav.csv")


# convert power values, log transform and standardize (see Stephani et. al, 2021)

behav$theta_scale_log <- scale(log10(behav$theta_pw), center=T, scale=T)
behav$alpha_scale_log <- scale(log10(behav$alpha_pw), center=T, scale=T)
behav$lowbeta_scale_log <- scale(log10(behav$low_beta_pw), center=T, scale=T)
behav$beta_gamma_scale_log <- scale(log10(behav$beta_gamma_pw), center=T, scale=T)

hist(behav$alpha_scale_log)

###############manual SDT calculation#####################################################

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
behav$previsyes = as.factor(behav$previsyes)
behav$prevconf = as.factor(behav$prevconf)

# create prev correct column
behav$prevacc = shift(behav$correct)

# test for autocorrelation between trials?

high_only = filter(behav, cue==0.75)
low_only = filter(behav, cue==0.25)

auto_model = glmer(isyes ~ previsyes*cue+(previsyes*cue|ID), data=behav, 
                  family=binomial(link='probit'),
)

summary(auto_model)

# fit sdt model

cue_model = glmer(sayyes ~ isyes*cue + (isyes*cue|ID), data=behav, 
                  family=binomial(link='probit'),
                  control=glmerControl(optimizer="bobyqa",
                  optCtrl=list(maxfun=2e5)),
                  )

saveRDS(cue_model, "D:\\expecon_ms\\analysis_code\\behav\\
        linear_mixed_models\\cue_model.rda")
cue_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_model.rda")

summary(cue_model)

cue_prev_model = glmer(sayyes ~ isyes*cue + prevsayyes
                           + (isyes*cue + prevsayyes|ID), data=behav, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

saveRDS(cue_prev_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_model.rda")
cue_prev_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_model.rda")

summary(cue_prev_model)

# Set the font family and size

par(family = "Arial", cex = 1.2)

est = sjPlot::plot_model(cue_model, type='est')
int = sjPlot::plot_model(cue_model, type='int')

# previous stimulus is not significant

# cue is still a significant predictor but less strong effect

# including interaction between cue and previous choice

cue_prev_int_model = glmer(sayyes ~ isyes*cue + prevsayyes*cue 
                           + (prevsayyes*cue + prevsayyes*cue|ID), data=behav, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                           optCtrl=list(maxfun=2e5)),
)

saveRDS(cue_prev_int_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_model.rda")
cue_prev_int_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_model.rda")

summary(cue_prev_int_model)

AIC(cue_model)
AIC(cue_prev_model)
AIC(cue_prev_int_model)

# save table to PDF
table1 = sjPlot::tab_model(cue_model, cue_prev_model, cue_prev_int_model, show.aic=TRUE, show.loglik=TRUE)

# Save the output as an HTML file
output_file <- "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\table1.html"
htmlTable(table1, file = output_file)

######################## separate model for signal and noise trials only###############

signal = filter(behav, isyes==1)
noise = filter(behav, isyes==0)

cue_prev_int_signal_model = glmer(sayyes ~ prevsayyes*cue 
                           + (prevsayyes*cue|ID), data=signal, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)


cue_prev_int_noise_model = glmer(sayyes ~ prevsayyes*cue 
                                  + (prevsayyes*cue|ID), data=noise, 
                                  family=binomial(link='probit'),
                                  control=glmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)),
)

saveRDS(cue_prev_int_signal_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_signal_model.rda")
saveRDS(cue_prev_int_noise_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_noise_model.rda")

cue_prev_int_signal_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_signal_model.rda")
cue_prev_int_noise_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_noise_model.rda")

summary(cue_prev_int_signal_model)
summary(cue_prev_int_noise_model)

est = sjPlot::plot_model(cue_prev_int_signal_model, type='est')
int = sjPlot::plot_model(cue_prev_int_signal_model, type='int')

est = sjPlot::plot_model(cue_prev_int_noise_model, type='est')
int = sjPlot::plot_model(cue_prev_int_noise_model, type='int')

##################separate model for confident/unconfident previous response#######

conf = filter(behav, prevconf==1)
unconf = filter(behav, prevconf==0)


cue_prev_int_conf_model = glmer(sayyes ~ isyes*cue + prevsayyes*cue+ 
                                  + (isyes*cue + prevsayyes*cue|ID), data=conf, 
                                  family=binomial(link='probit'),
                                  control=glmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)),
)


cue_prev_int_unconf_model = glmer(sayyes ~ isyes*cue + prevsayyes*cue 
                                 + (isyes*cue + prevsayyes*cue|ID), data=unconf, 
                                 family=binomial(link='probit'),
                                 control=glmerControl(optimizer="bobyqa",
                                                      optCtrl=list(maxfun=2e5)),
)


saveRDS(cue_prev_int_conf_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_conf_model.rda")
saveRDS(cue_prev_int_unconf_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_unconf_model.rda")

cue_prev_int_conf_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_conf_model.rda")
cue_prev_int_unconf_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_unconf_model.rda")

summary(cue_prev_int_conf_model)
summary(cue_prev_int_unconf_model)

est = sjPlot::plot_model(cue_prev_int_conf_model, type='est')
int = sjPlot::plot_model(cue_prev_int_conf_model, type='int')

est = sjPlot::plot_model(cue_prev_int_unconf_model, type='est')
int = sjPlot::plot_model(cue_prev_int_unconf_model, type='int')

######################################################################################


cue_prevacc_int_model = glmer(sayyes ~ isyes*cue+prevacc*cue+ (isyes*cue+prevacc*cue|ID), 
                                data=behav, family=binomial(link='probit'),
                                control=glmerControl(optimizer="bobyqa",
                                optCtrl=list(maxfun=2e5)))

saveRDS(cue_previsyes_int_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_previsyes_int_model.rda")
cue_previsyes_int_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_previsyes_int_model.rda")

summary(cue_previsyes_int_model)

AIC(cue_model)
AIC(cue_prev_model)
AIC(cue_prev_int_model)
AIC(cue_previsyes_int_model)

#################################Confidence#########################################################

# stronger prevchoice effect in confident trials

cue_prev_int_conf_model = glmer(sayyes ~ isyes*cue + prevsayyes*cue + (isyes*cue+prevsayyes*cue|ID), 
                                data=confident_trials_only, family=binomial(link='probit'),
                                control=glmerControl(optimizer="bobyqa",
                                optCtrl=list(maxfun=2e5)))

saveRDS(cue_prev_int_conf_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_conf_model.rda")
cue_prev_int_conf_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_conf_model.rda")

summary(cue_prev_int_conf_model)

AIC(cue_prev_int_model)
AIC(cue_prev_int_conf_model)

#######################################Prev confidence as regressor#################################

# include confidence in previous trial as regressor

cue_prev_int_confreg_model = glmer(sayyes ~ isyes*cue + prevsayyes*cue + prevconf + 
                              (isyes*cue+prevsayyes*cue+prevconf|ID), 
                              data=behav, family=binomial(link='probit'),
                              control=glmerControl(optimizer="bobyqa",
                              optCtrl=list(maxfun=2e5)))

saveRDS(cue_prev_int_confreg_model, "D:\\expecon_ms\\analysis_code\\
        behav\\linear_mixed_models\\cue_prev_int_confreg_model.rda")

cue_prev_int_confreg_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_confreg_model.rda")

summary(cue_prev_int_confreg_model)

# prev conf is not a sign. regressor

cue_prevconf_int = glmer(sayyes ~ isyes*cue + prevconf*cue + (isyes*cue+prevconf*cue|ID), 
                              data=behav, family=binomial(link='probit'),
                              control=glmerControl(optimizer="bobyqa",
                              optCtrl=list(maxfun=2e5)))

saveRDS(cue_prevconf_int, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prevconf_int.rda")
cue_prevconf_int <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prevconf_int.rda")

summary(cue_prevconf_int)

cue_prevacc_int_model = glmer(sayyes ~ isyes*cue+prevacc*cue+ (isyes*cue+prevacc*cue|ID), 
                              data=behav, family=binomial(link='probit'),
                              control=glmerControl(optimizer="bobyqa",
                                                   optCtrl=list(maxfun=2e5)))

saveRDS(cue_prevacc_int_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prevacc_int_model.rda")
cue_prevacc_int_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prevacc_int_model.rda")

summary(cue_prevacc_int_model)

# cue is still significant, interaction between cue and previous confidence rating not

###########################################Metacognition############################################

# does the detection response predict confidence?

resp_conf = glmer(conf ~ sayyes + (sayyes|ID), data=behav, family=binomial(link='probit'))

saveRDS(resp_conf, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\resp_conf.rda")

summary(resp_conf)

# higher confidence in no responses

# does accuracy predict confidence?

conf_acc = glmer(conf ~ correct + (correct|ID), data=behav, family=binomial(link='probit'))

saveRDS(conf_acc, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\conf_acc.rda")

summary(conf_acc)

# higher confidence in correct trials

conf_con = glmer(conf ~ congruency + (congruency|ID), data=behav, family=binomial(link='probit'))

saveRDS(conf_con, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\conf_con.rda")

summary(conf_con)

#higher confidence in congruent trials

conf_surprise = glmer(conf ~ surprise + (surprise|ID), data=behav, 
                      family=binomial(link='probit'))

saveRDS(conf_surprise, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\conf_surprise.rda")

summary(conf_surprise)

# higher surprise, less confident

#################################Plot model parameters##############################################

# Set the font family and size

par(family = "Arial", cex = 1.2)

est = sjPlot::plot_model(cue_model, type='est')
int = sjPlot::plot_model(cue_model, type='int')

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


##############################brain behavior modelling##################################

cue_theta = lmer(theta_scale_log ~ cue + (cue|ID), data=behav) # n.s.

cue_alpha = lmer(alpha_scale_log ~ cue + (cue|ID), data=behav, control=lmerControl(optimizer="bobyqa",
                                                                                    optCtrl=list(maxfun=2e5)))
# n.s.

cue_beta = lmer(lowbeta_scale_log ~ cue + (cue|ID), data=behav) # p = 0.02
cue_beta_gamma = lmer(beta_gamma_scale_log ~ cue + (cue|ID), data=behav) # n.s.

# previous choice predicts alpha, beta and beta_gamma power

prevchoice_beta = lmer(lowbeta_scale_log ~ prevsayyes + (prevsayyes|ID), data=behav)
prevchoice_alpha = lmer(alpha_scale_log ~ prevsayyes + (prevsayyes|ID), data=behav)
prevchoice_theta = lmer(theta_scale_log ~ prevsayyes + (prevsayyes|ID), data=behav)
prevchoice_beta_gamma = lmer(beta_gamma_scale_log ~ prevsayyes + (prevsayyes|ID), data=behav)

# SDT model with beta power as covariate

sdt_model_beta = glmer(sayyes ~ isyes*lowbeta_scale_log + prevsayyes*lowbeta_scale_log + 
                       (isyes*lowbeta_scale_log + prevsayyes*lowbeta_scale_log|ID), data=behav, 
                        family=binomial(link='probit'),
                        control=glmerControl(optimizer="bobyqa",
                        optCtrl=list(maxfun=2e5)))

sdt_model_beta_gamma = glmer(sayyes ~ isyes*beta_gamma_scale_log + prevsayyes*beta_gamma_scale_log + 
                            (isyes*beta_gamma_scale_log + prevsayyes*beta_gamma_scale_log|ID), data=behav, 
                             family=binomial(link='probit'),
                             control=glmerControl(optimizer="bobyqa",
                             optCtrl=list(maxfun=2e5)))



