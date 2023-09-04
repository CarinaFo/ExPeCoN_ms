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

####################################################################################################

# expecon 1
setwd("D:/expecon_ms")

# set working directory and load clean, behavioral dataframe
behav_path <- file.path("data", "behav", "behav_df", "prepro_behav_data.csv")

behav = read.csv(behav_path)

# expecon 2 behavioral data

setwd("D:/expecon_2/behav")

behav_path <- file.path("prepro_behav_data.csv")

behav = read.csv(behav_path)

# ID to exclude
ID_to_exclude <- 13

# Excluding the ID from the dataframe
behav <- behav[behav$ID != ID_to_exclude, ]

####################################brain behav#####################################################

brain_behav_path <- file.path("data", "behav", "behav_df", "brain_behav_cleanpower.csv")

# expecon 2

brain_behav_path <- file.path("behav", "brain_behav_cleanpower.csv")

behav = read.csv(brain_behav_path)

# Rename columns for expecon2
colnames(behav)[colnames(behav) == "stim_type"] <- "isyes"
colnames(behav)[colnames(behav) == "resp1"] <- "sayyes"
# Adding the previous response column to 'behav' data frame
behav <- behav %>%
  mutate(previous_response = lag(sayyes))
behav$previous_response = as.factor(behav$previous_response)

###############manual SDT calculation###############################################################

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


# Group by 'cue' and 'ID' and calculate the mean beta power for each group
result <- behav %>%
  group_by(cue, ID) %>%
  summarize(mean_beta_power = mean(beta_150to0, na.rm = TRUE))

# Pivot the data to have 'cue' as columns and 'beta' as values
df_wide <- pivot_wider(result, id_cols = ID, names_from = cue, values_from = mean_beta_power)

# Calculate the difference in beta power between the two conditions
df_diff <- df_wide %>%
  mutate(beta_difference = `0.25` - `0.75`)

# Filter the dataframe for each cue condition and calculate the mean criterion for each cue and participant
criterion_diff <- sdt_persub %>%
  filter(cue %in% c(0.75, 0.25)) %>%
  group_by(cue, ID) %>%
  summarize(mean_criterion = mean(crit))

# Calculate the difference between the criterion for the two cue conditions per participant
criterion_diff <- criterion_diff %>%
  pivot_wider(names_from = cue, values_from = mean_criterion) %>%
  mutate(criterion_difference = `0.25` - `0.75`)

cor.test(criterion_diff$criterion_difference, df_diff$beta_difference)

################################ linear mixed modelling ###########################################

# make factors:

behav$ID = as.factor(behav$ID)
behav$isyes = as.factor(behav$isyes)
behav$cue = as.factor(behav$cue)
behav$prevresp = as.factor(behav$prevresp)
behav$previsyes = as.factor(behav$previsyes)
behav$prevconf = as.factor(behav$prevconf)
behav$correct = as.factor(behav$correct)
behav$prevcue = as.factor(behav$prevcue)

# Remove NaN trials (first trial for each)
behav <- na.omit(behav) 

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

# does the cue condition predict beta power?

beta_cue = lmer(alpha_600to400 ~ cue + (cue|ID), data=behav) # yes
summary(beta_cue)
beta_prevchoice = lmer(alpha_900to700 ~ prevsayyes + (prevsayyes|ID), data=behav)
summary(beta_prevchoice)

# plot the difference
sjPlot::plot_model(beta_cue, type='pred')
# plot the difference
sjPlot::plot_model(beta_prevchoice, type='pred')


################################GLMMs##############################################################
# fit sdt model

cue_model2 = glmer(sayyes ~ isyes+cue+isyes*cue + (isyes+cue+isyes*cue|ID), data=behav, 
                  family=binomial(link='probit'),
                  control=glmerControl(optimizer="bobyqa",
                  optCtrl=list(maxfun=2e5)),
                  )

# check model performance
check_collinearity(cue_model)
check_convergence(cue_model)

summary(cue_model)

# Post hoc tests for interactions (estimated marginal means)
emm_model <- emmeans(cue_model, "isyes", by = "cue")
con = contrast(emm_model) # fdr corrected for 2 tests
con

# save and load model from and to disk
cue_model_path = file.path("data", "behav", "mixed_models", "cue_model.rda")
saveRDS(cue_model, cue_model_path)
cue_model <- readRDS(cue_model_path)

########################### add previous choice predictor###########################################

cue_prev_model = glmer(sayyes ~ isyes + cue + prevresp + isyes*cue +
                           + (isyes + cue + prevresp + isyes*cue|ID), data=behav, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_model)

check_collinearity(cue_prev_model)
check_convergence(cue_prev_model)

# save and load model from and to disk
cue_prev_model_path = file.path("data", "behav", "mixed_models", "cue_prev_model.rda")
saveRDS(cue_prev_model, cue_prev_model_path)
cue_prev_model2 <- readRDS(cue_prev_model_path)

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

########################### include confidence ###################################################

cue_prev_prevcue_model = glmer(sayyes ~ isyes + cue + prevresp + prevcue + isyes*cue +
                         + (isyes + cue + prevresp + prevcue + isyes*cue|ID), data=behav, 
                       family=binomial(link='probit'),
                       control=glmerControl(optimizer="bobyqa",
                                            optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_conf_model)
# save and load model from and to disk
cue_prev_conf_model_path = file.path("data", "behav", "mixed_models", "cue_prev_conf_model.rda")
saveRDS(cue_prev_conf_model, cue_prev_conf_model_path)
cue_prev_conf_model <- readRDS(cue_prev_conf_model_path)

################ including interaction between cue and previous choice#############################

cue_prev_int_model = glmer(sayyes ~ isyes + cue + prevresp + prevcue + prevresp*cue
                                        + cue*isyes +
                                        (isyes + cue + prevresp + prevcue + prevresp*cue
                                       + cue*isyes|ID), data=behav, 
                                      family=binomial(link='probit'),
                                      control=glmerControl(optimizer="bobyqa",
                                      optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model)

check_collinearity(cue_prev_int_model)
check_convergence(cue_prev_int_model)

# Post hoc tests for behavior interaction
emm_model <- emmeans(cue_prev_int_model, "prevresp", by = "cue")
con <- contrast(emm_model)
con

# Post hoc tests for power
emm_model <- emmeans(cue_prev_int_model_cue150to100, "beta_150to0", by = "prevsayyes", 
                     at = list(beta_150to0 = c(-1.16, 0.91)))
con <- contrast(emm_model)
con

emm_model <- emmeans(cue_prev_int_model_prev300to100, "beta_300to100", by = "cue", 
                     at = list(beta_300to100 = c(-1.16, 0.91)))
con <- contrast(emm_model)
con

# save and load model from and to disk
cue_prev_int_model_path = file.path("data", "behav", "mixed_models", "cue_prev_int_model.rda")

saveRDS(cue_prev_int_model, cue_prev_int_model_path)

cue_prev_int_model2 <- readRDS(cue_prev_int_model_path)

############################################### Model comparision##################################

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

output_file_path <- file.path("figs", "manuscript_figures", "Tables", "table1.html")
htmlTable(table1, file = output_file_path)

################################Plot estimates and interactions#####################################

theme_set(theme_sjplot())
save_path = file.path("figs", "manuscript_figures", "figure3_glmermodels")
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

# mean plus minus one sd for continious variables

intg = arrangeGrob(cue_signal_int1, cue_signal_int2, cue_signal_prev_int1,
             cue_signal_prev_int2, nrow=2)

ggsave('model_interactions.svg', dpi = 300, height = 8, width = 10, plot=intg)

cue_signal_int1 = plot_model(cue_model, type='int', mdrt.values = "meansd")
cue_signal_int2 = plot_model(cue_model2, type='int', mdrt.values = "meansd") 

cue_signal_prev_int1 = plot_model(cue_prev_int_model, type='pred',
                                  terms = c("prevresp", "cue"))
cue_signal_prev_int2 = plot_model(cue_prev_int_model2, type='pred', terms = c("prevresp", "cue"))

# change order of variables after model fitting
plot_model(cue_model, type = "pred", 
           terms = c("isyes", "beta_600to400 [-1.16, -0.12, 0.91]"))
###########################separate models for signal and noise trials#############################

signal = filter(behav, isyes==1)
noise = filter(behav, isyes==0)

cue_prev_int_model_signal = glmer(sayyes ~ prevresp + cue + prevresp*cue +
                             (prevresp + cue + prevresp*cue|ID), data=signal, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_signal)

check_collinearity(cue_prev_int_model_signal)
check_convergence(cue_prev_int_model_signal)

# save and load model from and to disk
cue_prev_int_model_signal_path = file.path("data", "behav", "mixed_models",
                                           "cue_prev_int_model_signal.rda")

saveRDS(cue_prev_int_model_signal, cue_prev_int_model_signal_path)

cue_prev_int_model_signal <- readRDS(cue_prev_int_model_signal_path)

# noise model

cue_prev_int_model_noise = glmer(sayyes ~ prevresp + cue + prevresp*cue +
                                    (prevresp + cue + prevresp*cue |ID), data=noise, 
                                  family=binomial(link='probit'),
                                  control=glmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_noise)

emmeans::emmeans(cue_prev_int_model_noise, ~ cue * prevsayyes)

check_collinearity(cue_prev_int_model_noise)
check_convergence(cue_prev_int_model_noise)

# save and load model from and to disk
cue_prev_int_model_noise_path = file.path("data", "behav", "mixed_models",
                                           "cue_prev_int_model_noise.rda")

saveRDS(cue_prev_int_model_noise, cue_prev_int_model_noise_path)

cue_prev_int_model_noise <- readRDS(cue_prev_int_model_noise_path)

##################separate model for confident/unconfident previous response#######

acc_conf_model = glmer(correct ~ conf + (conf|ID), data=behav, 
                       family=binomial(link='probit'),
                       control=glmerControl(optimizer="bobyqa",
                                            optCtrl=list(maxfun=2e5)))

summary(acc_conf_model)

conf = filter(behav, prevconf==1)
unconf = filter(behav, prevconf==0)

cue_prev_int_model_conf = glmer(sayyes ~ isyes + prevresp + cue + isyes*cue+prevresp*cue +
                             (isyes + prevresp + cue + isyes*cue+prevresp*cue|ID), data=conf, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_conf)
check_convergence(cue_prev_int_model_conf)

summary(cue_prev_int_model_conf)

# save and load model from and to disk
cue_prev_int_model_conf_path = file.path("data", "behav", "mixed_models",
                                         "cue_prev_int_model_conf.rda")

saveRDS(cue_prev_int_model_conf, cue_prev_int_model_conf_path)

cue_prev_int_model_conf <- readRDS(cue_prev_int_model_conf_path)

cue_prev_int_model_unconf = glmer(sayyes ~ isyes + prevresp + cue + isyes*cue+prevresp*cue +
                             (isyes + prevresp + cue + isyes*cue + prevresp*cue|ID), 
                             data=unconf, 
                             family=binomial(link='probit'),
                             control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_unconf)
check_convergence(cue_prev_int_model_unconf)

summary(cue_prev_int_model_unconf)

# save and load model from and to disk

cue_prev_int_model_unconf_path = file.path("data", "behav", "mixed_models",
                                         "cue_prev_int_model_unconf.rda")

saveRDS(cue_prev_int_model_unconf, cue_prev_int_model_unconf_path)

cue_prev_int_model_unconf <- readRDS(cue_prev_int_model_unconf_path)


###########################Brain behavior mediation ################################################################################################

# make sure lme4test is not loaded, this prevents the mediation model to work properly


fit.mediator <- glmer(beta_150to0 ~ cue + prevsayyes + isyes + cue*isyes + (cue + prevsayyes + isyes + cue*isyes | ID),
                        data = behav)


fit.dv <- glmer(sayyes ~ beta_150to0 + cue + prevsayyes + isyes + cue*isyes +  (beta_150to0 + cue + prevsayyes + isyes + cue*isyes| ID),
                         data = behav, family=binomial(link='probit'), 
                          control=glmerControl(optimizer="bobyqa",
                                      optCtrl=list(maxfun=2e5)))

results_prev <- mediate(fit.mediator, fit.dv, treat='cue', mediator='beta_150to0')

summary(results)

beta_cue_res = lmer(beta_150to0 ~ cue + (cue|ID), data=behav)
beta_prevchoice_res = lmer(beta_150to0 ~ prevsayyes + (prevsayyes|ID), data=behav)
beta_cue_prev = lmer(beta_150to0 ~ prevsayyes + cue + (prevsayyes+cue|ID), data=behav)

#
alpha_precue <- lmer(alpha_900to700 ~ prevsayyes + prevconf + previsyes + (prevsayyes+prevconf+previsyes|ID), data=behav)
alpha_postcue <- lmer(alpha_300to100 ~ prevsayyes + isyes + (prevsayyes+isyes|ID), data=behav)

beta_prestim <- lmer(beta_150to0 ~ cue + isyes + cue*isyes + (cue + isyes + cue*isyes|ID),
                     data=behav)

cue_prev_model_neur = glmer(sayyes ~ isyes + beta_150to0 + alpha_900to700 + isyes*beta_150to0 +
                         + (isyes + beta_150to0 + alpha_900to700 + isyes*beta_150to0|ID), data=behav, 
                       family=binomial(link='probit'),
                       control=glmerControl(optimizer="bobyqa",
                                            optCtrl=list(maxfun=2e5)),
)

est = plot_model(beta_prestim, type='est', 
                 title='prestimulus beta power ~',
                 sort.est = TRUE, show.values =TRUE, 
                 value.offset = 0.3, colors='Accent') +
  theme(plot.background = element_blank(),
        text = element_text(family = "Arial", size = 12)) +
  ylab("beta estimates")

est

# mean plus minus one sd for continious variables
plot_model(alpha_precue, type='pred', mdrt.values = "meansd") 

# change order of variables after model fitting
plot_model(cue_model, type = "pred", 
           terms = c("isyes", "beta_600to400 [-1.16, -0.12, 0.91]"))

save_path = file.path("figs", "manuscript_figures", "Figure5")
# Save the plot as an SVG file
ggsave(save_path, plot = int, device = "svg")
ggsave(save_path, plot = est, device = "svg")
