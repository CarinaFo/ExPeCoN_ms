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
library(lmerTest) # no p values without this package for linear mixed mdoels
library(dplyr)# pandas style
library(tidyr)
library(data.table) # for shift function
library(sjPlot)
library(ggplot2)
library(htmlTable)
library(emmeans)
library(gridExtra) # for subplots

# Set the font family and size

par(family = "Arial", cex = 1.2)

####################################################################################################

# set working directory and load clean, behavioral dataframe
behav_path <- file.path("data", "behav", "behav_df", "prepro_behav_data.csv")

behav = read.csv(behav_path)

# brain behav path
# cluster before stimulation cue onset from contrast previous choice
brain_behav_path <- file.path("data", "behav", "behav_df", "brain_behav_cleanpower_precue.csv")

# cluster close to stimulation onset from contrast high - low stim. probability
brain_behav_path <- file.path("data", "behav", "behav_df", "brain_behav_cleanpower.csv")

behav = read.csv(brain_behav_path)

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


# Group by 'cue' and 'ID' and calculate the mean beta power for each group
result <- behav %>%
  group_by(cue, ID) %>%
  summarize(mean_beta_power = mean(beta, na.rm = TRUE))

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
behav$prevsayyes = as.factor(behav$prevsayyes)
behav$previsyes = as.factor(behav$previsyes)
behav$prevconf = as.factor(behav$prevconf)
behav$correct = as.factor(behav$correct)

# Remove the first row (assures equal amount of data for all models)
behav <- behav[-1, ]

# test for autocorrelation between trials?
high = filter(behav, cue==0.75)
low = filter(behav, cue==0.25)

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

beta_cue = lmer(beta ~ cue + (cue|ID), data=behav) # yes
beta_prevchoice = lmer(beta ~ prevsayyes + (prevsayyes|ID), data=behav)

# plot the difference
sjPlot::plot_model(beta_cue, type='pred')
# plot the difference
sjPlot::plot_model(beta_prevchoice, type='pred')

# fit sdt model

cue_model = glmer(sayyes ~ isyes+beta+isyes*beta + (isyes+beta+isyes*beta|ID), data=behav, 
                  family=binomial(link='probit'),
                  control=glmerControl(optimizer="bobyqa",
                  optCtrl=list(maxfun=2e5)),
                  )

emmeans::emmeans(cue_model, 'cue')

check_collinearity(cue_model)
check_convergence(cue_model)

saveRDS(cue_model, "D:\\expecon_ms\\data\\behav\\mixed_models\\cue_model_betapow.rda")

cue_model <- readRDS("D:\\expecon_ms\\data\\behav\\mixed_models\\cue_model.rda")

summary(cue_model)

est = sjPlot::plot_model(cue_model, type='est', title='detection response ~',
                          axis.lim=c(0.8,8)) +
  theme(plot.background = element_blank(),
        text = element_text(family = "Arial", size = 14)) +
        ylab("Odds Ratio")

est

int = sjPlot::plot_model(cue_model, type='int', title='Predicted probability of responding yes')+
      theme(plot.background = element_blank(),
            text = element_text(family = "Arial", size = 14)) +
      ylab('detection response') +
      xlab('stimulus')
int

# Save the plot as an SVG file
ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\est_cue_model.svg", plot = est, device = "svg")
ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\int_cue_model.svg", plot = int, device = "svg")

cue_prev_model = glmer(sayyes ~ isyes + cue + prevsayyes + isyes*cue +
                           + (isyes + cue + prevsayyes + isyes*cue|ID), data=behav, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_model)
check_convergence(cue_prev_model)

saveRDS(cue_prev_model, "D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_model.rda")
cue_prev_model <- readRDS("D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_model.rda")

summary(cue_prev_model)

# Plot model

est = sjPlot::plot_model(cue_prev_model, type='est', title='detection response ~',
                         axis.lim=c(0.8,8)) +
  theme(plot.background = element_blank(),
        text = element_text(family = "Arial", size = 14)) +
  ylab("Odds Ratio")

est

# Save the plot as an SVG file
ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\cue_prev_model_est.svg", plot = est, device = "svg")

# cue is still a significant predictor but less strong effect

# including interaction between cue and previous choice

cue_prev_int_model = glmer(sayyes ~ isyes + alpha + prevsayyes + alpha*prevsayyes + alpha*isyes +
                            (isyes + alpha + prevsayyes + alpha*prevsayyes + alpha*isyes|ID), data=behav, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                           optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model)

emmeans::emmeans(cue_prev_int_model, pairwise ~ prevsayyes*beta)

# Print the post hoc comparisons with adjusted p-values
print(interaction_comparisons, adjust = "holm")
check_collinearity(cue_prev_int_model)
check_convergence(cue_prev_int_model)

saveRDS(cue_prev_int_model, "D:\\expecon_ms\\data\\behav\\mixed_models\\alpha_prev_int_model.rda")

cue_prev_int_model <- readRDS("D:\\expecon_ms\\data\\behav\\mixed_models\\beta_prev_int_model.rda")

est = sjPlot::plot_model(cue_prev_int_model, type='est', title='detection response ~',
                         sort.est = TRUE, transform='plogis', show.values =TRUE, 
                         value.offset = 0.3, colors='Accent') +
  theme(plot.background = element_blank(),
        text = element_text(family = "Arial", size = 12)) +
  ylab("Probabilities")

est

int = sjPlot::plot_model(cue_prev_int_model, type='int', mdrt.values = "meansd")

int = int[[2]]+
  theme(plot.background = element_blank(),
        text = element_text(family = "Arial", size = 12)) +
  ylab('detection response') +
  xlab('stimulus')

int

# Save the plot as an SVG file
ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure5\\beta_prev_int_int_signal_model.svg", plot = int, device = "svg")
ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure5\\beta_prev_int_est_model.svg", plot = est, device = "svg")

# Model comparision

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
output_file <- "D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\table1.html"
htmlTable(table1, file = output_file)

###########################separate models for signal and noise trials#############################

signal = filter(behav, isyes==1)
noise = filter(behav, isyes==0)

cue_prev_int_model_signal = glmer(sayyes ~ prevsayyes + beta + prevsayyes*beta +
                             (prevsayyes + beta + prevsayyes*beta|ID), data=signal, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_signal)

emmeans::emmeans(cue_prev_int_model_signal, pairwise ~ cue * prevsayyes)

check_collinearity(cue_prev_int_model_signal)
check_convergence(cue_prev_int_model_signal)

saveRDS(cue_prev_int_model_signal, "D:\\expecon_ms\\data\\behav\\mixed_models\\beta_prev_int_model_signal.rda")
cue_prev_int_model_signal <- readRDS("D:\\expecon_ms\\data\\behav\\mixed_models\\beta_prev_int_model_signal.rda")

# noise model

cue_prev_int_model_noise = glmer(sayyes ~ prevsayyes + beta + prevsayyes*beta +
                                    (prevsayyes + beta + prevsayyes*beta|ID), data=noise, 
                                  family=binomial(link='probit'),
                                  control=glmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)),
)

summary(cue_prev_int_model_noise)

emmeans::emmeans(cue_prev_int_model_noise, ~ cue * prevsayyes)

check_collinearity(cue_prev_int_model_noise)
check_convergence(cue_prev_int_model_noise)

saveRDS(cue_prev_int_model_noise, "D:\\expecon_ms\\data\\behav\\mixed_models\\beta_prev_int_model_noise.rda")
cue_prev_int_model_noise <- readRDS("D:\\expecon_ms\\data\\behav\\mixed_models\\beta_prev_int_model_noise.rda")


int = sjPlot::plot_model(cue_prev_int_model_noise, type='int',  mdrt.values = "meansd")+
  theme(plot.background = element_blank(),
        text = element_text(family = "Arial", size = 14)) +
  ylab('detection response') +
  xlab('previous choice')

int

# Save the plot as an SVG file
ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\cue_prev_int_int_signal_model.svg", plot = int, device = "svg")
ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\cue_prev_int_int_noise_model.svg", plot = int, device = "svg")

##################separate model for confident/unconfident previous response#######

acc_conf_model = glmer(correct ~ conf + (conf|ID), data=behav, 
                       family=binomial(link='probit'),
                       control=glmerControl(optimizer="bobyqa",
                                            optCtrl=list(maxfun=2e5)))

summary(acc_conf_model)

conf = filter(behav, prevconf==1)
unconf = filter(behav, prevconf==0)

cue_prev_int_model_conf = glmer(sayyes ~ isyes + prevsayyes + beta + isyes*beta+prevsayyes*beta +
                             (isyes + prevsayyes + beta + isyes*beta + prevsayyes*beta|ID), data=conf, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_conf)
check_convergence(cue_prev_int_model_conf)

summary(cue_prev_int_model_conf)

cue_prev_int_model_unconf = glmer(sayyes ~ isyes + prevsayyes + cue + isyes*cue+prevsayyes*cue +
                             (isyes + prevsayyes + cue + isyes*cue + prevsayyes*cue|ID), 
                             data=unconf, 
                             family=binomial(link='probit'),
                             control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_unconf)
check_convergence(cue_prev_int_model_unconf)

summary(cue_prev_int_model_unconf)

saveRDS(cue_prev_int_model_conf, "D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_int_model_conf.rda")
saveRDS(cue_prev_int_model_unconf, "D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_int_model_unconf.rda")

cue_prev_int_model_conf <- readRDS("D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_int_model_conf.rda")
cue_prev_int_model_unconf <- readRDS("D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_int_model_unconf.rda")

int = sjPlot::plot_model(cue_prev_int_model_unconf, type='int', title='Predicted probability of responding yes')

int = int[[2]]+
  theme(plot.background = element_blank(),
        text = element_text(family = "Arial", size = 14)) +
  ylab('detection response') +
  xlab('previous choice')

int

# Save the plot as an SVG file
ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\cue_prev_int_int_conf_model.svg", plot = int, device = "svg")
ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\cue_prev_int_int_unconf_model.svg", plot = int, device = "svg")


# Figure 5

combined_plot <- grid.arrange(plot_output1, plot_output2, ..., ncol = 3)

