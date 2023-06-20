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

# Set the font family and size

par(family = "Arial", cex = 1.2)

####################################################################################################

# set working directory and load clean, behavioral dataframe

behav = read.csv("D:\\expecon_ms\\data\\behav\\behav_df\\prepro_behav_data.csv")

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

# fit sdt model

cue_model = glmer(sayyes ~ isyes+cue+isyes*cue + (isyes+cue+isyes*cue|ID), data=behav, 
                  family=binomial(link='probit'),
                  control=glmerControl(optimizer="bobyqa",
                  optCtrl=list(maxfun=2e5)),
                  )

check_collinearity(cue_model)
check_convergence(cue_model)

saveRDS(cue_model, "D:\\expecon_ms\\data\\behav\\mixed_models\\cue_model.rda")

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

cue_prev_int_model = glmer(sayyes ~ isyes + prevsayyes + cue + prevsayyes*cue + isyes*cue +
                            (isyes + prevsayyes + cue + prevsayyes*cue + isyes*cue|ID), data=behav, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                           optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model)
check_convergence(cue_prev_int_model)

saveRDS(cue_prev_int_model, "D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_int_model.rda")

cue_prev_int_model <- readRDS("D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_int_model.rda")

summary(cue_prev_int_model)

est = sjPlot::plot_model(cue_prev_int_model, type='est', title='detection response ~',
                         axis.lim=c(0.8,8)) +
  theme(plot.background = element_blank(),
        text = element_text(family = "Arial", size = 14)) +
  ylab("Odds Ratio")

est

int = sjPlot::plot_model(cue_prev_int_model, type='int', title='Predicted probability of responding yes')

int = int[[1]]+
  theme(plot.background = element_blank(),
        text = element_text(family = "Arial", size = 14)) +
  ylab('detection response') +
  xlab('stimulus')

int

# Save the plot as an SVG file
ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\cue_prev_int_int_model.svg", plot = int, device = "svg")
ggsave("D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\cue_prev_int_est_model.svg", plot = est, device = "svg")

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

# save table to PDF
table1 = sjPlot::tab_model(cue_model, cue_prev_model, cue_prev_int_model, 
                           show.aic=TRUE, show.loglik=TRUE)

# Save the output as an HTML file
output_file <- "D:\\expecon_ms\\figs\\manuscript_figures\\Figure2\\table1.html"
htmlTable(table1, file = output_file)

###########################separate models for signal and noise trials#############################

signal = filter(behav, isyes==1)
noise = filter(behav, isyes==0)

cue_prev_int_model_signal = glmer(sayyes ~ prevsayyes + cue + prevsayyes*cue +
                             (prevsayyes + cue + prevsayyes*cue|ID), data=signal, 
                           family=binomial(link='probit'),
                           control=glmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_signal)
check_convergence(cue_prev_int_model_signal)

saveRDS(cue_prev_int_model_signal, "D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_int_model_signal.rda")
cue_prev_int_model_signal <- readRDS("D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_int_model_signal.rda")

summary(cue_prev_int_model_signal)

# noise model

cue_prev_int_model_noise = glmer(sayyes ~ prevsayyes + cue + prevsayyes*cue +
                                    (prevsayyes + cue + prevsayyes*cue|ID), data=noise, 
                                  family=binomial(link='probit'),
                                  control=glmerControl(optimizer="bobyqa",
                                                       optCtrl=list(maxfun=2e5)),
)

check_collinearity(cue_prev_int_model_noise)
check_convergence(cue_prev_int_model_noise)

saveRDS(cue_prev_int_model_noise, "D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_int_model_noise.rda")
cue_prev_int_model_noise <- readRDS("D:\\expecon_ms\\data\\behav\\mixed_models\\cue_prev_int_model_noise.rda")

summary(cue_prev_int_model_noise)

int = sjPlot::plot_model(cue_prev_int_model_noise, type='int', title='Predicted probability of responding yes')+
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

cue_prev_int_model_conf = glmer(sayyes ~ isyes + prevsayyes + cue + isyes*cue+prevsayyes*cue +
                             (isyes + prevsayyes + cue + isyes*cue + prevsayyes*cue|ID), data=conf, 
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
