################################Plot GLMM estimates and interactions###############################


# Author: Carina Forster
# Email: forster@mpg.cbs.de

# libraries
library(lme4) # mixed models
library(lmerTest) # no p values without this package for linear mixed mdoels
library(dplyr)# pandas style
library(tidyr)
library(sjPlot)
library(ggplot2)
library(gridExtra)

# don't forget to give credit to the amazing authors of those packages
#citation("emmeans")

# Set the font family and size
par(family = "Arial", cex = 1.2)

# which dataset to analyze (1 => mini-block, 2 => trial-by-trial design)
theme_set(theme_sjplot())

# set base directory
setwd("E:/expecon_ms")

####################################################################################################
# read behavioral models
expecon = 1

# behavior only
filename = paste("cue_prev_model_expecon", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
cue_prev_1 <- readRDS(cue_model_path)

filename = paste("cue_prev_int_model", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
cue_prev_int_1 <- readRDS(cue_model_path)

expecon = 2

filename = paste("cue_prev_model_expecon", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
cue_prev_2 <- readRDS(cue_model_path)

filename = paste("cue_prev_int_model", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
cue_prev_int_2 <- readRDS(cue_model_path)

# plot estimates for behavior only
# where to save the figure
save_path_figs = file.path("figs", "manuscript_figures", "figure2_expecon2_paradigm_behav",
                           "supplfig")
setwd(save_path_figs)

est_cue1 = plot_model(cue_prev_1, type='est', 
                      title='yes response ~',
                      sort.est = TRUE, transform='plogis', show.values =TRUE, 
                      value.offset = 0.3, colors='black')

est_cue2 = plot_model(cue_prev_2, type='est', 
                      title='yes response ~',
                      sort.est = TRUE, transform='plogis', show.values =TRUE, 
                      value.offset = 0.3, colors='black')

est_cue_prev1 = plot_model(cue_prev_int_1, type='est',
                           sort.est = TRUE, transform='plogis', show.values =TRUE, 
                           value.offset = 0.3, colors='black')

est_cue_prev2 = plot_model(cue_prev_int_2, type='est', 
                           sort.est = TRUE, transform='plogis', show.values =TRUE, 
                           value.offset = 0.3, colors='black')

# arange plots in a grid
g = arrangeGrob(est_cue1, est_cue2, est_cue_prev1, est_cue_prev2, nrow = 2)

# save figure
ggsave('behavior_model_estimates.svg', dpi = 300, height = 8, width = 10, plot=g)

################################# plot interactions ################################################

# change the order of predictors (x and label)
cue_signal_int1 = plot_model(cue_prev_int_1, type='pred',terms = c("cue", 'isyes'))
cue_signal_int2 = plot_model(cue_prev_int_2, type='pred', terms = c("cue", "isyes"))

cue_prev_int1 = plot_model(cue_prev_int_1, type='pred',terms = c("cue", 'prevresp'))
cue_prev_int2 = plot_model(cue_prev_int_2, type='pred', terms = c("cue", "prevresp"))

# arange plots in a grid
g = arrangeGrob(cue_signal_int1, cue_signal_int2, cue_prev_int1, cue_prev_int2,
                nrow = 2)

# save figure
ggsave('behavior_model_interactions.svg', dpi = 300, height = 8, width = 10, plot=g)

###################################################################################################
# read brain behavior models

expecon = 1
filename = paste("alpha_base_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
alpha_base_glm_1 <- readRDS(cue_model_path)

filename = paste("beta_base_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
beta_glm_1 <- readRDS(cue_model_path)

filename = paste("alpha_prev_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
alpha_prev_glm_1 <- readRDS(cue_model_path)

filename = paste("beta_prev_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
beta_prev_glm_1 <- readRDS(cue_model_path)

filename = paste("beta_int_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
beta_int_glm_1 <- readRDS(cue_model_path)

filename = paste("alpha_int_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
alpha_int_glm_1 <- readRDS(cue_model_path)

expecon = 2

filename = paste("alpha_base_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
alpha_base_glm_2 <- readRDS(cue_model_path)

filename = paste("beta_base_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
beta_base_glm_2 <- readRDS(cue_model_path)

filename = paste("alpha_prev_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
alpha_prev_glm_2 <- readRDS(cue_model_path)

filename = paste("beta_prev_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
beta_prev_glm_2 <- readRDS(cue_model_path)

filename = paste("beta_int_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
beta_int_glm_2 <- readRDS(cue_model_path)

filename = paste("alpha_int_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
alpha_int_glm_2 <- readRDS(cue_model_path)

######################plot brain behavior estimates ###############################################

# where to save the figure
save_path_figs = file.path("figs", "manuscript_figures", "figure5_brain_behavior")
setwd(save_path_figs)

est_alpha_expecon1 = plot_model(alpha_int_glm_1, type='est', 
                                title='yes response ~',
                                sort.est = TRUE, transform='plogis', show.values =TRUE, 
                                value.offset = 0.3, colors='black')

est_beta_expecon1 = plot_model(beta_int_glm_1, type='est', 
                               title='yes response ~',
                               sort.est = TRUE, transform='plogis', show.values =TRUE, 
                               value.offset = 0.3, colors='black')

est_alpha_expecon2 = plot_model(alpha_int_glm_2, type='est', 
                                title='yes response ~',
                                sort.est = TRUE, transform='plogis', show.values =TRUE, 
                                value.offset = 0.3, colors='black')

est_beta_expecon2 = plot_model(beta_int_glm_2, type='est', 
                               title='yes response ~',
                               sort.est = TRUE, transform='plogis', show.values =TRUE, 
                               value.offset = 0.3, colors='black')

# mean plus minus one sd for continious variables

intg = arrangeGrob(est_alpha_expecon1, est_alpha_expecon2, est_beta_expecon1, est_beta_expecon2, 
                   nrow=2)

ggsave('model_brain_behavior.svg', dpi = 300, height = 8, width = 10, plot=intg)
################################# plot interactions ################################################

# change the order of predictors (x and label)
beta_1 = plot_model(beta_int_glm_1, type='pred',terms = c("beta", 'isyes'))
beta_2 = plot_model(beta_int_glm_2, type='pred', terms = c("beta", "isyes"))
beta_int_1 = plot_model(beta_int_glm_1, type='pred',terms = c("beta", 'prevresp'))
beta_int_2 = plot_model(beta_int_glm_2, type='pred', terms = c("beta", "prevresp"))


alpha_1 = plot_model(alpha_int_glm_1, type='pred',terms = c("alpha", 'isyes'))
alpha_2 = plot_model(alpha_int_glm_2, type='pred', terms = c("alpha", "isyes"))
alpha_int_1 = plot_model(alpha_int_glm_1, type='pred',terms = c("alpha", 'prevresp'))
alpha_int_2 = plot_model(alpha_int_glm_2, type='pred', terms = c("alpha", "prevresp"))

# arange plots in a grid:beta
g = arrangeGrob(beta_1, beta_2, beta_int_1, beta_int_2, 
                nrow = 2)
# alpha
g_alpha = arrangeGrob(alpha_1, alpha_2, alpha_int_1, alpha_int_2, 
                nrow = 2)
# save figure
ggsave('brain_behavior_model_interactions_beta.svg', dpi = 300, height = 8, width = 10, plot=g)
ggsave('brain_behavior_model_interactions_alpha.svg', dpi = 300, height = 8, width = 10, plot=g_alpha)