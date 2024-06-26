#####################################ExPeCoN study#################################################
# generalized linear mixed models to estimate signal detection theory parameters
# including previous choice effects: Brain-behavior modelling

# author: Carina Forster
# email: forster@mpg.cbs.de
 
Sys.Date()

# libraries
library(lme4) # mixed models
library(lmerTest) # no p values without this package for linear mixed mdoels
library(dplyr)# pandas style
library(tidyr)
library(emmeans)
library(sjPlot)
library(ggplot2)
library(performance)
library(gridExtra)

# don't forget to give credit to the amazing authors of those packages
#citation("emmeans")

# Set the font family and size
par(family = "Arial", cex = 1.2)

####################################brain behav#####################################################
setwd("E:/expecon_ms")

filename_1 = paste("brain_behav_cleaned_source_1.csv", sep="")
filename_2 = paste("brain_behav_cleaned_source_2.csv", sep="")
brain_behav_path_1 <- file.path("data", "behav", filename_1)
brain_behav_path_2 <- file.path("data", "behav", filename_2)

behav_1 = read.csv(brain_behav_path_1)
behav_2 = read.csv(brain_behav_path_2)
################################prepare variables for linear mixed modelling #######################

# which dataset do you want to analyze?
expecon=1
behav = behav_1

# make factors for categorical variables:
behav$ID = as.factor(behav$ID) # subject ID
behav$isyes = as.factor(behav$isyes) # stimulus
behav$cue = as.factor(behav$cue) # probability for a signal
behav$prevresp = as.factor(behav$prevresp) # previous response
behav$previsyes = as.factor(behav$previsyes) # previous stimulus
behav$prevconf = as.factor(behav$prevconf) # previous confidence
behav$correct = as.factor(behav$correct) # performance
behav$prevcue = as.factor(behav$prevcue) # previous probability
behav$congruency <- as.integer(as.logical(behav$congruency))
behav$congruency_stim <- as.integer(as.logical(behav$congruency_stim))

# Remove NaN trials for model comparision (models neeed to have same amount of data)
# (first row for previous trial factor)
behav <- na.omit(behav) 

#rename beta power variable
behav$beta <- behav$beta_source_prob

hist(behav$beta)

# kick out very high or low beta values
behav <- behav[!(behav$beta > 3 | behav$beta < -3), ]
########################### Brain behavior GLMMs ###################################################

# replace stimulus probability regressor with beta power per trial 
# (neural correlate of stimulus probability)

beta_cue = lmer(beta ~ cue+ (cue|ID), 
             data=behav, control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5)))

summary(beta_cue)
###################################################################################################
### does prestimulus power predict detection, while controlling for previous choice

beta_base_glm <- glmer(sayyes ~ isyes + beta + cue +
                         beta*isyes + cue*isyes +
                    (isyes+cue|ID),
                  data = behav, family=binomial(link='probit'), 
                  control=glmerControl(optimizer="bobyqa",
                                       optCtrl=list(maxfun=2e5)))

# singularity with beta and beta*stimulus interaction
# fits beta for volatile env.
# check model performance
check_collinearity(beta_base_glm) # VIF should be < 3
check_convergence(beta_base_glm)

summary(beta_base_glm)

# save models to disk
filename = paste("beta_base_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", "brain_behav", filename)
saveRDS(beta_base_glm, cue_model_path)
beta_base_glm <- readRDS(cue_model_path)

# add previous response
beta_prev_glm <- glmer(sayyes ~ isyes + beta + cue + prevresp + 
                    beta*isyes + cue*isyes +
                     (isyes+cue+prevresp|ID),
                   data = behav, family=binomial(link='probit'), 
                   control=glmerControl(optimizer="bobyqa",
                                        optCtrl=list(maxfun=2e5)))

check_collinearity(beta_prev_glm) # VIF should be < 3
check_convergence(beta_prev_glm)

summary(beta_prev_glm)

# save models to disk
filename = paste("beta_prev_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", "brain_behav", filename)
saveRDS(beta_prev_glm, cue_model_path)
beta_prev_glm <- readRDS(cue_model_path)

##### no we fit the interaction between prestimulus power and previous choice
# beta interaction
beta_int_glm1 <- glmer(sayyes ~ isyes*beta*cue + beta*prevresp + cue*prevresp +
                        (isyes+cue + prevresp| ID),
                      data = behav, family=binomial(link='probit'), 
                      control=glmerControl(optimizer="bobyqa",
                                           optCtrl=list(maxfun=2e5)))

check_collinearity(beta_int_glm1) # VIF should be < 3
check_convergence(beta_int_glm1)

summary(beta_int_glm1)

# Simons' comment: beta independent of cue predicting response?
plot_model(beta_int_glm1, type = 'pred', terms=c('beta', 'isyes', 'cue'))

# Post hoc tests for behavior interaction
emm_model <- contrast(emmeans(beta_int_glm1, "prevresp", by = "cue"))
emm_model

# save models to disk
filename = paste("beta_int_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", "brain_behav", filename)
saveRDS(beta_int_glm1, cue_model_path)
beta_int_glm1 <- readRDS(cue_model_path)

plot_model(beta_int_glm1, type = 'pred', terms=c('beta', 'isyes', 'cue'))
# now plot figure 4

# Define the plots with custom theme and axis labels
plot1 <- plot_model(beta_int_glm1, type='pred', terms = c("beta", 'prevresp'), 
                    show.legend = FALSE, colors = c("#e319d0ff", "#35b85bff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 16)) +
  labs(x = "Pre-stimulus Beta Power", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(labels = function(x) paste0(x*100))  # Remove percentage sign

plot2 <- plot_model(beta_int_glm1, type='pred', terms = c("beta", 'isyes'), 
                    show.legend = FALSE, colors = c("#fba72aff", "#9c57c1ff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 16)) +
  labs(x = "Pre-stimulus Beta Power", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(labels = function(x) paste0(x*100))  # Remove percentage sign

plot3 <- plot_model(beta_int_glm1, type='pred', terms = c("cue", 'prevresp'), 
                    show.legend = FALSE, colors = c("#e319d0ff", "#35b85bff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 16)) +
  labs(x = "Stimulus Probability", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(labels = function(x) paste0(x*100)) +  # Remove percentage sign
  scale_x_continuous(breaks = c(0.25, 0.75), labels = c("low", "high"))

# volatile env.
plot4 <- plot_model(beta_int_glm2, type='pred', terms = c("beta", 'prevresp'), 
                    show.legend = FALSE, colors = c("#e319d0ff", "#35b85bff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 16)) +
  labs(x = "Pre-stimulus Beta Power", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(labels = function(x) paste0(x*100))  # Remove percentage sign

plot5 <- plot_model(beta_int_glm2, type='pred', terms = c("beta", 'isyes'), 
                    show.legend = FALSE, colors = c("#fba72aff", "#9c57c1ff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 16)) +
  labs(x = "Pre-stimulus Beta Power", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(labels = function(x) paste0(x*100))  # Remove percentage sign

plot6 <- plot_model(beta_int_glm2, type='pred', terms = c("cue", 'prevresp'), 
                    show.legend = FALSE, colors = c("#e319d0ff", "#35b85bff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 16)) +
  labs(x = "Stimulus Probability", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(labels = function(x) paste0(x*100)) +  # Remove percentage sign
  scale_x_continuous(breaks = c(0.25, 0.75), labels = c("low", "high"))

# Arrange plots into a grid
fig4 = grid.arrange(plot2, plot3, plot1, plot5, plot6, plot4, ncol = 3, nrow = 2)

# Define the directory path
directory <- "E:\\expecon_ms\\figs\\manuscript_figures\\figure4_detection_modelling"

# Set the dimensions for an A4 page
a4_width <- 8.3  # Width in inches
a4_height <- 6 # Height in inches

# Save the arranged grid of plots as PNG with 300 DPI
ggsave(file.path(directory, "figure4.png"), plot = fig4, 
       width = a4_width, height = a4_height, dpi = 300)

# Save the arranged grid of plots as SVG with 300 DPI
ggsave(file.path(directory, "figure.svg"), plot = fig4, 
       width = a4_width, height = a4_height, dpi = 300)

# save table to docx
filename = paste("beta_expecon", expecon, ".docx", sep="_")
output_file_path_beta <- file.path("figs", "manuscript_figures", "Tables", filename)

models = list("base" = beta_base_glm, "add previous response" = beta_prev_glm,
              "interaction" = beta_int_glm1)

modelsummary::modelsummary(models, estimate  = "{estimate} [{conf.low}, {conf.high}], {stars}", 
                           statistic = NULL,  output = output_file_path_beta)

################################ fit confidence with beta power ####################################

behav$sayyes = as.factor(behav$sayyes)

conf_full_model_1 = glmer(conf ~ sayyes*cue + prevconf + correct + 
                                 (sayyes*cue + prevconf + correct|ID), data=behav, 
                               family=binomial(link='probit'),
                               control=glmerControl(optimizer="bobyqa",
                                                    optCtrl=list(maxfun=2e5)))

check_collinearity(conf_full_model_1)
check_convergence(conf_full_model_1)

summary(conf_full_model_1)

# Post hoc tests for behavior interaction
contrast(emmeans(conf_full_model_1, "sayyes", by = c("cue")))

# now replace the cue with beta power

conf_full_model_beta_1 = glmer(conf ~ sayyes*beta + prevconf + correct +
                            (sayyes+ prevconf + correct|ID), data=behav, 
                          family=binomial(link='probit'),
                          control=glmerControl(optimizer="bobyqa",
                                               optCtrl=list(maxfun=2e5)))

check_collinearity(conf_full_model_beta_1)
check_convergence(conf_full_model_beta_1)

summary(conf_full_model_beta_1)

# save to disk
filename = paste("conf_model_full_model", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", "behavior", filename)
saveRDS(conf_full_model_1, cue_model_path)
conf_full_model_1 <- readRDS(cue_model_path)

filename = paste("conf_model_full_model_beta", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", "behavior", filename)
saveRDS(conf_full_model_beta_1, cue_model_path)
conf_full_model_beta_1 <- readRDS(cue_model_path)

# now for volatile env.
conf_full_model_2 = glmer(conf ~ sayyes*cue + prevconf + correct +
                            (sayyes*cue + prevconf + correct|ID), data=behav, 
                          family=binomial(link='probit'),
                          control=glmerControl(optimizer="bobyqa",
                                               optCtrl=list(maxfun=2e5)))

check_collinearity(conf_full_model_2)
check_convergence(conf_full_model_2)

summary(conf_full_model_2)

# Post hoc tests for behavior interaction
contrast(emmeans(conf_full_model_2, "sayyes", by = c("cue")))

# now replace the cue with beta power

conf_full_model_beta_2 = glmer(conf ~ sayyes*beta + prevconf + correct +
                                 (sayyes+ prevconf + correct|ID), data=behav, 
                               family=binomial(link='probit'),
                               control=glmerControl(optimizer="bobyqa",
                                                    optCtrl=list(maxfun=2e5)))

check_collinearity(conf_full_model_beta_2)
check_convergence(conf_full_model_beta_2)

summary(conf_full_model_beta_2)

# save to disk
filename = paste("conf_model_full_model", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", "behavior", filename)
saveRDS(conf_full_model_2, cue_model_path)
conf_full_model_2 <- readRDS(cue_model_path)

filename = paste("conf_model_full_model_beta", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", "behavior", filename)
saveRDS(conf_full_model_beta_2, cue_model_path)
conf_full_model_beta_2 <- readRDS(cue_model_path)

# save models as tables for manuscript
#https://modelsummary.com/articles/modelsummary.html

filename_conf = paste("conf_expecon", expecon, ".docx", sep="_")
output_file_path_conf<- file.path("figs", "manuscript_figures", "Tables", filename_conf)

models = list("prob" = conf_full_model_2, "beta" = conf_full_model_beta_2)

modelsummary::modelsummary(models, estimate  = "{estimate} [{conf.low}, {conf.high}], {stars}", 
                           statistic = NULL,  output = output_file_path_conf)

# plot figure 5
# Define the common y-axis limits
common_ylim <- c(0, 1)  # Adjust the range as needed

# Define the plots with custom theme
plot1 <- plot_model(conf_full_model_1, type='pred',terms = c("sayyes", 'cue'), 
                    show.legend = TRUE, colors = c("#357db8ff", "#e31919ff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 16)) +
  labs(x = "Detection Response", 
       y = "Predicted Probability of\nHigh Confidence Rating") +
  scale_x_continuous(breaks = c(0, 1), labels = c("no", "yes")) +  # Custom x-axis labels
  scale_y_continuous(labels = function(x) paste0(x*100)) +  # Remove percentage sign
 coord_cartesian(ylim = common_ylim)  # Set common y-axis limits

plot2 <- plot_model(conf_full_model_beta_1, type='pred',terms = c("sayyes", 'beta [1, -1]'), 
                    show.legend = TRUE, colors = c("#357db8ff", "#e31919ff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 16)) +
  labs(x = "Detection Response", 
       y = "Predicted Probability of\nHigh Confidence Rating") +
  scale_x_continuous(breaks = c(0, 1), labels = c("no", "yes")) +  # Custom x-axis labels
  scale_y_continuous(labels = function(x) paste0(x*100)) +  # Remove percentage sign
  coord_cartesian(ylim = common_ylim)  # Set common y-axis limits

plot3 <- plot_model(conf_full_model_2, type='pred',terms = c("sayyes", 'cue'), 
                    show.legend = TRUE, colors = c("#357db8ff", "#e31919ff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 16)) +
  labs(x = "Detection Response", 
       y = "Predicted Probability of\nHigh Confidence Rating") +
  scale_x_continuous(breaks = c(0, 1), labels = c("no", "yes")) +  # Custom x-axis labels
  scale_y_continuous(labels = function(x) paste0(x*100)) +  # Remove percentage sign
  coord_cartesian(ylim = common_ylim)  # Set common y-axis limits

plot4 <- plot_model(conf_full_model_beta_2, type='pred',terms = c("sayyes", 'beta [1, -1]'), 
                    show.legend = TRUE, colors = c("#357db8ff", "#e31919ff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 16)) +
  labs(x = "Detection response", 
       y = "Predicted Probability of\nHigh Confidence Rating") +
  scale_x_continuous(breaks = c(0, 1), labels = c("no", "yes")) +  # Custom x-axis labels
  scale_y_continuous(labels = function(x) paste0(x*100)) +  # Remove percentage sign
  coord_cartesian(ylim = common_ylim)  # Set common y-axis limits

# Arrange plots into a grid
fig5 <- grid.arrange(plot1, plot3, plot2, plot4, ncol = 2, nrow = 2)

# Define the directory path
fig5_path <- "E:\\expecon_ms\\figs\\manuscript_figures\\figure5_confidence_modelling"

# Save the arranged grid of plots as PNG with 300 DPI
ggsave(file.path(fig5_path, "fig5.png"), plot = fig5, width = a4_width, height = a4_height,
        dpi = 300)

# Save the arranged grid of plots as SVG with 300 DPI
ggsave(file.path(fig5_path, "fig5.svg"), plot = fig5, 
       width = a4_width, height = a4_height, dpi = 300)

beta_values = c(1, -1)

# Post hoc tests for behavior interaction
contrast(emmeans(conf_full_model_beta_2, "beta", by = "sayyes", at = list(beta=beta_values)))
contrast(emmeans(conf_full_model_2, "cue", by = "sayyes"))

# save to disk
filename = paste("conf_model_full_model", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", "behavior", filename)
saveRDS(conf_full_model_1, cue_model_path)
cue_prev_int_model <- readRDS(cue_model_path)

filename = paste("conf_model_full_model_beta", expecon, ".rda", sep="_")
cue_model_path = file.path("data", "behav", "mixed_models", "behavior", filename)
saveRDS(conf_full_model_beta_1, cue_model_path)
cue_prev_int_model <- readRDS(cue_model_path)
############################################### Model comparision ##################################

# Likelihood ratio tests
anova(beta_base_glm, beta_prev_glm)
anova(beta_prev_glm, beta_int_glm2)

# difference in AIC and BIC
diff_aic_1 = AIC(beta_prev_glm) - AIC(beta_base_glm)
diff_bic_1 = BIC(beta_prev_glm) - BIC(beta_base_glm)
print(diff_aic_1)
print(diff_bic_1)

diff_aic_2 = AIC(beta_int_glm2) - AIC(beta_prev_glm)
diff_bic_2 = BIC(beta_int_glm1) - BIC(beta_prev_glm)

print(diff_aic_2)
print(diff_bic_2)

##########################congruency###############################################################

# does beta power per trial predict congruent responses in both probability conditions?

con_beta = glmer(congruency ~ beta * cue + isyes + (cue|ID), data=behav, 
                 family=binomial(link='probit'), 
                 control=glmerControl(optimizer="bobyqa",
                                      optCtrl=list(maxfun=2e5)))
summary(con_beta)

# Post hoc tests for behavior interaction
emm_model <- emmeans(con_beta, "cue", by = "beta")
con <- contrast(emm_model)
con

con_plot_beta = plot_model(con_beta, type='int', mdrt.values = "meansd")

# save congruency plot
filename = paste("congruency_beta_", expecon, ".svg", sep="")
savepath_fig5 = file.path("figs", "manuscript_figures", "figure5_brain_behavior", filename)
ggsave(savepath_fig5, dpi = 300, height = 8, width = 10, plot=con_plot_alpha)