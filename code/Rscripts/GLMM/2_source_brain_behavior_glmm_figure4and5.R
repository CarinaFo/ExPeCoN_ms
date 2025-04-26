#####################################ExPeCoN study#################################################
# generalized linear mixed models to estimate signal detection theory parameters
# including previous choice effects: Brain-behavior modelling
# script produces figure 4 and 5 from Forster et al., 20205

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

####################################load data#####################################################
setwd("E:/expecon_ms")
study1 = "brain_behav_cleaned_1.csv"
study2 = "brain_behav_cleanded_2.csv"

brain_behav_path_1 <- file.path("data", "behav", study1)
brain_behav_path_2 <- file.path("data", "behav", study2)

behav = read.csv(brain_behav_path_1)
################################prepare variables for linear mixed modelling #######################

# which dataset do you want to analyze?
expecon=2

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
behav$level_0 <- NULL
behav <- na.omit(behav) 

#rename beta power variable
behav$beta <- behav$beta_source_prob

hist(behav$beta)

# kick out very high or low beta values
behav <- behav[!(behav$beta > 3 | behav$beta < -3), ]
########################### Brain behavior GLMMs ###################################################

# replace stimulus probability regressor with beta power per trial 
# (neural correlate of stimulus probability)

beta_cue = glmer(cue ~ beta +  (beta|ID), family=binomial(link='probit'), 
             data=behav, control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5)))

summary(beta_cue)

# Wald Confidence Interval
confint(beta_cue, method='Wald')

summary(lmerTest::lmer(respt1 ~ beta + cue + (beta+cue|ID), data= behav))
###################################################################################################
### does prestimulus power predict detection, while controlling for previous choice

beta_base_glm <- glmer(sayyes ~ isyes * prevresp + 
                    ( isyes *prevresp|ID),
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
beta_int_glm2 <- readRDS(cue_model_path)

#####################
###################################### Plot figure 4
# revision: make sure the y axis limits align

# Define correct y-axis limits (assuming predicted probabilities are between 0 and 1)
y_limits_beta_prevresp <- c(0, 0.20)  
y_limits_beta_isyes <- c(0, 0.75)     
y_limits_cue_prevresp <- c(0, 0.20)   

y_breaks_beta_prevresp <- seq(0, 0.2, by = 0.05)   
y_breaks_beta_isyes <- seq(0, 0.75, by = 0.25)     
y_breaks_cue_prevresp <- seq(0, 0.2, by = 0.05)     

# Stable environment plots
plot1 <- plot_model(beta_int_glm1, type='pred', terms = c("beta", 'prevresp'), 
                    show.legend = FALSE, colors = c("#e319d0ff", "#35b85bff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 13)) +
  labs(x = "Pre-stimulus Beta Power", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(limits = y_limits_beta_prevresp, breaks = y_breaks_beta_prevresp, labels = function(x) paste0(x * 100))

plot2 <- plot_model(beta_int_glm1, type='pred', terms = c("beta", 'isyes'), 
                    show.legend = FALSE, colors = c("#fba72aff", "#9c57c1ff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 13)) +
  labs(x = "Pre-stimulus Beta Power", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(limits = y_limits_beta_isyes, breaks = y_breaks_beta_isyes, labels = function(x) paste0(x * 100))

plot3 <- plot_model(beta_int_glm1, type='pred', terms = c("cue", 'prevresp'), 
                    show.legend = FALSE, colors = c("#e319d0ff", "#35b85bff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 13)) +
  labs(x = "Stimulus Probability", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(limits = y_limits_cue_prevresp, breaks = y_breaks_cue_prevresp, labels = function(x) paste0(x * 100)) +
  scale_x_continuous(breaks = c(0.25, 0.75), labels = c("low", "high"))

# Volatile environment plots
plot4 <- plot_model(beta_int_glm2, type='pred', terms = c("beta", 'prevresp'), 
                    show.legend = FALSE, colors = c("#e319d0ff", "#35b85bff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 13)) +
  labs(x = "Pre-stimulus Beta Power", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(limits = y_limits_beta_prevresp, breaks = y_breaks_beta_prevresp, labels = function(x) paste0(x * 100))

plot5 <- plot_model(beta_int_glm2, type='pred', terms = c("beta", 'isyes'), 
                    show.legend = FALSE, colors = c("#fba72aff", "#9c57c1ff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 13)) +
  labs(x = "Pre-stimulus Beta Power", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(limits = y_limits_beta_isyes, breaks = y_breaks_beta_isyes, labels = function(x) paste0(x * 100))

plot6 <- plot_model(beta_int_glm2, type='pred', terms = c("cue", 'prevresp'), 
                    show.legend = FALSE, colors = c("#e319d0ff", "#35b85bff"), title=" ") +
  theme(axis.text = element_text(size = 13),
        axis.title = element_text(size = 13)) +
  labs(x = "Stimulus Probability", 
       y = "Predicted Probability\nof Yes Response") +
  scale_y_continuous(limits = y_limits_cue_prevresp, breaks = y_breaks_cue_prevresp, labels = function(x) paste0(x * 100)) +
  scale_x_continuous(breaks = c(0.25, 0.75), labels = c("low", "high"))

# Arrange plots into a grid
fig4 <- grid.arrange(plot2, plot3, plot1, plot5, plot6, plot4, ncol = 3, nrow = 2)

a4_width <- 11.69  # A4-Breite in Zoll (Querformat)
a4_height <- 8.27  # A4-Höhe in Zoll (Querformat)

# Save the arranged grid of plots as PNG with 300 DPI
ggsave(file.path(directory, "figure4.png"), plot = fig4, 
       width = a4_width, height = a4_height, dpi = 300)

# Save the arranged grid of plots as SVG with 300 DPI
ggsave(file.path(directory, "figure4.svg"), plot = fig4, 
       width = a4_width, height = a4_height, dpi = 300)

                 
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

############# 
######################################plot figure 5

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