#####################################ExPeCoN study#################################################
# generalized linear mixed models to estimate signal detection theory parameters
# including previous choice effects: Brain-behavior modelling

# author: Carina Forster
# email: forster@mpg.cbs.de

# libraries
library(lme4) # mixed models
library(lmerTest) # no p values without this package for linear mixed mdoels
library(dplyr)# pandas style
library(tidyr)
library(data.table) # for shift function
library(htmlTable)
library(emmeans)
library(performance)
library(brms)
library(sjPlot)
library(ggplot2)

# don't forget to give credit to the amazing authors of those packages
#citation("emmeans")

# Set the font family and size
par(family = "Arial", cex = 1.2)

# which dataset to analyze (1 => mini-block, 2 => trial-by-trial design)

expecon <- 1

####################################brain behav#####################################################
setwd("E:/expecon_ms")

filename = paste("brain_behav_cleaned_", expecon, ".csv", sep="")
brain_behav_path <- file.path("data", "behav", filename)

behav = read.csv(brain_behav_path)
  
################################prepare variables for linear mixed modelling #######################

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

# Remove NaN trials for model comparision (models neeed to have same amount of data)(first row for
# previous trial factor)
behav <- na.omit(behav) 

#rename alpha and beta power variables
behav$alpha <- behav$pre_alpha
behav$beta <- behav$pre_beta
########################### Brain behavior GLMMs ###################################################

# does beta or alpha predict the stimulus? control analysis:

alpha_stim <- glm(isyes ~ -1 + alpha,
                        data = behav, family=binomial(link='probit'))
summary(alpha_stim)

beta_stim <- glm(isyes ~ -1 + beta,
                  data = behav, family=binomial(link='probit'))
summary(beta_stim)

# replace stimulus probability regressor with beta or alpha power per trial 
# (neural correlate of stimulus probability)

alpha_cue = lmer(alpha ~ cue + (cue|ID), data=behav, 
             control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5)))

summary(alpha_cue)

beta_cue = lmer(beta ~ cue+ (cue|ID), 
             data=behav, control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5)))

summary(beta_cue)

# the cue sign. predicts beta in both environments, alpha only in the stable env.

###################################################################################################
### does prestimulus power predict detection, while controlling for previous choice


alpha_base_glm <- glmer(sayyes ~ isyes + alpha + cue + prevresp +
                          alpha*isyes + cue*isyes +
                          (isyes + cue + prevresp|ID),
                        data = behav, family=binomial(link='probit'), 
                        control=glmerControl(optimizer="bobyqa",
                                             optCtrl=list(maxfun=2e5)))

# interaction term doesn't fit for volatile env.
# check model performance
check_collinearity(alpha_base_glm) # VIF should be < 3
check_convergence(alpha_base_glm)

summary(alpha_base_glm)

beta_base_glm <- glmer(sayyes ~ isyes + beta + cue + prevresp +
                         beta*isyes + cue*isyes +
                    (isyes + cue + prevresp|ID),
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
filename = paste("alpha_base_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(alpha_base_glm, cue_model_path)
alpha_base_glm <- readRDS(cue_model_path)


filename = paste("beta_base_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(beta_base_glm, cue_model_path)
beta_base_glm <- readRDS(cue_model_path)

# add previous response
alpha_prev_glm <- glmer(sayyes ~ isyes + alpha + cue + prevresp + 
                         alpha*isyes + cue*isyes + cue*prevresp +
                         (isyes + prevresp + cue | ID),
                       data = behav, family=binomial(link='probit'), 
                       control=glmerControl(optimizer="bobyqa",
                                            optCtrl=list(maxfun=2e5)))
# did not converge with interaction term
# check model performance
check_collinearity(alpha_prev_glm) # VIF should be < 3
check_convergence(alpha_prev_glm)

summary(alpha_prev_glm)

beta_prev_glm <- glmer(sayyes ~ isyes + beta + cue + prevresp + 
                     beta*isyes + cue*isyes + cue*prevresp +
                     (isyes + prevresp +cue|ID),
                   data = behav, family=binomial(link='probit'), 
                   control=glmerControl(optimizer="bobyqa",
                                        optCtrl=list(maxfun=2e5)))

check_collinearity(beta_prev_glm) # VIF should be < 3
check_convergence(beta_prev_glm)

summary(beta_prev_glm)

# save models to disk
filename = paste("alpha_prev_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(alpha_prev_glm, cue_model_path)
alpha_prev_glm <- readRDS(cue_model_path)


filename = paste("beta_prev_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(beta_prev_glm, cue_model_path)
beta_prev_glm <- readRDS(cue_model_path)

##### no we fit the interaction between prestimulus power and previous choice

# alpha interaction
alpha_int_glm <- glmer(sayyes ~ isyes + alpha + cue + prevresp + 
                                alpha*isyes + cue*isyes + cue*prevresp + alpha*prevresp + 
                                (isyes + prevresp +cue|ID),
                                data = behav, family=binomial(link='probit'), 
                                control=glmerControl(optimizer="bobyqa",
                                                     optCtrl=list(maxfun=2e5)))

check_collinearity(alpha_int_glm) # VIF should be < 3
check_convergence(alpha_int_glm)

summary(alpha_int_glm)

# beta interaction
beta_int_glm <- glmer(sayyes ~ isyes + beta + cue + prevresp + 
                        beta*isyes + cue*isyes + cue*prevresp + beta*prevresp +
                        (isyes + prevresp + cue| ID),
                      data = behav, family=binomial(link='probit'), 
                      control=glmerControl(optimizer="bobyqa",
                                           optCtrl=list(maxfun=2e5)))

check_collinearity(beta_int_glm) # VIF should be < 3
check_convergence(beta_int_glm)

summary(beta_int_glm)

# Post hoc tests for behavior interaction
emm_model <- emmeans(beta_int_glm, "beta", by = "prevresp")
con <- contrast(emm_model)
con

# save models to disk
filename = paste("alpha_int_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(alpha_int_glm, cue_model_path)
alpha_int_glm <- readRDS(cue_model_path)


filename = paste("beta_int_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", filename)
saveRDS(beta_int_glm, cue_model_path)
beta_int_glm <- readRDS(cue_model_path)

############################################### Model comparision ##################################

# Likelihood ratio tests
anova(beta_base_glm, beta_prev_glm)
anova(alpha_base_glm, alpha_prev_glm)
anova(beta_prev_glm, beta_int_glm)
anova(alpha_prev_glm, alpha_int_glm)

# difference in AIC and BIC
diff_aic_1 = AIC(beta_prev_model) - AIC(beta_beta_glm)
diff_bic_1 = BIC(beta_prev_model) - BIC(beta_base_glm)
print(diff_aic_1)
print(diff_bic_1)

diff_aic_2 = AIC(beta_int_glm) - AIC(beta_prev_glm)
diff_bic_2 = BIC(beta_int_glm) - BIC(beta_prev_glm)

print(diff_aic_2)
print(diff_bic_2)

# save table to html
table1 = sjPlot::tab_model(beta_base_glm, beta_prev_glm, beta_int_glm, 
                           show.aic=TRUE, show.loglik=TRUE)

table2 = sjPlot::tab_model(alpha_base_glm, alpha_prev_glm, alpha_int_glm, 
                           show.aic=TRUE, show.loglik=TRUE)

filename = paste("beta_expecon", expecon, ".html", sep="_")
output_file_path <- file.path("figs", "manuscript_figures", "Tables", filename)
htmlTable(table1, file = output_file_path)

filename = paste("alpha_expecon", expecon, ".html", sep="_")
output_file_path <- file.path("figs", "manuscript_figures", "Tables", filename)
htmlTable(table2, file = output_file_path)
##########################congruency###############################################################

# does beta power per trial predict congruent responses in both probability conditions?

con_beta = glmer(congruency ~ beta * cue + isyes + (cue|ID), data=behav, 
                 family=binomial(link='probit'), 
                 control=glmerControl(optimizer="bobyqa",
                                      optCtrl=list(maxfun=2e5)))
summary(con_beta)

con_alpha = glmer(congruency ~ alpha * cue + isyes + (cue|ID), data=behav, 
                 family=binomial(link='probit'), 
                 control=glmerControl(optimizer="bobyqa",
                                      optCtrl=list(maxfun=2e5)))
summary(con_alpha)

# Post hoc tests for behavior interaction
emm_model <- emmeans(con_beta, "cue", by = "beta")
con <- contrast(emm_model)
con

con_plot_beta = plot_model(con_beta, type='int', mdrt.values = "meansd")
con_plot_alpha = plot_model(con_alpha, type='int', mdrt.values = "meansd")
# save congruency plot
filename = paste("congruency_alpha_", expecon, ".svg", sep="")
savepath_fig5 = file.path("figs", "manuscript_figures", "figure5_brain_behavior", filename)
ggsave(savepath_fig5, dpi = 300, height = 8, width = 10, plot=con_plot_alpha)