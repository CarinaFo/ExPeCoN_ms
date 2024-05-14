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

# don't forget to give credit to the amazing authors of those packages
#citation("emmeans")

# Set the font family and size
par(family = "Arial", cex = 1.2)

####################################brain behav#####################################################
setwd("E:/expecon_ms")

filename_1 = paste("brain_behav_cleaned_1.csv", sep="")
filename_2 = paste("brain_behav_cleaned_2.csv", sep="")
brain_behav_path_1 <- file.path("data", "behav", filename_1)
brain_behav_path_2 <- file.path("data", "behav", filename_2)

behav_1 = read.csv(brain_behav_path_1)
behav_2 = read.csv(brain_behav_path_2)
################################prepare variables for linear mixed modelling #######################

# which dataset do you want to analyze?
expecon=2
behav = behav_2

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
behav$beta <- behav$pre_beta
behav$alpha <- behav$pre_alpha

hist(behav$beta)

# Subsetting to keep rows where source_s1 values are within the desired range
behav <- behav[!(behav$beta > 3 | behav$beta < -3), ]
behav <- behav[!(behav$alpha > 3 | behav$alpha < -3), ]
########################### Brain behavior GLMMs ###################################################

# does beta predict the stimulus? control analysis:

beta_stim <- glm(isyes ~ -1 + beta,
                  data = behav, family=binomial(link='probit'))
summary(beta_stim)

# does beta predict reaction times?

beta_rt <- lmer(respt1 ~ beta + (1|ID),
                 data = behav)
summary(beta_stim)


# replace stimulus probability regressor with beta power per trial 
# (neural correlate of stimulus probability)

beta_cue = lmer(beta ~ cue+ (cue|ID), 
             data=behav, control=lmerControl(optimizer="bobyqa",
            optCtrl=list(maxfun=2e5)))

summary(beta_cue)

alpha_cue = lmer(alpha ~ cue+ (cue|ID), 
                data=behav, control=lmerControl(optimizer="bobyqa",
                                                optCtrl=list(maxfun=2e5)))

summary(alpha_cue)

alpha_cue = lmer(beta ~ prevresp + (prevresp|ID), 
                 data=behav, control=lmerControl(optimizer="bobyqa",
                                                 optCtrl=list(maxfun=2e5)))

summary(alpha_cue)
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
beta_prev_glm <- glmer(sayyes ~ isyes + alpha + cue + prevresp + 
                    alpha*isyes + cue*isyes +
                     (isyes+cue + prevresp|ID),
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
beta_int_glm <- glmer(sayyes ~ isyes*beta+ isyes*cue + beta*prevresp + cue*prevresp +
                        (isyes+cue + prevresp| ID),
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
filename = paste("beta_int_glm_", expecon, ".rda", sep="")
cue_model_path = file.path("data", "behav", "mixed_models", "brain_behav", filename)
saveRDS(beta_int_glm, cue_model_path)
beta_int_glm <- readRDS(cue_model_path)
################################ fit confidence with beta power ####################################

behav$sayyes = as.factor(behav$sayyes)

conf_full_model_2 = glmer(conf ~ isyes*sayyes*cue + prevresp + prevconf +
                                 (isyes + sayyes + prevresp + prevconf + cue|ID), data=behav, 
                               family=binomial(link='probit'),
                               control=glmerControl(optimizer="bobyqa",
                                                    optCtrl=list(maxfun=2e5)))

check_collinearity(conf_full_model_1)
check_convergence(conf_full_model_1)

summary(conf_full_model_1)

# Post hoc tests for behavior interaction, 3way
contrast(emmeans(conf_full_model_1, "sayyes", by = c("isyes", "cue")))

# Post hoc tests for behavior interaction
contrast(emmeans(conf_full_model_1, "sayyes", by = c("cue")))

# plot estimates
p1<-plot_model(conf_full_model_1,  
               show.values = TRUE, 
               value.offset = .5,
               axis.labels = rev(c("stimulus", "yes response", 'high probability', 
                                   'previous yes response', 
                                   'previous high confidence', 'stimulus*yes', 
                                   'stimulus * high prob.',
                                   'yes response * high prob.',
                                   'stimulus * yes response * high prob.')),
               auto.label = FALSE)
p1

# now replace the cue with beta power

conf_full_model_beta_2 = glmer(conf ~ isyes*sayyes*beta + prevresp + prevconf +
                            (isyes + sayyes + prevresp + prevconf|ID), data=behav, 
                          family=binomial(link='probit'),
                          control=glmerControl(optimizer="bobyqa",
                                               optCtrl=list(maxfun=2e5)))

check_collinearity(conf_full_model_beta_1)
check_convergence(conf_full_model_beta_1)

summary(conf_full_model_beta_1)

# save models as tables for manuscript
#https://modelsummary.com/articles/modelsummary.html

filename = paste("conf_beta_expecon", expecon, ".html", sep="_")
output_file_path_beta <- file.path("figs", "manuscript_figures", "Tables", filename)

models = list("probability" = conf_full_model_1, "beta power" = conf_full_model_beta_1)

modelsummary::modelsummary(models, estimate  = "{estimate} [{conf.low}, {conf.high}], {stars}", 
                           statistic = NULL,  output = output_file_path_beta)

# plot estimates
p1<-plot_model(conf_full_model_beta_1,  
               show.values = TRUE, 
               value.offset = .5,
               axis.labels = rev(c("stimulus", "yes response", 'beta', 
                                   'previous yes response', 
                                   'previous high confidence', 'stimulus*yes', 
                                   'stimulus*beta.',
                                   'yes response* beta.',
                                   'stimulus*yes response* beta.')),
               auto.label = FALSE)
p1

# change the order of predictors (x and label)
plot_model(conf_full_model_1, type='pred',terms = c("sayyes", 'cue'), show.legend = TRUE,
                     colors = c("#357db8ff", "#e31919ff"),  title=" ")

# change the order of predictors (x and label)
plot_model(conf_full_model_beta_1, type='pred',terms = c("sayyes", 'beta'), show.legend = TRUE,
           colors = c("#357db8ff", "#e31919ff"),  title=" ")

# stable env
beta_values = c(0.83, -1.13)

# volatile env
beta_values = c(0.88, -1.08)

# Post hoc tests for behavior interaction, 3way
contrast(emmeans(conf_full_model_2, "cue", by = "sayyes", at = list(beta=beta_values)))

# Post hoc tests for behavior interaction
contrast(emmeans(conf_full_model_beta_2, "beta", by = "sayyes", at = list(beta=beta_values)))

filename = paste("conf_beta_expecon", expecon, ".html", sep="_")
output_file_path_beta <- file.path("figs", "manuscript_figures", "Tables", filename)

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
anova(beta_prev_glm, beta_int_glm)

# difference in AIC and BIC
diff_aic_1 = AIC(beta_prev_glm) - AIC(beta_base_glm)
diff_bic_1 = BIC(beta_prev_glm) - BIC(beta_base_glm)
print(diff_aic_1)
print(diff_bic_1)

diff_aic_2 = AIC(beta_int_glm) - AIC(beta_prev_glm)
diff_bic_2 = BIC(beta_int_glm) - BIC(beta_prev_glm)

print(diff_aic_2)
print(diff_bic_2)

# save table to html
filename = paste("beta_expecon", expecon, ".docx", sep="_")
output_file_path_beta <- file.path("figs", "manuscript_figures", "Tables", filename)

models = list("base" = beta_base_glm, "add previous response" = beta_prev_glm,
              "interaction" = beta_int_glm)

modelsummary::modelsummary(models, estimate  = "{estimate} [{conf.low}, {conf.high}], {stars}", 
                           statistic = NULL,  output = output_file_path_beta)

filename_conf = paste("conf_expecon", expecon, ".docx", sep="_")
output_file_path_conf<- file.path("figs", "manuscript_figures", "Tables", filename_conf)


models = list("prob" = conf_full_model_1, "beta" = conf_full_model_beta_1)

modelsummary::modelsummary(models, estimate  = "{estimate} [{conf.low}, {conf.high}], {stars}", 
                           statistic = NULL,  output = output_file_path_conf)
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