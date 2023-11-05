#####################################ExPeCoN study#################################################
# causal mediation analysis brain-behavior

# author: Carina Forster
# email: forster@mpg.cbs.de

# libraries
library(lme4) # mixed models
library(lmerTest)  # for p-values of lmer models
library(mediation)
library(dplyr)
library(bayestestR)
library(brms)
library(rstanarm)
library(parallel) 

# Check the version of a specific package (e.g., "ggplot2")
package_version <- packageVersion("mediation")

# skip scientific notation
options(scipen=999)

# which dataset to analyze (1 => block design, 2 => trial-by-trial design)
expecon <- 2

####################################brain behav#####################################################
setwd("E:/expecon_ms")

filename = paste("brain_behav_cleaned_", expecon, ".csv", sep="")
brain_behav_path <- file.path("data", "behav", filename)

behav = read.csv(brain_behav_path)
################################prepare variables for linear mixed modelling #######################

# treatment variable can not be a factor for mediation analysis
# make factors for categorical variables:
behav$ID = as.factor(behav$ID) # subject ID

# Remove NaN trials
behav <- na.omit(behav) 

######################## mediation #############################################################

# https://towardsdatascience.com/doing-and-reporting-your-first-mediation-analysis-in-r-2fe423b92171

# rename
behav$beta <- behav$pre_beta
behav$alpha <- behav$pre_alpha

# dummy recode
behav$cue <- ifelse(behav$cue == 0.25, 0, 1) 

# Create a new variable "highlowbeta
# based on whether beta power is above or below the mean for each participant
behav_grouped <- behav %>%
  group_by(ID) %>%
  mutate(meanBeta = mean(beta),
         highlowbeta = ifelse(beta > meanBeta, 1, 0))

#### mediation model using mixed linear models and the mediation package
# research question: does beta power mediate the effect of stimulus probability on the
# detection response?

# fit mediator: does stimulus probability predict prestimulus power? 
# (does the IV predict the mediator?)
# with p-value: lmerTest
mediatoralpha  <- lmerTest::lmer(alpha ~ -1 + cue + (cue|ID), data = behav) # significant 
summary(mediatoralpha)
mediatorbeta  <- lmerTest::lmer(beta ~ -1 + cue + (cue|ID), data = behav) # significant 
summary(mediatorbeta)

# without p-values, model for mediation function
med.model_alpha  <- lme4::lmer(alpha ~ -1 + cue + (cue|ID), data = behav)
med.model_beta  <- lme4::lmer(beta ~ -1 + cue + (cue|ID), data = behav)

# mediation for previous response
med.model_prevresp_alpha  <- lme4::lmer(alpha ~ -1 + prevresp + (prevresp|ID), data = behav)
med.model_prevresp_beta  <- lme4::lmer(beta ~ -1 + prevresp + (prevresp|ID), data = behav)

# fit outcome model: do the mediator (beta) and the IV (stimulus probability cue) predict the
# detection response? included stimulus at a given trial as covariate
out.model_alpha <- glmer(sayyes ~ isyes + cue + alpha + isyes*cue+
                  (cue+isyes|ID),
                data = behav,
                control=glmerControl(optimizer="bobyqa",
                                     optCtrl=list(maxfun=2e5)),
                family=binomial(link='probit'))

summary(out.model_alpha)

out.model_beta <- glmer(sayyes ~ isyes + cue + beta + isyes*cue+
                     (cue+isyes|ID),
                   data = behav,
                   control=glmerControl(optimizer="bobyqa",
                                        optCtrl=list(maxfun=2e5)),
                   family=binomial(link='probit'))

summary(out.model_beta)

# now fit previous response
out.model_prevresp_alpha <- glmer(sayyes ~ isyes + prevresp + alpha + 
                           (prevresp+isyes|ID),
                         data = behav,
                         control=glmerControl(optimizer="bobyqa",
                                              optCtrl=list(maxfun=2e5)),
                         family=binomial(link='probit'))

summary(out.model_prevresp_alpha)

out.model_prevresp_beta <- glmer(sayyes ~ isyes + prevresp + beta +
                          (prevresp+isyes|ID),
                        data = behav,
                        control=glmerControl(optimizer="bobyqa",
                                             optCtrl=list(maxfun=2e5)),
                        family=binomial(link='probit'))

summary(out.model_prevresp_beta)

# now fit mediation model for stimulus probability and previous choice for alpha and beta power
# for alpha power
mediation_cue_alpha <- mediate(med.model_alpha, out.model_alpha, treat='cue', mediator='alpha')
mediation_prevresp_alpha <- mediate(med.model_prevresp_alpha, out.model_prevresp_alpha, treat='prevresp', mediator='alpha')
summary(mediation_cue_alpha)
summary(mediation_prevresp_alpha)

# for beta power
mediation_cue_beta <- mediate(med.model_beta, out.model_beta, treat='cue', mediator='beta')
mediation_prevresp_beta <- mediate(med.model_prevresp_beta, out.model_prevresp_beta, treat='prevresp', mediator='beta')
summary(mediation_cue_beta)
summary(mediation_prevresp_beta)

# save mediation output: probability cue
setwd("E:/expecon_ms/data/behav/mediation/")
# probability model alpha power
filenname = paste('med_cue_alpha_', expecon, '.rds', sep="")
saveRDS(mediation_cue_alpha, filename)
# probability model beta power
filenname = paste('med_cue_beta_', expecon, '.rds', sep="")
saveRDS(mediation_cue_beta, filename)

# save previous choice mediation models
filenname = paste('med_prevresp_alpha_', expecon, '.rds', sep="")
saveRDS(mediation_prevresp_alpha, filename)
filenname = paste('med_prevresp_beta_', expecon, '.rds', sep="")
saveRDS(mediation_prevresp_beta, filename)

# save results in a table
extract_mediation_summary(mediation_cue_alpha)
#Bayesian mediation model: mediation with brms######################################################
# https://easystats.github.io/bayestestR/articles/mediation.html

# how many cores do we have available on  this machine?
detectCores()

# Fit Bayesian mediation model in brms
# probability cue models
# alpha
med.model <- bf(alpha ~ -1 + cue + (cue|ID))
out.model <- bf(sayyes ~ isyes + cue + alpha + isyes*cue+
                  (cue+isyes|ID), family=bernoulli(link='probit'))

med_cue_alpha <- brm(med.model + out.model + set_rescor(FALSE), 
                     data = behav, refresh = 1, 
                     cores=10)

# beta
med.model <- bf(beta ~ -1 + cue + (cue|ID))
out.model <- bf(sayyes ~ isyes + cue + beta + isyes*cue+
                  (cue+isyes|ID), family=bernoulli(link='probit'))

med_cue_beta <- brm(med.model + out.model + set_rescor(FALSE), 
                    data = behav, refresh = 1, 
                    cores=10)

# previous choice model
# alpha
med.model <- bf(alpha ~ -1 + prevresp + (prevresp|ID))
out.model <- bf(sayyes ~ isyes + prevresp + alpha + isyes*cue+
                  (prevresp+isyes|ID), family=bernoulli(link='probit'))

med_prevresp_alpha <- brm(med.model + out.model + set_rescor(FALSE), 
                         data = behav, refresh = 1, 
                         cores=10)
# beta
med.model <- bf(beta ~ -1 + prevresp + (prevresp|ID))
out.model <- bf(sayyes ~ isyes + prevresp + beta + isyes*cue+
                  (prevresp+isyes|ID), family=bernoulli(link='probit'))

med_prevresp_beta <- brm(med.model + out.model + set_rescor(FALSE), 
                    data = behav, refresh = 1, 
                    cores=10)

# save model
filename = paste('med_cue_beta_', expecon, '.rds', sep="")
saveRDS(med_cue_beta, filename)
# alpha cue model
filename = paste('med_cue_alpha_', expecon, '.rds', sep="")
saveRDS(med_cue_alpha, filename)


# mediation for brms (to have evidence against mediation for alpha, 
# see Tilamns comment on manuscript draft)
bayes_med_output_beta_cue = mediation(med_cue_beta)
bayes_med_output_alpha_cue = mediation(med_cue_alpha)
################################ helper functions ##################################################
extract_mediation_summary <- function (x) { 
  
  clp <- 100 * x$conf.level
  isLinear.y <- ((class(x$model.y)[1] %in% c("lm", "rq")) || 
                   (inherits(x$model.y, "glm") && x$model.y$family$family == 
                      "gaussian" && x$model.y$family$link == "identity") || 
                   (inherits(x$model.y, "survreg") && x$model.y$dist == 
                      "gaussian"))
  
  printone <- !x$INT && isLinear.y
  
  if (printone) {
    
    smat <- c(x$d1, x$d1.ci, x$d1.p)
    smat <- rbind(smat, c(x$z0, x$z0.ci, x$z0.p))
    smat <- rbind(smat, c(x$tau.coef, x$tau.ci, x$tau.p))
    smat <- rbind(smat, c(x$n0, x$n0.ci, x$n0.p))
    
    rownames(smat) <- c("ACME", "ADE", "Total Effect", "Prop. Mediated")
    
  } else {
    smat <- c(x$d0, x$d0.ci, x$d0.p)
    smat <- rbind(smat, c(x$d1, x$d1.ci, x$d1.p))
    smat <- rbind(smat, c(x$z0, x$z0.ci, x$z0.p))
    smat <- rbind(smat, c(x$z1, x$z1.ci, x$z1.p))
    smat <- rbind(smat, c(x$tau.coef, x$tau.ci, x$tau.p))
    smat <- rbind(smat, c(x$n0, x$n0.ci, x$n0.p))
    smat <- rbind(smat, c(x$n1, x$n1.ci, x$n1.p))
    smat <- rbind(smat, c(x$d.avg, x$d.avg.ci, x$d.avg.p))
    smat <- rbind(smat, c(x$z.avg, x$z.avg.ci, x$z.avg.p))
    smat <- rbind(smat, c(x$n.avg, x$n.avg.ci, x$n.avg.p))
    
    rownames(smat) <- c("ACME (control)", "ACME (treated)", 
                        "ADE (control)", "ADE (treated)", "Total Effect", 
                        "Prop. Mediated (control)", "Prop. Mediated (treated)", 
                        "ACME (average)", "ADE (average)", "Prop. Mediated (average)")
    
  }
  
  colnames(smat) <- c("Estimate", paste(clp, "% CI Lower", sep = ""), 
                      paste(clp, "% CI Upper", sep = ""), "p-value")
  smat
  
}