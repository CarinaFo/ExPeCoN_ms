#####################################ExPeCoN study#################################################
# mediation analysis using the mediation package

# author: Carina Forster
# email: forster@mpg.cbs.de

# libraries
library(lme4) # mixed models
library(lmerTest) 
library(mediation)
library(dplyr)

# Check the version of a specific package (e.g., "ggplot2")
package_version <- packageVersion("mediation")

# skip scientific notation
options(scipen=999)

# which dataset to analyze (1 => mini-block, 2 => trial-by-trial design)
expecon <- 1

####################################brain behav#####################################################

if (expecon == 1) {
  
  # expecon 1
  setwd("D:/expecon_ms")
  
  brain_behav_path <- file.path("data", "behav", "behav_df", "brain_behav_cleanpower.csv")
  
  behav = read.csv(brain_behav_path)
  
} else {
  
  # expecon 2 behavioral data
  setwd("D:/expecon_2")
  
  brain_behav_path <- file.path("behav", "brain_behav_cleanpower.csv")
  
  behav = read.csv(brain_behav_path)
  
}
################################prepare variables for linear mixed modelling #######################

# treatment variable can not be a factor for mediation analysis

# make factors for categorical variables:
behav$ID = as.factor(behav$ID) # subject ID

# Remove NaN trials for model comparision (models neeed to have same amount of data)
behav <- na.omit(behav) 

######################## mediation #############################################################

# https://towardsdatascience.com/doing-and-reporting-your-first-mediation-analysis-in-r-2fe423b92171

behav$beta <- behav$pre_beta
behav$alpha <- behav$pre_alpha

behav$cue <- ifelse(behav$cue == 0.25, 0, 1) 

# Create a new variable "highlowbeta
# based on whether beta power is above or below the mean for each participant
behav <- behav %>%
  group_by(ID) %>%
  mutate(meanBeta = mean(beta),
         highlowbeta = ifelse(beta > meanBeta, 1, 0))

# fit mediator
# with p-value
mediatorbeta  <- lmer(alpha ~ -1 + cue + (cue|ID), data = behav)
summary(mediatorbeta)

med.model  <- lme4::lmer(alpha ~ -1 + cue + (cue|ID), data = behav)
summary(med.model)

med.model  <- lme4::lmer(alpha ~ -1 + prevresp + (1|ID), data = behav)
summary(med.model)

# fit outcome model
out.model <- glmer(sayyes ~ isyes + cue + alpha + isyes*cue+
                  (cue+isyes+cue*isyes|ID),
                data = behav,
                control=glmerControl(optimizer="bobyqa",
                                     optCtrl=list(maxfun=2e5)),
                family=binomial(link='probit'))

summary(out.model)

# for previous response instead of cue
out.model <- glmer(sayyes ~ isyes + prevresp + beta + isyes*prevresp+
                     (prevresp+isyes|ID),
                   data = behav,
                   control=glmerControl(optimizer="bobyqa",
                                        optCtrl=list(maxfun=2e5)),
                   family=binomial(link='probit'))

summary(out.model)

# now fit mediation
results_beta_expecon1 <- mediate(med.model, out.model, treat='cue', mediator='beta')

results_beta_expecon2 <- mediate(med.model, out.model, treat='cue', mediator='beta')
summary(results_beta_expecon2)

# alpha
results_alpha_expecon1 <- mediate(med.model, out.model, treat='cue', mediator='alpha')
summary(results_alpha_expecon1)

results_alpha_expecon2 <- mediate(med.model, out.model, treat='cue', mediator='alpha')
summary(results_alpha_expecon2)

# prev. response
results_prev_expecon1 <- mediate(med.model, out.model, treat='prevresp', mediator='beta')
summary(results_prev_expecon1)

results_prev_expecon2 <- mediate(med.model, out.model, treat='prevresp', mediator='beta')
summary(results_prev_expecon2)

extract_mediation_summary(results_beta_expecon1)

# save mediation output
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
# moderated mediation (unsed at the moment)
# https://ademos.people.uic.edu/Chapter15.html
# include previous response as covariate (covariate has to be in mediation and outcome model)
results_beta_expecon1_moderated <- mediate(med.model, out.model, treat='cue', mediator='beta', 
                                 covariates = list(prevresp=1))

# test wether moderation effect is sign., doesn"t work, no idea why
test.modmed(results_beta_expecon1_moderated, covariates.1 = list(prevresp = 1),   
            covariates.2 = list(prevresp = 0), sims = 1000)