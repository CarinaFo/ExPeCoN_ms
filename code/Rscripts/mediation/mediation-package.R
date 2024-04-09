#####################################ExPeCoN study#################################################
# causal mediation analysis brain-behavior

# author: Carina Forster
# email: forster@mpg.cbs.de

# libraries
library(lme4) # mixed models
library(lmerTest)  # for p-values of lmer models
library(mediation)
library(dplyr)
library(parallel) 
library(performance)

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
behav$ID = as.factor(behav$ID)


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
# detection and/or confidence response?

############################ stable env.###################################################

# fit mediator: does stimulus probability predict prestimulus power? 
# (does the IV predict the mediator?)

# without p-values, model for mediation function
med.model_beta <- lme4::lmer(beta ~  isyes + cue + prevresp*cue + isyes*cue+ 
                               (cue + isyes + prevresp|ID), 
                             data = behav)
summary(med.model_beta)

# fit outcome model: do the mediator (beta) and the IV (stimulus probability cue) predict the
# detection response? included stimulus and previous choice at a given trial as covariates
# and interaction between prev. response and cue

out.model_beta <- glmer(sayyes ~ isyes + cue + beta + prevresp*cue + isyes*cue+
                     (cue+isyes+prevresp|ID),
                   data = behav,
                   control=glmerControl(optimizer="bobyqa",
                                        optCtrl=list(maxfun=2e5)),
                   family=binomial(link='probit'))

check_collinearity(out.model_beta)
summary(out.model_beta)

# now fit mediation model
# moderated mediation for stable env.

# beta
mediation_cue_beta_prevno <- mediate(med.model_beta, out.model_beta, treat='cue', 
                                     mediator='beta', 
                                     covariates=list(prevresp=0))

mediation_cue_beta_prevyes <- mediate(med.model_beta, out.model_beta, treat='cue', 
                                      mediator='beta', 
                                      covariates=list(prevresp=1))

summary(mediation_cue_beta_prevno)
summary(mediation_cue_beta_prevyes)

# save mediation output: probability cue
setwd("E:/expecon_ms")
# probability model beta power
filename = paste('medglm_cue_beta_', expecon, '.rds', sep="")
model_path = file.path("data", "behav", "mediation", filename)
saveRDS(out.model_beta, model_path)
out.model_beta = readRDS(model_path)

# save table to html
filename = paste("mediationglm_expecon", expecon, ".html", sep="_")
output_file_path <- file.path("figs", "manuscript_figures", "Tables", filename)

sjPlot::tab_model(out.model_beta,
                           show.aic=TRUE, show.loglik=TRUE,
                  file = output_file_path)

####################################### volatile env.##############################################

# without p-values, model for mediation function
med.model_beta <- lme4::lmer(beta ~  isyes + cue + isyes*cue+   (cue+prevresp+isyes|ID), 
                             data = behav,
                             control=lmerControl(optimizer="bobyqa",
                                                  optCtrl=list(maxfun=2e5))) # significant 
summary(med.model_beta)


# fit outcome model: do the mediator (beta) and the IV (stimulus probability cue) predict the
# detection response? included stimulus and previous choice at a given trial as covariates,
# but no interaction between prev. resp and cue
out.model_beta <- glmer(sayyes ~ isyes + cue + beta + prevresp +isyes*cue+
                          (cue+isyes+prevresp|ID),
                        data = behav,
                        control=glmerControl(optimizer="bobyqa",
                                             optCtrl=list(maxfun=2e5)),
                        family=binomial(link='probit'))

summary(out.model_beta)


mediation_cue_beta <- mediate(med.model_beta, out.model_beta, treat='cue', 
                                      mediator='beta')

summary(mediation_cue_beta)

# save mediation output: probability cue
setwd("E:/expecon_ms")
# probability model beta power
filename = paste('medglm_cue_beta_', expecon, '.rds', sep="")
model_path = file.path("data", "behav", "mediation", filename)
saveRDS(out.model_beta, model_path)
out.model_beta = readRDS(model_path)

filename = paste("mediationglm_expecon", expecon, ".html", sep="_")
output_file_path <- file.path("figs", "manuscript_figures", "Tables", filename)

# save table to html
sjPlot::tab_model(out.model_beta,
                  show.aic=TRUE, show.loglik=TRUE,
                  file = output_file_path)

# save mediation output: probability cue
setwd("E:/expecon_ms")
# probability model beta power
filename = paste('med_cue_beta_prevyes_', expecon, '.rds', sep="")
model_path = file.path("data", "behav", "mediation", filename)
saveRDS(mediation_cue_beta_prevyes, model_path)
mediation_cue_beta_yes = readRDS(model_path)

filename = paste('med_cue_beta_prevno_', expecon, '.rds', sep="")
model_path = file.path("data", "behav", "mediation", filename)
saveRDS(mediation_cue_beta_prevno, model_path)
mediation_cue_beta_no = readRDS(model_path)

filename = paste('med_cue_beta_', expecon, '.rds', sep="")
model_path = file.path("data", "behav", "mediation", filename)
saveRDS(mediation_cue_beta, model_path)
mediation_cue_beta = readRDS(model_path)

# save results in a table
extract_mediation_summary(mediation_cue_beta)

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