# mixed-levelm logistic regression mediation using the 'lavaan' package
# Anne Urai, 2021
# https://github.com/anne-urai/2022_Urai_choicehistory_MEG/blob/zenodo/mediation_lavaan.R
# adapted by Carina Forster, 2023

# https://ademos.people.uic.edu/Chapter15.html
# https://lavaan.ugent.be/tutorial/mediation.html
# https://nmmichalak.github.io/nicholas_michalak/blog_entries/2018/nrg01/nrg01.html

library("lavaan")
library("lavaanPlot")
library("semPlot")
set.seed(2021)

# Set the font family and size
par(family = "Arial", cex = 1.2)

# which data set?
expecon <- 2

####################################################################################################

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

# where to store lavaan results
datapath = file.path("D:", "expecon_ms", "analysis_code", "behav", "R", "mediation")
###################################################################################################

# make factors:
behav$ID = as.factor(behav$ID)
behav$isyes = as.factor(behav$isyes)
behav$cue = as.factor(behav$cue)
behav$prevresp = as.factor(behav$prevresp)
behav$previsyes = as.factor(behav$previsyes)
behav$prevconf = as.factor(behav$prevconf)
behav$correct = as.factor(behav$correct)
behav$prevcue = as.factor(behav$prevcue)

# drop nans
mydata <- subset(behav, select=c(ID, isyes, sayyes, cue,conf,
                                  prevresp, prevcorrect, prevconf, pre_alpha, 
                                 pre_beta))
mydata <- na.omit(mydata)

## potentially take only previous correct or previous error trials (from Anne Urai, 2021)
prevfeedback <- 'alldata'

if(prevfeedback == 'correct') {
  print('keeping only previous correct trials')
  mydata <- mydata[(mydata$prevcorrect == 1),]
} else if(prevfeedback == 'error') {
  print('keeping only previous error trials')
  mydata <- mydata[(mydata$prevcorrect == 0),]  
} else {
  print('keeping all trials')
}

# ============================= #

multipleMediation <- '
                      # a path in diagram (x on the mediator)
                      # first, define the regression equations
                      pre_alpha ~ a1 * prevresp # + s1* prevresp
                      pre_beta ~ a2 * prevresp # *prevresp
                      
                      # b path: 
                      sayyes ~  b1 * pre_alpha + b2 * pre_beta +
                                c * prevresp + d * isyes
                      
                      # b1 * alpha_close_stimonset +
                      #sayyes ~ b2 * beta_close_stimonset + c * cue
                      
                      # c path:
                      #sayyes ~ c * cue*isyes
                      
                      # define the effects
                      indirect_alpha := a1 * b1
                      indirect_beta := a2 * b2
              
                      #ab := a*b
                      #total := c + ab"
                      direct    := c
                      total     := c  +  (a2 * b2) # + (a1 * b1)
                      
                      # specify covariance between the two mediators
                      # https://paolotoffanin.wordpress.com/2017/05/06/multiple-mediator-analysis-with-lavaan/comment-page-1/
                      pre_alpha ~~ pre_beta
                      '

singleMediation <- '
                      # first, define the regression equations
                      med ~ prevresp:cue
                      pre_beta ~ a * med
                      sayyes ~ b * pre_beta + c * med + d*isyes

                      # define the effects
                      indirect := a * b
                      direct    := c
                      total     := c + (a * b)
                      '

# ============================= #
# loop over subjects
mediation_results  <- data.frame()

for ( subj in unique(c(mydata$ID)) ) {
  tmpdata <- mydata[(mydata$ID == subj),]
  fit <- sem(model = singleMediation, 
             data = tmpdata, 
             ordered=c("sayyes"), estimator="WLSMV")
  
  param_estimates <- parameterEstimates(fit, standardized = TRUE)
  
  # append to dataframe
  summ2 = as.data.frame(param_estimates)
  summ2$subj_idx <- subj
  mediation_results <- rbind(mediation_results, summ2) # append
  write.csv(mediation_results, file.path(datapath, 'lavaan_mediation_multiple_alldata_expecon2.csv')) # write at each iteration
  print(subj)
  
}

lavaanPlot(model = fit, node_options = list(shape = "box", fontname = "Helvetica"), 
           edge_options = list(color = "grey"), coefs = TRUE, stars='regress')

# # ============================= #                    
# # alternative: properly specify the mixed effects structure!
# # NOPE, not implemented for ordered data (logistic regression). stick to single-subject approach for now.