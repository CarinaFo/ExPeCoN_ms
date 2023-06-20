
#######################Currently unused#############################################################


cue_prevacc_int_model = glmer(sayyes ~ isyes*cue+prevacc*cue+ (isyes*cue+prevacc*cue|ID), 
                              data=behav, family=binomial(link='probit'),
                              control=glmerControl(optimizer="bobyqa",
                                                   optCtrl=list(maxfun=2e5)))

saveRDS(cue_previsyes_int_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_previsyes_int_model.rda")
cue_previsyes_int_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_previsyes_int_model.rda")

summary(cue_previsyes_int_model)

AIC(cue_model)
AIC(cue_prev_model)
AIC(cue_prev_int_model)
AIC(cue_previsyes_int_model)

#################################Confidence#########################################################

# stronger prevchoice effect in confident trials

cue_prev_int_conf_model = glmer(sayyes ~ isyes*cue + prevsayyes*cue + (isyes*cue+prevsayyes*cue|ID), 
                                data=confident_trials_only, family=binomial(link='probit'),
                                control=glmerControl(optimizer="bobyqa",
                                                     optCtrl=list(maxfun=2e5)))

saveRDS(cue_prev_int_conf_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_conf_model.rda")
cue_prev_int_conf_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_conf_model.rda")

summary(cue_prev_int_conf_model)

AIC(cue_prev_int_model)
AIC(cue_prev_int_conf_model)

#######################################Prev confidence as regressor#################################

# include confidence in previous trial as regressor

cue_prev_int_confreg_model = glmer(sayyes ~ isyes*cue + prevsayyes*cue + prevconf + 
                                     (isyes*cue+prevsayyes*cue+prevconf|ID), 
                                   data=behav, family=binomial(link='probit'),
                                   control=glmerControl(optimizer="bobyqa",
                                                        optCtrl=list(maxfun=2e5)))

saveRDS(cue_prev_int_confreg_model, "D:\\expecon_ms\\analysis_code\\
        behav\\linear_mixed_models\\cue_prev_int_confreg_model.rda")

cue_prev_int_confreg_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prev_int_confreg_model.rda")

summary(cue_prev_int_confreg_model)

# prev conf is not a sign. regressor

cue_prevconf_int = glmer(sayyes ~ isyes*cue + prevconf*cue + (isyes*cue+prevconf*cue|ID), 
                         data=behav, family=binomial(link='probit'),
                         control=glmerControl(optimizer="bobyqa",
                                              optCtrl=list(maxfun=2e5)))

saveRDS(cue_prevconf_int, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prevconf_int.rda")
cue_prevconf_int <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prevconf_int.rda")

summary(cue_prevconf_int)

cue_prevacc_int_model = glmer(sayyes ~ isyes*cue+prevacc*cue+ (isyes*cue+prevacc*cue|ID), 
                              data=behav, family=binomial(link='probit'),
                              control=glmerControl(optimizer="bobyqa",
                                                   optCtrl=list(maxfun=2e5)))

saveRDS(cue_prevacc_int_model, "D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prevacc_int_model.rda")
cue_prevacc_int_model <- readRDS("D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\cue_prevacc_int_model.rda")

summary(cue_prevacc_int_model)

# cue is still significant, interaction between cue and previous confidence rating not

###########################################Metacognition############################################

# does the detection response predict confidence?

resp_conf = glmer(conf ~ sayyes + (sayyes|ID), data=behav, family=binomial(link='probit'))

saveRDS(resp_conf, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\resp_conf.rda")

summary(resp_conf)

# higher confidence in no responses

# does accuracy predict confidence?

conf_acc = glmer(conf ~ correct + (correct|ID), data=behav, family=binomial(link='probit'))

saveRDS(conf_acc, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\conf_acc.rda")

summary(conf_acc)

# higher confidence in correct trials

conf_con = glmer(conf ~ congruency + (congruency|ID), data=behav, family=binomial(link='probit'))

saveRDS(conf_con, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\conf_con.rda")

summary(conf_con)

#higher confidence in congruent trials

conf_surprise = glmer(conf ~ surprise + (surprise|ID), data=behav, 
                      family=binomial(link='probit'))

saveRDS(conf_surprise, file="D:\\expecon_ms\\analysis_code\\behav\\linear_mixed_models\\conf_surprise.rda")

summary(conf_surprise)

# higher surprise, less confident

#################################Plot model parameters##############################################

# Set the font family and size

par(family = "Arial", cex = 1.2)

est = sjPlot::plot_model(cue_model, type='est')
int = sjPlot::plot_model(cue_model, type='int')

# Extract coefficients

coeffs <- as.data.frame(summary(cue_lag_int_model)$coefficients)

# Plot coefficients
p1 <- ggplot(coeffs, aes(x = rownames(coeffs), y = Estimate)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(size = 3.5, color = "#2E8B57") +
  geom_errorbar(aes(ymin = Estimate - 1.96 * sdt, ymax = Estimate + 1.96 * sdt),  width = 0.3, size=1, color = "#2E8B57") +
  coord_flip() +
  labs(x = "", y = "Effect Size") +
  theme_classic() +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.y = element_text(size = 12),
        axis.line.x = element_blank(),
        panel.border = element_blank())
p1

# Create an effects object for the interaction
eff <- effect("cue0.75:lagsayyes1", cue_lag_int_model)

# Plot the predicted probabilities
plot(eff, type="response", rug=FALSE)

ggsave('D:\\expecon_ms\\figs\\behavior\\regression\\prev_sdt_estimates.svg', p1, device='svg', width=10, height=10)
ggsave('D:\\expecon_ms\\figs\\behavior\\regression\\prev_sdt_int.svg', int[[1]], device='svg')
ggsave('D:\\expecon_ms\\figs\\behavior\\regression\\prev_sdt_int_2.svg', int[[2]], device='svg')


##############################brain behavior modelling##################################

cue_theta = lmer(theta_scale_log ~ cue + (cue|ID), data=behav) # n.s.

cue_alpha = lmer(alpha_scale_log ~ cue + (cue|ID), data=behav, control=lmerControl(optimizer="bobyqa",
                                                                                   optCtrl=list(maxfun=2e5)))
# n.s.

cue_beta = lmer(lowbeta_scale_log ~ cue + (cue|ID), data=behav) # p = 0.02
cue_beta_gamma = lmer(beta_gamma_scale_log ~ cue + (cue|ID), data=behav) # n.s.

# previous choice predicts alpha, beta and beta_gamma power

prevchoice_beta = lmer(lowbeta_scale_log ~ prevsayyes + (prevsayyes|ID), data=behav)
prevchoice_alpha = lmer(alpha_scale_log ~ prevsayyes + (prevsayyes|ID), data=behav)
prevchoice_theta = lmer(theta_scale_log ~ prevsayyes + (prevsayyes|ID), data=behav)
prevchoice_beta_gamma = lmer(beta_gamma_scale_log ~ prevsayyes + (prevsayyes|ID), data=behav)

# SDT model with beta power as covariate

sdt_model_beta = glmer(sayyes ~ isyes*lowbeta_scale_log + prevsayyes*lowbeta_scale_log + 
                         (isyes*lowbeta_scale_log + prevsayyes*lowbeta_scale_log|ID), data=behav, 
                       family=binomial(link='probit'),
                       control=glmerControl(optimizer="bobyqa",
                                            optCtrl=list(maxfun=2e5)))

sdt_model_beta_gamma = glmer(sayyes ~ isyes*beta_gamma_scale_log + prevsayyes*beta_gamma_scale_log + 
                               (isyes*beta_gamma_scale_log + prevsayyes*beta_gamma_scale_log|ID), data=behav, 
                             family=binomial(link='probit'),
                             control=glmerControl(optimizer="bobyqa",
                                                  optCtrl=list(maxfun=2e5)))



