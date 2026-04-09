# Created on Mon Mar 3 11:03:05 2026

# @author: herttaleinonen

# ============================================================
# Nonlinear mixed-effects model for visibility
# Data: long.csv
# Tasks: vt1-vt5
# Outcome: accuracy (proportion correct)
# Predictors: eccentricity (deg), speed (deg/s)
# Random effect: participant
# ============================================================

# 0) Packages
install.packages(c("readr", "dplyr", "ggplot2", "nlme"))

library(readr)
library(dplyr)
library(ggplot2)
library(nlme)

# 1) Load data
dat <- read_csv("data/long.csv", show_col_types = FALSE)

# 2) Keep visibility tasks and define speed
vt <- dat %>%
  filter(task %in% paste0("vt", 1:5)) %>%
  mutate(
    participant = factor(participant),
    speed = case_when(
      task == "vt1" ~ 0,
      task == "vt2" ~ 3,
      task == "vt3" ~ 6,
      task == "vt4" ~ 8,
      task == "vt5" ~ 11,
      TRUE ~ NA_real_
    ),
    correct = as.numeric(correct),
    ecc_deg = as.numeric(ecc_deg)
  ) %>%
  filter(!is.na(speed), !is.na(correct), !is.na(ecc_deg))

# 3) Aggregate to participant-level proportions
#    (accuracy per participant × speed × eccentricity)
vt_participant <- vt %>%
  group_by(participant, speed, ecc_deg) %>%
  summarise(
    acc = mean(correct, na.rm = TRUE),
    n_trials = n(),
    .groups = "drop"
  )

print(vt_participant)

# ------------------------------------------------------------
# 4) Nonlinear mixed model
#
# Logistic visibility function:
#   acc = 1 / (1 + exp(-(a + b*ecc_deg + c*speed + d*ecc_deg*speed)))
#
# Interpretation:
#   a = baseline visibility
#   b = eccentricity effect
#   c = speed effect
#   d = speed × eccentricity interaction
#
# Random effect:
#   participant-specific intercept on parameter a
# ------------------------------------------------------------

# Starting values matter in nlme; these are reasonable guesses
start_vals <- c(
  a = 4.0,    # baseline
  b = -0.20,  # accuracy falls with eccentricity
  c = -0.05,  # small speed effect
  d = -0.01   # interaction: speed worsens eccentricity effect
)

vis_nlme <- nlme(
  acc ~ 1 / (1 + exp(-(a + b * ecc_deg + c * speed + d * ecc_deg * speed))),
  data = vt_participant,
  fixed = a + b + c + d ~ 1,
  random = a ~ 1 | participant,
  start = start_vals,
  na.action = na.omit,
  control = nlmeControl(
    maxIter = 200,
    pnlsMaxIter = 50,
    msMaxIter = 200
  )
)

cat("\n====================================================\n")
cat("Nonlinear mixed-effects model for visibility\n")
cat("====================================================\n\n")

print(summary(vis_nlme))

# 5) Predicted values for observed data
vt_participant$pred <- predict(vis_nlme)

# 6) Create smooth prediction grid for plotting
pred_grid <- expand.grid(
  ecc_deg = seq(min(vt_participant$ecc_deg, na.rm = TRUE),
                max(vt_participant$ecc_deg, na.rm = TRUE),
                length.out = 200),
  speed = c(0, 3, 6, 8, 11)
)

pred_grid$pred <- predict(vis_nlme, newdata = pred_grid, level = 0)

# 7) Group summary for observed means ± SEM
vt_summary <- vt_participant %>%
  group_by(speed, ecc_deg) %>%
  summarise(
    mean_acc = mean(acc, na.rm = TRUE),
    sem = sd(acc, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  ) %>%
  mutate(speed = factor(speed, levels = c(0, 3, 6, 8, 11)))

pred_grid <- pred_grid %>%
  mutate(speed = factor(speed, levels = c(0, 3, 6, 8, 11)))

# 8) Plot observed means + SEM + nonlinear fitted curves
p_vis_nlme <- ggplot() +
  geom_errorbar(
    data = vt_summary,
    aes(x = ecc_deg, y = mean_acc, ymin = mean_acc - sem, ymax = mean_acc + sem, color = speed),
    width = 0.2
  ) +
  geom_point(
    data = vt_summary,
    aes(x = ecc_deg, y = mean_acc, color = speed),
    size = 2.5
  ) +
  geom_line(
    data = pred_grid,
    aes(x = ecc_deg, y = pred, color = speed, group = speed),
    linewidth = 1.2
  ) +
  labs(
    x = "Eccentricity (deg)",
    y = "Accuracy",
    color = "Speed (deg/s)",
    title = "Nonlinear mixed-effects model of visibility"
  ) +
  scale_y_continuous(
    breaks = seq(0.5, 1.0, 0.1),
    limits = c(0.5, 1.0)
  ) +
  theme_classic() +
  theme(
    text = element_text(size = 14),
    panel.grid.major.y = element_line(color = "grey85", linewidth = 0.6),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white")
  )

print(p_vis_nlme)

# 9) Save plot
ggsave(
  filename = "visibility_nonlinear_mixed_model.png",
  plot = p_vis_nlme,
  width = 7,
  height = 5,
  dpi = 300
)

# 10) Extract fixed effects 
cat("\n--- Fixed effects estimates ---\n")
print(fixed.effects(vis_nlme))

cat("\n--- Random effects ---\n")
print(ranef(vis_nlme))

# 11) Compare to reduced model without interaction
vis_nlme_no_int <- nlme(
  acc ~ 1 / (1 + exp(-(a + b * ecc_deg + c * speed))),
  data = vt_participant,
  fixed = a + b + c ~ 1,
  random = a ~ 1 | participant,
  start = c(a = 4.0, b = -0.20, c = -0.05),
  na.action = na.omit,
  control = nlmeControl(
    maxIter = 200,
    pnlsMaxIter = 50,
    msMaxIter = 200
  )
)

cat("\n--- Model comparison: with vs without interaction ---\n")
print(anova(vis_nlme_no_int, vis_nlme))
