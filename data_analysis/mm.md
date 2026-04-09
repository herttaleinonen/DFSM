# Created on Mon Feb 26 11:38:18 2026

# @author: herttaleinonen


install.packages(c("broom.mixed"))

# ============================================================
# Linear Mixed Models for Dynamic Search Tasks (DT1–DT5)
# Fixed: speed (linear numeric) * target_present
# Random: (1 | participant) [optionally (1 + speed | participant)]
# DVs: RT, Accuracy, Fixation count, Scanpath length, Dispersion, Distance from centre
# Output: model summaries, ANOVA (Type III), emmeans, plots
# ============================================================

# 0) Packages
install.packages(c(
  "readr","dplyr","tidyr","ggplot2","stringr",
  "lme4","lmerTest","emmeans","car","performance"
))
library(broom.mixed)
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(stringr)
library(lme4)
library(lmerTest)     # p-values for lmer
library(emmeans)
library(car)          # type-III Anova
library(performance)  # R2, ICC, etc.

# 1) Load long data
dat <- read_csv("data/long.csv", show_col_types = FALSE)

# 2) Keep DT tasks only + create factors
dt <- dat %>%
  filter(task %in% paste0("dt", 1:5)) %>%
  mutate(
    speed_num = case_when(
      task == "dt1" ~ 0,
      task == "dt2" ~ 3,
      task == "dt3" ~ 6,
      task == "dt4" ~ 8,
      task == "dt5" ~ 11,
      TRUE ~ NA_real_
    ),

    # factor version for plotting
    speed_fac = factor(speed_num, levels = c(0, 3, 6, 8, 11)),

    participant = factor(participant),

    target_present = factor(
      target_present,
      levels = c(0, 1),
      labels = c("absent", "present")
    )
  )

# 3) Aggregate to condition means 
dt_rt <- dt %>%
  filter(correct == 1) %>%
  group_by(participant, speed_num, speed_fac, target_present) %>%
  summarise(rt = mean(rt_s, na.rm = TRUE), .groups = "drop")

dt_acc <- dt %>%
  group_by(participant, speed_num, speed_fac, target_present) %>%
  summarise(acc = mean(correct, na.rm = TRUE), .groups = "drop")

dt_eye <- dt %>%
  group_by(participant, speed_num, speed_fac, target_present) %>%
  summarise(
    fix_count   = mean(fix_count, na.rm = TRUE),
    scanpath    = mean(fix_path_length_deg, na.rm = TRUE),
    dispersion  = mean(fix_dispersion_deg2, na.rm = TRUE),
    center_dist = mean(fix_center_dist_deg, na.rm = TRUE),
    n_trials    = sum(!is.na(fix_center_dist_deg)),
    .groups = "drop"
  ) %>%
  filter(n_trials > 0)

# Combined aggregated file
dt_all <- dt_rt %>%
  left_join(dt_acc, by = c("participant","speed_num","speed_fac","target_present")) %>%
  left_join(dt_eye, by = c("participant","speed_num","speed_fac","target_present"))

write_csv(dt_all, "data/dt_all_aggregated.csv")

# ------------------------------------------------------------
# 4) Mixed model helpers
# ------------------------------------------------------------

# Type III tests: set sum-to-zero contrasts
options(contrasts = c("contr.sum", "contr.poly"))

fit_lmm <- function(df, dv_name, random_slope_speed = FALSE) {
  rand <- if (random_slope_speed) "(1 + speed_num | participant)" else "(1 | participant)"
  f <- as.formula(paste0(dv_name, " ~ speed_num * target_present + ", rand))
  lmer(f, data = df, REML = FALSE)
}

report_lmm <- function(df, dv_name, random_slope_speed = FALSE) {
  cat("\n====================================================\n")
  cat("LMM for DV:", dv_name, "\n")
  cat("Random structure:", if (random_slope_speed) "(1 + speed_num | participant)" else "(1 | participant)", "\n")
  
  m <- fit_lmm(df, dv_name, random_slope_speed = random_slope_speed)
  
  cat("\n--- Model summary (fixed effects) ---\n")
  print(summary(m))
  
  cat("\n--- Type III ANOVA (Wald tests) ---\n")
  print(car::Anova(m, type = 3))
  
  cat("\n--- Performance (R2, ICC) ---\n")
  print(performance::r2(m))
  print(performance::icc(m))
  
  invisible(m)
}

# ------------------------------------------------------------
# 5) Fit models 
# ------------------------------------------------------------

m_rt    <- report_lmm(dt_rt,  "rt")
m_acc   <- report_lmm(dt_acc, "acc")
m_fix   <- report_lmm(dt_eye, "fix_count")
m_scan  <- report_lmm(dt_eye, "scanpath")
m_disp  <- report_lmm(dt_eye, "dispersion")
m_ctr   <- report_lmm(dt_eye, "center_dist")

# ------------------------------------------------------------
# 6) Post-hoc / simple effects with emmeans
# ------------------------------------------------------------

cat("\n===== RT: speed trend within target_present =====\n")
print(emtrends(m_rt, ~ target_present, var = "speed_num"))

cat("\n===== RT: target_present difference at each speed level (grid) =====\n")

# Evaluate target effect at the observed speeds (0..11)
rt_grid <- emmeans(m_rt, ~ target_present | speed_num,
                   at = list(speed_num = c(0,3,6,8,11)))
print(pairs(rt_grid, adjust = "holm"))

cat("\n===== Dispersion: speed trend within target_present =====\n")
print(emtrends(m_disp, ~ target_present, var = "speed_num"))

cat("\n===== Dispersion: target_present difference at each speed level =====\n")
disp_grid <- emmeans(m_disp, ~ target_present | speed_num,
                     at = list(speed_num = c(0,3,6,8,11)))
print(pairs(disp_grid, adjust = "holm"))

cat("\n===== Centre distance: speed trend within target_present =====\n")
print(emtrends(m_ctr, ~ target_present, var = "speed_num"))

cat("\n===== Centre distance: target_present difference at each speed level =====\n")
ctr_grid <- emmeans(m_ctr, ~ target_present | speed_num,
                    at = list(speed_num = c(0,3,6,8,11)))
print(pairs(ctr_grid, adjust = "holm"))

# ------------------------------------------------------------
# 7) Visualization: predicted lines from the LMM
# ------------------------------------------------------------

plot_lmm_pred <- function(model, df, dv_name, ylab) {
  newdat <- expand.grid(
    speed_num = c(0,3,6,8,11),
    target_present = levels(df$target_present)
  )
  newdat$pred <- predict(model, newdata = newdat, re.form = NA)

  ggplot(newdat, aes(x = speed_num, y = pred, color = target_present, group = target_present)) +
    geom_line() +
    geom_point() +
    labs(x = "Speed (deg/s)", y = ylab, color = "Target") +
    theme_minimal() +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      text = element_text(size = 14)
    )
}

p_rt   <- plot_lmm_pred(m_rt,   dt_rt,  "rt",         "Predicted RT (s) [correct trials]")
p_acc  <- plot_lmm_pred(m_acc,  dt_acc, "acc",        "Predicted Accuracy")
p_fix  <- plot_lmm_pred(m_fix,  dt_eye, "fix_count",  "Predicted fixation count")
p_scan <- plot_lmm_pred(m_scan, dt_eye, "scanpath", "Predicted scanpath length (deg)")
p_disp <- plot_lmm_pred(m_disp, dt_eye, "dispersion", "Predicted dispersion (deg²)")

print(p_rt)
print(p_acc)
print(p_fix)
print(p_scan)
print(p_disp)


plot_spaghetti_with_lmm <- function(df, dv_name, model, ylab,
                                    speed_var = "speed_num",
                                    group_var = "participant",
                                    cond_var  = "target_present",
                                    speeds = c(0,3,6,8,11)) {
  
  # ensure the vars exist
  stopifnot(all(c(dv_name, speed_var, group_var, cond_var) %in% names(df)))
  
  # Condition means (across participants) for plotting mean points
  means <- df %>%
    group_by(.data[[speed_var]], .data[[cond_var]]) %>%
    summarise(mean = mean(.data[[dv_name]], na.rm = TRUE), .groups = "drop") %>%
    rename(speed = .data[[speed_var]], cond = .data[[cond_var]])
  
  # Fixed-effect predictions from the LMM (no random effects)
  pred_grid <- expand.grid(
    speed_num = speeds,
    target_present = levels(df[[cond_var]])
  )
  # adapt if speed_var/cond_var names differ in df/model
  names(pred_grid)[names(pred_grid) == "speed_num"] <- speed_var
  names(pred_grid)[names(pred_grid) == "target_present"] <- cond_var
  
  pred_grid$pred <- predict(model, newdata = pred_grid, re.form = NA)
  
  # Plot
  ggplot(df, aes(x = .data[[speed_var]], y = .data[[dv_name]], color = .data[[cond_var]])) +
    
    # Individual participant lines (spaghetti)
    geom_line(aes(group = interaction(.data[[group_var]], .data[[cond_var]])),
              alpha = 0.20, linewidth = 0.6) +
    
    # Individual participant points
    geom_point(alpha = 0.20, size = 1.3) +
    
    # Condition means (bold points)
    geom_point(data = means, aes(x = speed, y = mean, color = cond),
               size = 2.6, alpha = 0.95) +
    
    # LMM fixed-effect prediction line (bold slope)
    geom_line(data = pred_grid,
              aes(x = .data[[speed_var]], y = pred, color = .data[[cond_var]],
                  group = .data[[cond_var]]),
              linewidth = 1.3, alpha = 0.95) +
    
    labs(x = "Speed (deg/s)", y = ylab, color = "Target") +
    theme_minimal() +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      text = element_text(size = 14)
    )
}

# RT
p_rt_spag <- plot_spaghetti_with_lmm(dt_rt, "rt", m_rt,
                                     ylab = "RT (s) [correct trials]")

# Accuracy
p_acc_spag <- plot_spaghetti_with_lmm(dt_acc, "acc", m_acc,
                                      ylab = "Accuracy (proportion)")

# Fix count
p_fix_spag <- plot_spaghetti_with_lmm(dt_eye, "fix_count", m_fix,
                                      ylab = "Fixation count")

# Scanpath
p_scan_spag <- plot_spaghetti_with_lmm(dt_eye, "scanpath", m_scan,
                                       ylab = "Scanpath length (deg)")

# Dispersion
p_disp_spag <- plot_spaghetti_with_lmm(
  dt_eye,
  "dispersion",
  m_disp,
  ylab = "Dispersion (deg²)"
)

# Distance from centre
p_ctr_spag <- plot_spaghetti_with_lmm(
  dt_eye,
  "center_dist",
  m_ctr,
  ylab = "Fixation distance from centre (deg)"
)

print(p_rt_spag)
print(p_acc_spag)
print(p_fix_spag)
print(p_scan_spag)
print(p_disp_spag)
print(p_ctr_spag)

