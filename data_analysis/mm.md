# Created on Mon Feb 26 11:38:18 2026
# @author: herttaleinonen

# ============================================================
# Packages
# ============================================================

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lme4)
library(lmerTest)
library(emmeans)
library(car)
library(performance)
library(broom.mixed)

# ============================================================
# Load data
# ============================================================

dat <- read_csv("data/long.csv", show_col_types = FALSE)

# ============================================================
# Speed mapping (exact values for models, CUSTOM LABELS)
# ============================================================

speed_map <- tibble(
  task      = paste0("dt", 1:5),
  speed_num = c(0.000, 2.703, 5.406, 8.109, 10.812),   # exact deg/s
  speed_lab = c("0", "3", "5.5", "8", "11")           # labels on x axis
)

speed_breaks <- speed_map$speed_num
speed_labels <- speed_map$speed_lab

# ============================================================
# DT tasks + factors
# ============================================================

dt <- dat %>%
  filter(task %in% paste0("dt", 1:5)) %>%
  left_join(speed_map, by = "task") %>%
  mutate(
    speed_fac = factor(speed_lab, levels = speed_labels),
    participant = factor(participant),
    target_present = factor(
      target_present,
      levels = c(0, 1),
      labels = c("absent", "present")
    )
  )

# ============================================================
# Aggregate data
# ============================================================

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

# ============================================================
# Mixed models
# ============================================================

options(contrasts = c("contr.sum", "contr.poly"))

fit_lmm <- function(df, dv) {
  lmer(
    as.formula(paste0(dv, " ~ speed_num * target_present + (1|participant)")),
    data = df,
    REML = FALSE
  )
}

m_rt   <- fit_lmm(dt_rt,  "rt")
m_acc  <- fit_lmm(dt_acc, "acc")
m_fix  <- fit_lmm(dt_eye, "fix_count")
m_scan <- fit_lmm(dt_eye, "scanpath")
m_disp <- fit_lmm(dt_eye, "dispersion")
m_ctr  <- fit_lmm(dt_eye, "center_dist")

# ============================================================
# Spaghetti + LMM + 95% CI plotting function
# ============================================================

plot_spaghetti_with_lmm <- function(df, dv_name, model, ylab,
                                    speed_var = "speed_num",
                                    group_var = "participant",
                                    cond_var  = "target_present",
                                    speed_breaks,
                                    speed_labels) {

  df <- df %>% filter(!is.na(.data[[cond_var]]))

  means <- df %>%
    group_by(.data[[speed_var]], .data[[cond_var]]) %>%
    summarise(
      mean = mean(.data[[dv_name]], na.rm = TRUE),
      sd   = sd(.data[[dv_name]], na.rm = TRUE),
      n    = dplyr::n(),
      .groups = "drop"
    ) %>%
    mutate(
      se = sd / sqrt(n),
      ci = qt(0.975, df = n - 1) * se
    ) %>%
    rename(
      speed = .data[[speed_var]],
      cond  = .data[[cond_var]]
    )

  pred_grid <- expand.grid(
    speed_num = speed_breaks,
    target_present = levels(droplevels(df[[cond_var]]))
  )
  pred_grid$pred <- predict(model, newdata = pred_grid, re.form = NA)

  names(pred_grid)[names(pred_grid) == "speed_num"] <- speed_var
  names(pred_grid)[names(pred_grid) == "target_present"] <- cond_var

  ggplot(df,
         aes(x = .data[[speed_var]],
             y = .data[[dv_name]],
             color = .data[[cond_var]])) +

    geom_line(
      aes(group = interaction(.data[[group_var]], .data[[cond_var]])),
      alpha = 0.10,
      linewidth = 0.5
    ) +

    geom_errorbar(
      data = means,
      inherit.aes = FALSE,
      aes(
        x = speed,
        ymin = mean - ci,
        ymax = mean + ci,
        color = cond
      ),
      width = 0.25,
      linewidth = 0.8
    ) +

    geom_point(
      data = means,
      inherit.aes = FALSE,
      aes(x = speed, y = mean, color = cond),
      size = 2.8
    ) +

    geom_line(
      data = pred_grid,
      aes(
        x = .data[[speed_var]],
        y = pred,
        color = .data[[cond_var]],
        group = .data[[cond_var]]
      ),
      linewidth = 1.4
    ) +

    scale_x_continuous(
      breaks = speed_breaks,
      labels = speed_labels
    ) +

    labs(
      x = "Velocity (deg/s)",
      y = ylab,
      color = "Target"
    ) +
    theme_minimal(base_size = 18) +
    theme(
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank()
    )
}

# ============================================================
# Generate plots
# ============================================================

p_rt_spag <- plot_spaghetti_with_lmm(
  dt_rt, "rt", m_rt, "RT (s)",
  speed_breaks = speed_breaks,
  speed_labels = speed_labels
)

p_acc_spag <- plot_spaghetti_with_lmm(
  dt_acc, "acc", m_acc, "Accuracy",
  speed_breaks = speed_breaks,
  speed_labels = speed_labels
)

p_fix_spag <- plot_spaghetti_with_lmm(
  dt_eye, "fix_count", m_fix, "Fixation count",
  speed_breaks = speed_breaks,
  speed_labels = speed_labels
)

p_scan_spag <- plot_spaghetti_with_lmm(
  dt_eye, "scanpath", m_scan, "Scanpath length (deg)",
  speed_breaks = speed_breaks,
  speed_labels = speed_labels
)

p_disp_spag <- plot_spaghetti_with_lmm(
  dt_eye, "dispersion", m_disp, "Dispersion (deg²)",
  speed_breaks = speed_breaks,
  speed_labels = speed_labels
)

p_ctr_spag <- plot_spaghetti_with_lmm(
  dt_eye, "center_dist", m_ctr, "Distance from centre (deg)",
  speed_breaks = speed_breaks,
  speed_labels = speed_labels
)

print(p_rt_spag)
print(p_acc_spag)
print(p_fix_spag)
print(p_scan_spag)
print(p_disp_spag)
print(p_ctr_spag)
