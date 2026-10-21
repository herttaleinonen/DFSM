# Created on Wed Oct  7 10:46:53 2026

# @author: herttaleinonen

# ============================================================
# Experiment 1 (free viewing) vs Experiment 2 (central fixation)
# Combined mixed models: does the velocity effect differ by experiment?

# Inputs : data/long_1.csv (free viewing), data/long_2.csv (centre gaze)
# Outputs: data/lmm_experiment_ftests.csv
#          data/lmm_experiment_slopes.csv
# ============================================================

library(readr)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(emmeans)

# ------------------------------------------------------------
# Load both experiments; unique participant IDs (different people!)
# ------------------------------------------------------------
load_exp <- function(path, exp_label) {
  read_csv(path, show_col_types = FALSE) %>%
    mutate(experiment = exp_label,
           pid = paste(exp_label, participant, sep = "_"))
}

dat <- bind_rows(
  load_exp("data/long_1.csv", "free"),
  load_exp("data/long_2.csv", "centre")
)

speed_map <- tibble(
  task      = paste0("dt", 1:5),
  speed_num = c(0.000, 2.703, 5.406, 8.109, 10.812),
  speed_lab = c("0", "3", "5.5", "8", "11")
)

dt <- dat %>%
  filter(task %in% paste0("dt", 1:5)) %>%
  left_join(speed_map, by = "task") %>%
  mutate(
    pid            = factor(pid),
    experiment     = factor(experiment, levels = c("free", "centre")),
    target_present = factor(target_present, levels = c(0, 1),
                            labels = c("absent", "present"))
  ) %>%
  filter(!is.na(target_present))   # drops padding rows from the wide format

cat("\nParticipants per experiment:\n")
print(dt %>% distinct(experiment, pid) %>% count(experiment))

# -0.5 / +0.5 coding: experiment coefficient = centre - free,
# target coefficient = present - absent
set_contr <- function(d) {
  contrasts(d$target_present) <- c(-0.5, 0.5)
  contrasts(d$experiment)     <- c(-0.5, 0.5)
  d
}

# ------------------------------------------------------------
# Aggregate to participant x speed x target cells 
# ------------------------------------------------------------
cells <- c("pid", "experiment", "speed_num", "target_present")

dt_rt <- dt %>%
  filter(correct == 1) %>%
  group_by(across(all_of(cells))) %>%
  summarise(log_rt = mean(log(rt_s), na.rm = TRUE), .groups = "drop") %>%
  set_contr()

n_per_cell <- dt %>%
  group_by(across(all_of(cells))) %>%
  summarise(n = n(), .groups = "drop") %>%
  pull(n) %>% median()
eps <- 0.5 / n_per_cell
cat(sprintf("\nAccuracy continuity correction eps = %.4f (median %d trials/cell)\n",
            eps, round(n_per_cell)))

dt_acc <- dt %>%
  group_by(across(all_of(cells))) %>%
  summarise(acc = mean(correct, na.rm = TRUE), .groups = "drop") %>%
  mutate(acc_clipped = pmin(acc, 1 - eps),
         logit_acc   = log(acc_clipped / (1 - acc_clipped))) %>%
  set_contr()

dt_eye <- dt %>%
  group_by(across(all_of(cells))) %>%
  summarise(
    fix_count   = mean(fix_count,           na.rm = TRUE),
    scanpath    = mean(fix_path_length_deg, na.rm = TRUE),
    dispersion  = mean(fix_dispersion_deg2, na.rm = TRUE),
    center_dist = mean(fix_center_dist_deg, na.rm = TRUE),
    n_trials    = sum(!is.na(fix_center_dist_deg)),
    .groups = "drop"
  ) %>%
  filter(n_trials > 0) %>%
  set_contr()

# ------------------------------------------------------------
# Models: same procedure as the per-experiment analysis
# (maximal random slopes; fall back to uncorrelated velocity slopes
#  on a convergence WARNING). Singularity is reported explicitly.
# experiment is between-participant, so it has no random slope.
# ------------------------------------------------------------
fit_combined <- function(dv, data) {
  f_full <- paste0(dv, " ~ speed_num * target_present * experiment + ",
                   "(1 + speed_num + target_present | pid)")
  m <- tryCatch(
    lmer(as.formula(f_full), data = data, REML = FALSE),
    warning = function(w) {
      cat(sprintf("[WARN] %s: %s\n  -> falling back to (1 + speed_num || pid)\n",
                  dv, conditionMessage(w)))
      NULL
    }
  )
  structure <- "(1 + speed_num + target_present | pid)"
  if (is.null(m)) {
    f_fb <- paste0(dv, " ~ speed_num * target_present * experiment + ",
                   "(1 + speed_num || pid)")
    m <- lmer(as.formula(f_fb), data = data, REML = FALSE)
    structure <- "(1 + speed_num || pid)"
  }
  cat(sprintf("%s: random structure %s; singular = %s\n",
              dv, structure, isSingular(m)))
  m
}

specs <- list(
  "RT (log)"         = list(dv = "log_rt",      data = dt_rt),
  "Accuracy (logit)" = list(dv = "logit_acc",   data = dt_acc),
  "Fixation count"   = list(dv = "fix_count",   data = dt_eye),
  "Scanpath length"  = list(dv = "scanpath",    data = dt_eye),
  "Dispersion"       = list(dv = "dispersion",  data = dt_eye),
  "Centre distance"  = list(dv = "center_dist", data = dt_eye)
)

models <- lapply(specs, function(s) fit_combined(s$dv, s$data))

# ------------------------------------------------------------
# Type III F-tests (Satterthwaite). Key terms for the reviewers:
#   speed_num:experiment                   (does the velocity effect differ?)
#   speed_num:target_present:experiment    (does the velocity x target pattern differ?)
# ------------------------------------------------------------
f_tab <- bind_rows(lapply(names(models), function(nm) {
  a <- as.data.frame(anova(models[[nm]]))
  a$term <- rownames(a)
  a$outcome <- nm
  a %>% select(outcome, term, everything())
}))
print(tibble::as_tibble(f_tab), n = Inf, width = Inf)
write_csv(f_tab, "data/lmm_experiment_ftests.csv")

# ------------------------------------------------------------
# Velocity slope within each experiment, with 95% CIs, and their difference.
# Slopes are per deg/s, averaged over target presence.
# RT slopes are on the log scale: percent change per deg/s = exp(b) - 1.
# ------------------------------------------------------------
slopes <- bind_rows(lapply(names(models), function(nm) {
  em <- emtrends(models[[nm]], ~ experiment, var = "speed_num",
                 data = specs[[nm]]$data, lmer.df = "satterthwaite")
  cat(sprintf("\n--- %s: velocity slope per experiment ---\n", nm))
  print(summary(em, infer = TRUE))
  cat("Difference (free - centre):\n")
  print(summary(pairs(em), infer = TRUE))
  as.data.frame(summary(em, infer = TRUE)) %>% mutate(outcome = nm, .before = 1)
}))
write_csv(slopes, "data/lmm_experiment_slopes.csv")


m_sub2 <- lmer(logit_acc ~ speed_num * target_present * experiment +
                   (1 + speed_num || pid), data = sub, REML = FALSE)
anova(m_sub2)
emtrends(m_sub2, ~ experiment, var = "speed_num", lmer.df = "satterthwaite")
pairs(emtrends(m_sub2, ~ experiment, var = "speed_num", lmer.df = "satterthwaite"))
