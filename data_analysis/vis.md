# Created on Fri Mar  6 16:52:32 2026

# @author: herttaleinonen

# ============================================================
# Visibility functions by speed
# Uses long.csv dataset
# Plots mean accuracy ± SEM for each speed condition
# ============================================================

library(readr)
library(dplyr)
library(ggplot2)

# ------------------------------------------------------------
# 1) Load data
# ------------------------------------------------------------

dat <- read_csv("data/long.csv", show_col_types = FALSE)

# ------------------------------------------------------------
# 2) Keep visibility tasks and define speed
# ------------------------------------------------------------

vt <- dat %>%
  filter(task %in% paste0("vt", 1:5)) %>%
  mutate(
    participant = factor(participant),

    speed = case_when(
      task == "vt1" ~ 0,
      task == "vt2" ~ 3,
      task == "vt3" ~ 6,
      task == "vt4" ~ 8,
      task == "vt5" ~ 11
    ),

    speed = factor(speed),
    correct = as.numeric(correct)
  )

# ------------------------------------------------------------
# 3) Participant-level accuracy
# ------------------------------------------------------------

vt_participant <- vt %>%
  group_by(participant, speed, ecc_deg) %>%
  summarise(
    acc = mean(correct, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------
# 4) Group mean + SEM
# ------------------------------------------------------------

vt_summary <- vt_participant %>%
  group_by(speed, ecc_deg) %>%
  summarise(
    mean_acc = mean(acc, na.rm = TRUE),
    sem = sd(acc, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

print(vt_summary)

# ------------------------------------------------------------
# 5) Plot
# ------------------------------------------------------------

p_visibility <- ggplot(vt_summary,
                       aes(x = ecc_deg,
                           y = mean_acc,
                           color = speed,
                           group = speed)) +

  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +

  geom_errorbar(
    aes(ymin = mean_acc - sem,
        ymax = mean_acc + sem),
    width = 0.2
  ) +

  labs(
    x = "Eccentricity (deg)",
    y = "Accuracy",
    color = "Velocity (deg/s)",
    title = "Visibility as a function of eccentricity"
  ) +

  scale_y_continuous(breaks = seq(0.6,1,0.1)) +
  coord_cartesian(ylim = c(0.6,1)) +

  theme_classic() +
  theme(
    text = element_text(size = 14),
    panel.grid.major.y = element_line(color = "grey85", linewidth = 0.6),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white")
  )

# ------------------------------------------------------------
# 6) Save figure
# ------------------------------------------------------------

ggsave(
  "visibility_functions_by_speed.png",
  p_visibility,
  width = 6,
  height = 4,
  dpi = 300
)
