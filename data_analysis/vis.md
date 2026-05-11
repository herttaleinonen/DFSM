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
#    - numeric speed: exact deg/s (used for calculations)
#    - speed_lab: factor used only for plotting / legend labels
# ------------------------------------------------------------

vt <- dat %>%
  filter(task %in% paste0("vt", 1:5)) %>%
  mutate(
    participant = factor(participant),

    # exact numeric speeds (deg/s)
    speed = case_when(
      task == "vt1" ~ 0.000,
      task == "vt2" ~ 2.703,
      task == "vt3" ~ 5.406,
      task == "vt4" ~ 8.109,
      task == "vt5" ~ 10.812
    ),

    # plotting labels 
    speed_lab = factor(
      c("0", "3", "5.5", "8", "11")[match(task, paste0("vt", 1:5))],
      levels = c("0", "3", "5.5", "8", "11")
    ),

    correct = as.numeric(correct),
    ecc_deg = as.numeric(ecc_deg)
  ) %>%
  filter(!is.na(speed), !is.na(correct), !is.na(ecc_deg))

# ------------------------------------------------------------
# 3) Participant-level accuracy
# ------------------------------------------------------------

vt_participant <- vt %>%
  group_by(participant, speed, speed_lab, ecc_deg) %>%
  summarise(
    acc = mean(correct, na.rm = TRUE),
    .groups = "drop"
  )

# ------------------------------------------------------------
# 4) Group mean + SEM
# ------------------------------------------------------------

vt_summary <- vt_participant %>%
  group_by(speed_lab, ecc_deg) %>%
  summarise(
    mean_acc = mean(acc, na.rm = TRUE),
    sem = sd(acc, na.rm = TRUE) / sqrt(n()),
    .groups = "drop"
  )

# ------------------------------------------------------------
# 5) Plot
# ------------------------------------------------------------

p_visibility <- ggplot(
  vt_summary,
  aes(
    x = ecc_deg,
    y = mean_acc,
    color = speed_lab,
    group = speed_lab
  )
) +
  geom_line(linewidth = 1.2) +
  geom_point(size = 2.5) +
  geom_errorbar(
    aes(
      ymin = mean_acc - sem,
      ymax = mean_acc + sem
    ),
    width = 0.2
  ) +
  labs(
    x = "Eccentricity (deg)",
    y = "Accuracy",
    color = "Speed (deg/s)",
    title = "Visibility as a function of eccentricity"
  ) +
  scale_y_continuous(
    breaks = seq(0.6, 1, 0.1),
    limits = c(0.6, 1)
  ) +
  theme_classic() +
  theme(
    text = element_text(size = 14),
    panel.grid.major.y = element_line(color = "grey85", linewidth = 0.6),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white"),
    plot.tag = element_text(size = 24, face = "bold")
  ) + labs(tag = "a.")

print(p_visibility)

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
