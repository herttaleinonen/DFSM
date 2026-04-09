# Created on Tue Feb 3 10:35:50 2026

# @author: herttaleinonen

library(readr)
library(dplyr)
library(tidyr)

wide <- read_csv(
  "data/wide.csv",
)

long_trials <- wide %>%
  pivot_longer(
    cols = -participant,
    names_to = "column",
    values_to = "value"
  ) %>%
  extract(
    column,
    into = c("task", "trial", "variable"),
    regex = "^(dt\\d|vt\\d)_t(\\d{3})_(.+)$"
  ) %>%
  mutate(trial = as.integer(trial)) %>%
  pivot_wider(
    names_from = variable,
    values_from = value
  )

write_csv(long_trials, "data/long.csv")
