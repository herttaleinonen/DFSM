## Data analysis scripts

This repository contains data analysis scripts for the behavioral and eye‑movement data recorded in the visual search and visibility experiments.

All `.md` scripts can be run in RStudio to reproduce the figures reported in the paper.

### Data files

The aggregated data used by the analysis scripts are located in the data-folder:

- `long.csv` — long‑format data (used by the R analysis scripts)
- `wide.csv` — wide‑format version of the same data

The complete, non‑aggregated dataset is available in:

`Dynamic_visual_environments/DFSM/data`  
https://github.com/herttaleinonen/Dynamic_visual_environments/tree/main/DFSM/data

### Scripts

- `wide.py`  
  Python script used to generate the aggregated dataset from the full raw data.

- `long.md`  
  R script that converts the wide‑format data into long format.

- `mm.md`  
  R script that runs the linear mixed‑effects model analyses for the behavioral and eye‑movement data from the visual search experiment.

- `vis_mm.md`  
  R script that runs the linear mixed‑effects model analyses for the behavioral data from the visibility experiment.

- `vis.md`  
  R script that generates the figure depicting peripheral visibility across different velocity conditions.

### Requirements
The R scripts require packages listed at the top of each `.md` file.

