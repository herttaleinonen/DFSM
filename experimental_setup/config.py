#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 23 13:09:28 2025

@author: herttaleinonen

      Contains global parameters for the tasks. 
"""

import numpy as np
from datetime import datetime
import os

# ========== Grid Settings ==========
grid_size_x = 35
grid_size_y = 35
cell_size = 35
DIAGONAL_SCALE = round(4 / np.sqrt(2), 2)

# ========== Task & Timing Settings ==========
num_trials = 100 # <-- CHANGE THIS TO CHANGE NUMBER OF TRIALS. 10 for training set and 100 for other sets.
trial_duration = 3.5 # 3500ms
feedback_duration = 1
timeout_feedback_text = "Time's up."

transition_steps = 10 # = <-- CHANGE THIS TO CHANGE OBJECT SPEED. 800 = 0 px/s, 27 = 100 px/s, 13 = 200 px/s, 10 = 300 px/s, 7 = 400 px/s

movement_delay = 1.0 / 30.0 # divided with the frame rate 

# ========== Stimulus Settings ==========
orientations = (170, 135, 120, 80, 20) 
spatial_frequencies = [1 / (cell_size / 2)]
target_orientation = 45
target_sf = 1 / (cell_size / 2)

# ========== Output Settings ==========
exp_info = {"Participant ID": ""}
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
