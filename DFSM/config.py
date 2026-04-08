#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:35:11 2026

@author: herttaleinonen
"""

CELL_SIZE_PX = 35.0
GRID_SIZE_X = 35
GRID_SIZE_Y = 35

GRID_PX_W = GRID_SIZE_X * CELL_SIZE_PX
GRID_PX_H = GRID_SIZE_Y * CELL_SIZE_PX

GRID_OFFSET_X = -GRID_PX_W / 2.0
GRID_OFFSET_Y = -GRID_PX_H / 2.0

PPD = 37.0
DEG_PER_CELL = CELL_SIZE_PX / PPD

DT_TO_SPEED = {"dt1": 0, "dt2": 100, "dt3": 200, "dt4": 300, "dt5": 400}
VT_TO_SPEED = {"vt1": 0, "vt2": 100, "vt3": 200, "vt4": 300, "vt5": 400}

SPEED_ORDER = [0, 100, 200, 300, 400]

GLOBAL_TND_S = 0.50 # non-decision time

MODEL_SAMPLING_RATE = 0.05 # 0.05 = fixed-sampling-rate model; None = object-displacement-based discretization model


# =========================
# Null model switches:
# =========================

GAZE_MODE = "real" # change to "random"/"shuffle_time"/"center"/"corner_tl"/"shift_time"/"random_from_real_hist" for null model runs, 
                   # keep      "real" for normal model run


VISIBILITY_MODE = "empirical" # change to  "constant_from_speed0" for null model run 
                              # keep       "empirical" for normal model run 
VIS_NULL_SOURCE_SPEED = 0
VIS_CONSTANT_RULE = "mean" # or "median"/"foveal"
VIS_REF_ECC_DEG = 0.0

