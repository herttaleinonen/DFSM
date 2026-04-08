#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:52:42 2026

@author: herttaleinonen
"""

import numpy as np

from config import (MODEL_SAMPLING_RATE, GAZE_MODE)
from parsing_and_gaze import debug_inventory
from replay import (run_full_replay, build_saccade_prediction_table)
from fitting import (fit_model_per_participant,run_model_recovery)


# =========================
# Main
# =========================
if __name__ == "__main__":
    SEARCH_DIR = "data/search"
    VIS_DIR = "data/visibility"
    debug_inventory(SEARCH_DIR, VIS_DIR)
    
    
    DO_RECOVERY = False # <-- change to True only when running model recovery

    if DO_RECOVERY:
        ETA_TRUE   = np.array([0.05, 0.144, 0.24])
        THETA_TRUE = np.array([8, 15, 20])
    
        ETA_GRID   = np.linspace(0.05, 0.30, 11)
        THETA_GRID = np.linspace(4.0, 22.0, 10)
        
        print("\n==============================")
        print("RUNNING MODEL RECOVERY")
        print("==============================")
    
        rec = run_model_recovery(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_true_grid=ETA_TRUE,
            theta_true_grid=THETA_TRUE,
            eta_fit_grid=ETA_GRID,
            theta_fit_grid=THETA_GRID,
            n_repeats_fit=5,
            spline_s=0.5,
            out_csv="model_recovery.csv",
        )
    

    DO_FIT = False  # <-- change to True only when re-fitting the two free parameters

    if DO_FIT:
        # parameter grid size 
        ETA_GRID   = np.linspace(0.002, 0.20, 12) 
        THETA_GRID = np.linspace(0.02, 1.0, 12)
    
        print("\n==============================")
        print("RUNNING PARAMETER GRID SEARCH")
        print("==============================")
    
        fit_df = fit_model_per_participant(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_grid=ETA_GRID,
            theta_grid=THETA_GRID,
            n_repeats_fit=5,
            spline_s=0.5,
            out_csv="fitted_params_test.csv",
            warmstart_csv=None,
            eta_lo=0.01,
            eta_hi=0.30,
            theta_lo=0.10,
            theta_hi=4.0,
            dt_override_s=MODEL_SAMPLING_RATE,
        )
    
        print("Wrote fitted_params_test.csv")
    
    
    DO_REPLAY = True # <-- change to True when running the DFSM model

    if DO_REPLAY:
        
        print("\n==============================")
        print("RUNNING DSFM")
        print("==============================")
        
        run_full_replay(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            output_csv="replay_model_results_test.csv",
            fitted_params_csv="fitted_params_test.csv",
            n_model_repeats=100,
            spline_s=0.5,
            max_calib_drift_deg=None,
            dt_override_s=MODEL_SAMPLING_RATE,
        )
    
    
    DO_SACCADE_PREDICTION = True  # <-- change to True only when running saccade prediction analysis
    
    if DO_SACCADE_PREDICTION:
        
        print("\n==============================")
        print("RUNNING SACCADE PREDICTION")
        print("==============================")
        
        sac_df = build_saccade_prediction_table(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            fitted_params_csv="fitted_params_test.csv",
            output_csv="saccade_prediction_table_test.csv",
            spline_s=0.5,
            dt_override_s=MODEL_SAMPLING_RATE,
            gaze_mode=GAZE_MODE,
        )
