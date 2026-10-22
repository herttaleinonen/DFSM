#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:52:42 2026

@author: herttaleinonen
"""

import numpy as np
import pandas as pd
import os

from config import (
    MODEL_SAMPLING_RATE,
    GAZE_MODE,
    GLOBAL_TND_S,
)

from parsing_and_gaze import (
    debug_inventory,
    collect_visibility_files,
    collect_search_pairs,
)

from fitting import (
    make_synthetic_human_trials,
    fit_eta_theta_from_trials_fast,
    run_loss_surface_batch,
    run_model_recovery,
    fit_model_comparison_per_participant,
    fit_model_comparison_visibility,
    preprocess_participant_trials,
    fit_model_per_participant_kfold,
    run_kfold_pooled_replay
)

from visibility import build_dprime_splines_for_participant

from replay import (
    run_full_replay,
    build_saccade_prediction_table,
)


# =========================
# Main
# =========================

if __name__ == "__main__":
    
    SEARCH_DIR = "data/search"
    VIS_DIR = "data/visibility"
    debug_inventory(SEARCH_DIR, VIS_DIR)
    
    DO_FIT = True
    
    if DO_FIT:
        
        print("\n=============================")
        print("RUNNING PARAMETER ESTIMATION")
        print("==============================")

        ETA_GRID   = np.linspace(0.008, 0.80, 12)       # old: 0.002 - 0.20
        THETA_GRID = np.geomspace(0.08, 4.0, 12)        # old: 0.02 - 1.0

        kfold_df, fit_df = fit_model_per_participant_kfold(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_grid=ETA_GRID,
            theta_grid=THETA_GRID,
            k=10,
            n_repeats_fit=5,
            spline_s=0.5,
            out_csv_folds="kfold_results.csv",           
            out_csv_final="fitted_params.csv",
            eta_lo=0.04,       
            eta_hi=1.20,       
            theta_lo=0.08,     
            theta_hi=16.0,     
            dt_override_s=MODEL_SAMPLING_RATE,
            use_group_visibility=False,
            split_seed_offset=42345,
            rt_tp_sd_override=0.2450,
            rt_ta_sd_override=0.3569,
            dprime_sd_override=0.694,
        )

        print("Wrote kfold_results.csv and fitted_params.csv")


    DO_REPLAY = True
    
    if DO_REPLAY:
        
        print("\n=============================")
        print("RUNNING DSFM")
        print("==============================")

        run_full_replay(
            search_dir=SEARCH_DIR, 
            visibility_dir=VIS_DIR,
            output_csv="replay_model_results.csv",
            fitted_params_csv="fitted_params.csv",   
            n_model_repeats=1000,
            spline_s=0.5,
            max_calib_drift_deg=None,
            dt_override_s=MODEL_SAMPLING_RATE,
            use_group_visibility=False,
            gaze_mode=GAZE_MODE,      
        )
        
    
    DO_KFOLD_POOLED_REPLAY = True
    
    if DO_KFOLD_POOLED_REPLAY:
        
        print("\n=============================")
        print("RUNNING KFOLD POOLED REPLAY")
        print("==============================")
        
        kfold_df = pd.read_csv("kfold_results.csv")
        run_kfold_pooled_replay(
            search_dir=SEARCH_DIR, 
            visibility_dir=VIS_DIR,
            kfold_df=kfold_df,
            output_csv="replay_model_results_kfold_pooled.csv",
            k=10,
            n_model_repeats=1000,
            dt_override_s=MODEL_SAMPLING_RATE,
            use_group_visibility=False,
            split_seed_offset=42345,   #42345
        )
    
    DO_SACCADE_PREDICTION = True  
    
    if DO_SACCADE_PREDICTION:
        
        print("\n==============================")
        print("RUNNING SACCADE PREDICTION")
        print("==============================")
        
        sac_df = build_saccade_prediction_table(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            fitted_params_csv="fitted_params.csv",
            output_csv="saccade_prediction_table.csv",
            spline_s=0.5,
            dt_override_s=MODEL_SAMPLING_RATE,
            gaze_mode=GAZE_MODE,
        )
 
    
    
    #=================================================================================
    #                       OPTIONAL DEBUG-RUNS BELOW:
    #=================================================================================
    
    DO_NAN_GAZE_CHECK = False # <-- change to True only when checking for frames missing gaze position
    if DO_NAN_GAZE_CHECK:
        print("\n==============================")
        print("NAN-GAZE FALLBACK CHECK")
        print("==============================")

        vis_by_pp = collect_visibility_files(VIS_DIR)
        search_by_pp = collect_search_pairs(SEARCH_DIR)

        n_nan_frames = 0
        n_total_frames = 0
        n_trials_all_nan = 0
        n_trials_total = 0

        for pp, pairs in sorted(search_by_pp.items()):
            if pp not in vis_by_pp:
                continue
            dprime_model_pp = build_dprime_splines_for_participant(vis_by_pp[pp], spline_s=0.5)
            trials_pp = preprocess_participant_trials(pp, pairs, dprime_model_pp)

            for trial in trials_pp:
                gaze = trial["gaze_cells"]   # <-- confirm this matches your actual key name
                nan_mask = np.any(np.isnan(gaze), axis=1)
                n_nan_frames += nan_mask.sum()
                n_total_frames += len(gaze)
                n_trials_total += 1
                if nan_mask.all():
                    n_trials_all_nan += 1

        print(f"{n_nan_frames}/{n_total_frames} frames "
              f"({100*n_nan_frames/n_total_frames:.3f}%) hit the center fallback")
        print(f"{n_trials_all_nan}/{n_trials_total} trials had gaze entirely missing")

        
    DO_VIS_COMPARISON = False # <-- change to True only when running visibility null-model comparisons
    
    if DO_VIS_COMPARISON:
        print("\n==============================")
        print("RUNNING VISIBILITY NULL MODEL COMPARISON")
        print("==============================")
        
        print("\n--- Group visibility vs individual ---")
        fit_model_comparison_visibility(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_grid = np.linspace(0.008, 0.80, 12),
            theta_grid = np.geomspace(0.08, 4.0, 12),
            n_repeats_fit=5,
            spline_s=0.5,
            out_csv="model_comparison_visibility_group.csv",
            null_mode="group",
            eta_lo=0.04, eta_hi=1.20, theta_lo=0.08, theta_hi=16.0,          
            dt_override_s=MODEL_SAMPLING_RATE,
            rt_tp_sd=0.2450, rt_ta_sd=0.3569, dprime_sd=0.694    
        )
        
        print("\n--- Mean-across-speeds visibility vs individual ---")
        fit_model_comparison_visibility(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_grid=np.linspace(0.008, 0.80, 12),
            theta_grid=np.geomspace(0.08, 4.0, 12),
            n_repeats_fit=5,
            spline_s=0.5,
            out_csv="model_comparison_visibility_mean_across_speeds.csv",
            null_mode="mean_across_speeds",
            eta_lo=0.04, eta_hi=1.20, theta_lo=0.08, theta_hi=16.0,
            dt_override_s=MODEL_SAMPLING_RATE,
            rt_tp_sd=0.2450, rt_ta_sd=0.3569, dprime_sd=0.694,
        )
        
        print("\n--- Constant visibility vs individual ---")
        fit_model_comparison_visibility(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_grid = np.linspace(0.008, 0.80, 12),
            theta_grid = np.geomspace(0.08, 4.0, 12),
            n_repeats_fit=5,
            spline_s=0.5,
            out_csv="model_comparison_visibility_constant.csv",
            null_mode="constant",
            eta_lo=0.04, eta_hi=1.20, theta_lo=0.08, theta_hi=16.0,          
            dt_override_s=MODEL_SAMPLING_RATE,
            rt_tp_sd=0.2450, rt_ta_sd=0.3569, dprime_sd=0.694    
        )

        
    DO_MODEL_COMPARISON = False # <-- change to True only when running parameter null model comparisons (AIC)
    
    if DO_MODEL_COMPARISON:
        
        _full = pd.read_csv("fitted_params.csv")   
        
        ETA_FIXED   = float(_full["eta"].mean())
        THETA_FIXED = float(_full["theta"].mean())
        
        ETA_GRID_C   = np.linspace(0.008, 0.80, 12)
        THETA_GRID_C = np.geomspace(0.08, 4.0, 12)
    
        print("\n==============================")
        print("RUNNING MODEL COMPARISON (AIC)")
        print("==============================")
    
        cmp_df = fit_model_comparison_per_participant(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_grid=ETA_GRID_C,
            theta_grid=THETA_GRID_C,
            eta_fixed=ETA_FIXED,
            theta_fixed=THETA_FIXED,
            n_repeats_fit=5,
            spline_s=0.5,
            eta_lo=0.04, eta_hi=1.20,
            theta_lo=0.08, theta_hi=16.0,
            out_csv="model_comparison.csv",        
            dt_override_s=MODEL_SAMPLING_RATE,
            rt_tp_sd=0.2450, rt_ta_sd=0.3569, dprime_sd=0.694,
        )
    
        cols = ["participant",
                "m2_AIC", "meta_AIC", "mtheta_AIC",
                "delta_AIC_vs_eta", "delta_AIC_vs_theta"]
        print("\n" + cmp_df[cols].to_string(index=False))
        

    DO_RECOVERY = False  # <-- change to True only when running model recovery analysis
    
    if DO_RECOVERY:
        
        ETA_TRUE   = np.linspace(0.08, 0.72, 5)
        THETA_TRUE = np.linspace(0.4, 3.6, 5)
    
        ETA_GRID   = np.linspace(0.008, 0.80, 20)
        THETA_GRID = np.geomspace(0.08, 4.0, 20)
    
        rec = run_model_recovery(
            search_dir=SEARCH_DIR,
            visibility_dir=VIS_DIR,
            eta_true_grid=ETA_TRUE,
            theta_true_grid=THETA_TRUE,
            eta_fit_grid=ETA_GRID,
            theta_fit_grid=THETA_GRID,
            n_repeats_fit=10,
            spline_s=0.5,
            out_csv="model_recovery.csv",   
            dt_override_s=MODEL_SAMPLING_RATE,
            rt_tp_sd=0.2450, rt_ta_sd=0.3569, dprime_sd=0.694,
        )
        
    
