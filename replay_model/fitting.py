#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:49:12 2026

@author: herttaleinonen
"""

import os
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from numpy.random import SeedSequence
import zlib
from scipy.stats import norm

from config import (
    GLOBAL_TND_S,
    SPEED_ORDER,
    DT_TO_SPEED,
    VISIBILITY_MODE,
    VIS_NULL_SOURCE_SPEED,
    VIS_CONSTANT_RULE,
    VIS_REF_ECC_DEG,
    GRID_SIZE_X,
    GRID_SIZE_Y,
    GAZE_MODE,
    TIMEOUT_DV_THRESHOLD,
    T_VIS_S,
)

from parsing_and_gaze import (
    safe_int,
    safe_float,
    safe_parse_positions,
    safe_parse_trajectory,
    parse_asc_events,
    estimate_dt_from_positions,
    dt_from_duration,
    gaze_series_cells_from_fixations,
    collect_visibility_files,
    collect_search_pairs,
    dt_from_name,
    stable_trial_seed,
    perturb_gaze,
)

from visibility import (
    build_dprime_splines_for_participant,
    make_visibility_null_model,
    build_group_mean_dprime_splines,
)

from replay import (
    run_replay_trial,
    infer_target_index,
    ecc_deg_from_cells,
)


# ------------------
# Pre-processing
# ------------------

def make_refined_grid(center: float, half_width: float, n: int, lo: float, hi: float) -> np.ndarray:
    """n-point grid from center-half_width .. center+half_width, clipped to [lo, hi]."""
    g = np.linspace(center - half_width, center + half_width, n)
    return np.clip(g, lo, hi)


def safe_log(x, eps=1e-6):
    return np.log(np.maximum(x, eps))


def _human_scales(human_by_speed, eps: float = 1e-6) -> dict:
    """
    Compute SD across speeds for each summary metric (ignore NaNs).

    For RT metrics, compute SD in log-space (to match loss space).
    Returns: dict like {"hit": sd, "fa": sd, "rt_tp": sd, "rt_ta": sd}
    """
    scales = {}

    for key in ["hit", "fa", "rt_tp", "rt_ta"]:
        vals = []
        for s in SPEED_ORDER:
            v = human_by_speed[s].get(key, np.nan)
            if np.isnan(v):
                continue

            if key.startswith("rt_"):
                vals.append(float(safe_log(v)))  # match loss space
            else:
                vals.append(float(v))

        vals = np.asarray(vals, float)
        sd = float(np.std(vals, ddof=1)) if vals.size >= 2 else np.nan

        # fallback: avoid division by tiny numbers
        if np.isnan(sd) or sd < eps:
            sd = 1.0

        scales[key] = sd

    return scales


def preprocess_participant_trials(
    pp: str,
    pairs: List[Tuple[str, str]],
    dprime_models_pp: Dict[int, UnivariateSpline],
) -> List[dict]:
    """
    Returns a list of dicts, each containing everything needed to simulate a trial.
    This is the expensive part (ASC parsing, gaze reconstruction, literal_eval of positions)
    and should be done ONCE per participant.
    """
    trials_out: List[dict] = []

    for csv_path, asc_path in pairs:
        dt = dt_from_name(csv_path)
        if dt is None:
            continue
        speed_px_s = DT_TO_SPEED[dt]
        if speed_px_s not in dprime_models_pp:
            continue

        # speed-dependent schedules (fixed during fit)
        #alpha_use = ALPHA_BASE * (1 - ALPHA_SPEED_SLOPE * (speed_px_s / 400))

        df = pd.read_csv(csv_path)
        needed = ["Trial", "Gabor Positions", "Target Present", "Target Trajectory",
          "Response", "Correct", "Reaction Time (s)"]

        if any(c not in df.columns for c in needed):
            continue

        (screen_w, screen_h), eye_trials = parse_asc_events(asc_path)

        df["gabor_pos"] = df["Gabor Positions"].apply(safe_parse_positions)

        for _, r in df.iterrows():
            trial = safe_int(r.get("Trial"))
            target_present = safe_int(r.get("Target Present"))
            human_resp = safe_int(r.get("Response"))
            human_corr = safe_int(r.get("Correct"))
            human_rt = safe_float(r.get("Reaction Time (s)"))
            if (trial is None or target_present is None or
                human_resp is None or human_corr is None or human_rt is None):
                continue


            obj = r.get("gabor_pos")
            if obj is None or not isinstance(obj, np.ndarray) or obj.ndim != 3 or obj.shape[0] < 2:
                continue

            # dt estimate
            dt_s = estimate_dt_from_positions(obj, speed_px_s) if speed_px_s > 0 else None
            if dt_s is None or dt_s <= 0:
                dt_s = dt_from_duration(obj, 3.5)

            # gaze series
            tr_eye = eye_trials.get(trial)
            if tr_eye is None:
                gaze_cells = np.full((obj.shape[0], 2), np.nan, dtype=float)
            else:
                gaze_cells = gaze_series_cells_from_fixations(tr_eye, obj.shape[0], dt_s, screen_w, screen_h)
                

            # target index if present
            target_index = None
            if target_present == 1:
                traj = safe_parse_trajectory(r.get("Target Trajectory"))
                if traj is not None:
                    target_index = infer_target_index(obj, traj)


            #alpha_trial = alpha_use   # same for TP and TA

            # stable per-trial seed (important)
            seed_str = f"{pp}|{os.path.basename(csv_path)}|{trial}"
            trial_seed = zlib.crc32(seed_str.encode("utf-8")) & 0xFFFFFFFF

            trials_out.append(dict(
                speed_px_s=speed_px_s,
                obj=obj,
                gaze_cells=gaze_cells,
                dt_s=dt_s,
                target_present=target_present,
                target_index=target_index,
                #alpha_trial=alpha_trial,
                trial_seed=trial_seed,
                
                # human fields for train/test summaries
                human_resp=human_resp,
                human_corr=human_corr,
                human_rt=human_rt,
            ))

    return trials_out


# ------------------
# Loss function
# ------------------

def dprime_loglinear(hit, n_pos, fa, n_neg):
    # Log-linear corrected d'. hit/fa are rates, n_pos/n_neg the trial counts of that cell.
    H = (hit * n_pos + 0.5) / (n_pos + 1.0)
    F = (fa  * n_neg + 0.5) / (n_neg + 1.0)
    return norm.ppf(H) - norm.ppf(F)


def loss_summary(human_by_speed, model_by_speed,
                 w_choice: float = 1.0,
                 w_rt: float = 1.0,
                 rt_tp_sd: float = 1.0,
                 rt_ta_sd: float = 1.0,
                 dprime_sd: float = 1.0,     
                 ) -> float:
    loss = 0.0
    for s in SPEED_ORDER:
        h = human_by_speed[s]
        m = model_by_speed[s]
        # d' (log-linear, standardised)
        if not (np.isnan(h["hit"]) or np.isnan(h["fa"]) or
                np.isnan(m["hit"]) or np.isnan(m["fa"])):
            n_pos, n_neg = h["n_pos"], h["n_neg"]
            if n_pos > 0 and n_neg > 0:
                d_h = dprime_loglinear(h["hit"], n_pos, h["fa"], n_neg)
                d_m = dprime_loglinear(m["hit"], n_pos, m["fa"], n_neg)
                loss += w_choice * ((d_h - d_m) / dprime_sd) ** 2
        # TP RT -- standardized
        if not (np.isnan(h["rt_tp"]) or np.isnan(m["rt_tp"])):
            z = (safe_log(h["rt_tp"]) - safe_log(m["rt_tp"])) / rt_tp_sd
            loss += w_rt * (z ** 2)
        # TA RT -- standardized
        if not (np.isnan(h["rt_ta"]) or np.isnan(m["rt_ta"])):
            z = (safe_log(h["rt_ta"]) - safe_log(m["rt_ta"])) / rt_ta_sd
            loss += w_rt * (z ** 2)
    return float(loss)



# ------------------
# Fitting helpers
# ------------------

def compute_global_rt_sds(all_human_summaries):
    """
    Computes pooled SD of log-RT (target-present and target-absent, separately)
    across ALL participants and speed conditions, using human data. Call this
    ONCE up front (e.g. using the training-set human summaries for every
    participant), then pass the resulting sds into every fit/loss call so all
    participants and grid points are scored against the same fixed scale --
    exactly analogous to how d' isn't re-normalized per participant.
    """
    tp_vals, ta_vals = [], []
    for human_by_speed in all_human_summaries:
        for s in SPEED_ORDER:
            h = human_by_speed[s]
            if not np.isnan(h["rt_tp"]):
                tp_vals.append(safe_log(h["rt_tp"]))
            if not np.isnan(h["rt_ta"]):
                ta_vals.append(safe_log(h["rt_ta"]))
    return float(np.std(tp_vals, ddof=1)), float(np.std(ta_vals, ddof=1))


def fit_eta_theta_from_trials(
    pp: str,
    trials_pp: List[dict],
    dprime_models_pp: Dict[int, UnivariateSpline],
    eta_grid: np.ndarray,
    theta_grid: np.ndarray,
    n_repeats_fit: int = 10,
    tnd_s: float = GLOBAL_TND_S,
) -> Tuple[float, float, float]:

    #Fit eta, theta on a provided trial list (which already has human_* fields).
    #Returns (eta_hat, theta_hat, best_loss) using the summary loss.

    # stable split like the fitter
    rng_split = np.random.default_rng(12345 + int(pp[2:]))
    idx = rng_split.permutation(len(trials_pp))
    n_train = int(0.7 * len(idx))
    train_trials = [trials_pp[i] for i in idx[:n_train]]

    human_train = compute_human_summary_from_preprocessed(train_trials)

    best = {"loss": np.inf, "eta": None, "theta": None}

    for eta in eta_grid:
        for theta in theta_grid:
            model = simulate_model_summary_from_preprocessed(
                trials_pp=train_trials,
                dprime_models_pp=dprime_models_pp,
                eta=float(eta),
                theta=float(theta),
                n_repeats=n_repeats_fit,
                tnd_s=tnd_s,
            )
            L = loss_summary(human_train, model, w_choice=1.0, w_rt=1.0)
            if L < best["loss"]:
                best = {"loss": float(L), "eta": float(eta), "theta": float(theta)}

    return best["eta"], best["theta"], best["loss"]


# used by the model recovery  
def fit_eta_theta_from_trials_fast(pp, trials_pp, dprime_models_pp, eta_grid, theta_grid,
                                   n_repeats_fit=10, tnd_s=GLOBAL_TND_S,
                                   rt_tp_sd=1.0, rt_ta_sd=1.0, dprime_sd=1.0,
                                   eta_lo=0.04, eta_hi=1.20, theta_lo=0.08, theta_hi=16.0,
                                   eta_half_width_refine=0.072, n_refine=12, dt_override_s=None):
    rng_split = np.random.default_rng(12345 + int(pp[2:]))
    idx = rng_split.permutation(len(trials_pp))
    n_train = int(0.7 * len(idx))
    train_trials = [trials_pp[i] for i in idx[:n_train]]
    human_train = compute_human_summary_from_preprocessed(train_trials)
    kin_cache = {i: precompute_trial_kinematics(tr, dprime_models_pp, dt_override_s=dt_override_s)
             for i, tr in enumerate(train_trials)}

    def _loss(eta, theta):
        model = simulate_model_summary_from_preprocessed_fast(
            trials_pp=train_trials, dprime_models_pp=dprime_models_pp,
            eta=float(eta), theta=float(theta), n_repeats=n_repeats_fit,
            tnd_s=tnd_s, kin_cache=kin_cache,
        )
        return loss_summary(human_train, model, w_choice=1.0, w_rt=1.0,
                            rt_tp_sd=rt_tp_sd, rt_ta_sd=rt_ta_sd, dprime_sd=dprime_sd)

    best = {"loss": np.inf, "eta": None, "theta": None}

    # coarse pass
    for eta in eta_grid:
        for theta in theta_grid:
            L = _loss(eta, theta)
            if L < best["loss"]:
                best = {"loss": L, "eta": float(eta), "theta": float(theta)}

    # refinement pass, same logic as the main fit
    eta_ref = make_refined_grid(best["eta"], eta_half_width_refine, n_refine, lo=eta_lo, hi=eta_hi)
    step = (theta_grid[-1] / theta_grid[0]) ** (1.0 / (len(theta_grid) - 1))
    theta_ref = np.geomspace(max(theta_lo, best["theta"] / step),
                             min(theta_hi, best["theta"] * step), n_refine)
    for eta in eta_ref:
        for theta in theta_ref:
            L = _loss(eta, theta)
            if L < best["loss"]:
                best = {"loss": L, "eta": float(eta), "theta": float(theta)}

    return best["eta"], best["theta"], best["loss"]



def _fit_grid_two_pass(train_trials, dprime_models_pp, eta_grid, theta_grid,
                        n_repeats_fit, tnd, dt_override_s,
                        eta_lo, eta_hi, theta_lo, theta_hi,
                        rt_tp_sd, rt_ta_sd, dprime_sd):
    """Shared coarse+refine grid search, factored out so both the
    k-fold loop and the final full-data fit call the exact same logic."""
    human_train = compute_human_summary_from_preprocessed(train_trials)

    best = {"loss": np.inf, "eta": None, "theta": None}
    for eta in eta_grid:
        for theta in theta_grid:
            model = simulate_model_summary_from_preprocessed(
                trials_pp=train_trials, dprime_models_pp=dprime_models_pp,
                eta=float(eta), theta=float(theta), n_repeats=n_repeats_fit,
                tnd_s=tnd, dt_override_s=dt_override_s,
            )
            L = loss_summary(human_train, model, rt_tp_sd=rt_tp_sd, rt_ta_sd=rt_ta_sd, dprime_sd=dprime_sd)
            if L < best["loss"]:
                best = {"loss": float(L), "eta": float(eta), "theta": float(theta)}

    #ETA_HALF_WIDTH_REFINE, THETA_HALF_WIDTH_REFINE, N_REFINE = 0.15, 1.0, 12
    ETA_HALF_WIDTH_REFINE, THETA_HALF_WIDTH_REFINE, N_REFINE = 0.072, 0.356, 12   # old: 0.018, 0.089
    eta_ref = make_refined_grid(best["eta"], ETA_HALF_WIDTH_REFINE, N_REFINE, lo=eta_lo, hi=eta_hi)
    #theta_ref = make_refined_grid(best["theta"], THETA_HALF_WIDTH_REFINE, N_REFINE, lo=theta_lo, hi=theta_hi)
    step = (theta_grid[-1] / theta_grid[0]) ** (1.0 / (len(theta_grid) - 1))   # ratio between neighbours
    theta_ref = np.geomspace(max(theta_lo, best["theta"] / step),
                         min(theta_hi, best["theta"] * step), N_REFINE)

    best2 = dict(best)
    for eta in eta_ref:
        for theta in theta_ref:
            model = simulate_model_summary_from_preprocessed(
                trials_pp=train_trials, dprime_models_pp=dprime_models_pp,
                eta=float(eta), theta=float(theta), n_repeats=n_repeats_fit,
                tnd_s=tnd, dt_override_s=dt_override_s,
            )
            L = loss_summary(human_train, model, w_choice=1.0, w_rt=1.0,
                              rt_tp_sd=rt_tp_sd, rt_ta_sd=rt_ta_sd, dprime_sd=dprime_sd)
            if L < best2["loss"]:
                best2 = {"loss": float(L), "eta": float(eta), "theta": float(theta)}
    return best2   # {"loss", "eta", "theta"}


# -----------------------
# 10-fold Fitting Loop
# -----------------------

def fit_model_per_participant_kfold(
    search_dir: str,
    visibility_dir: str,
    eta_grid: np.ndarray,
    theta_grid: np.ndarray,
    k: int = 10,
    n_repeats_fit: int = 10,
    spline_s: float = 0.5,
    out_csv_folds: str = "kfold_cv_results.csv",
    out_csv_final: str = "fitted_params_final.csv",
    eta_lo: float = 0.01, eta_hi: float = 0.30,
    theta_lo: float = 0.10, theta_hi: float = 4.0,
    dt_override_s: Optional[float] = None,
    use_group_visibility: bool = False,
    split_seed_offset: int = 12345,
    rt_tp_sd_override: Optional[float] = None,   
    rt_ta_sd_override: Optional[float] = None,
    dprime_sd_override: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    True k-fold cross-validation per participant, plus a separate final
    fit on each participant's complete dataset.

    - kfold_df: one row per (participant, fold) -- fitted (eta, theta) on
      the other k-1 folds, and held-out loss on the left-out fold. Use
      this to report cross-validated generalization performance
      (e.g. mean +/- SD held-out loss across folds).
    - final_df: one row per participant -- the (eta, theta) fit on their
      FULL dataset (all trials, no held-out portion). These are the
      parameters actually used for Figures 8-10, Table 1, etc.

    RT loss terms are standardized using a pooled log-RT SD computed
    once across ALL participants' full data, matching how d' is already
    expressed in standardized units.
    """
    K_PARAMS = 2
    tnd = float(GLOBAL_TND_S)

    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)

    from visibility import build_group_mean_dprime_splines
    if use_group_visibility:
        group_splines = build_group_mean_dprime_splines(vis_by_pp, spline_s=spline_s)
        dprime_models = {pp: group_splines for pp in vis_by_pp.keys()}
    else:
        dprime_models: Dict[str, Dict[int, UnivariateSpline]] = {}
        for pp, files in vis_by_pp.items():
            base_model = build_dprime_splines_for_participant(files, spline_s=spline_s)
            dprime_models[pp] = make_visibility_null_model(
                base_model, mode=VISIBILITY_MODE, source_speed=VIS_NULL_SOURCE_SPEED,
                constant_rule=VIS_CONSTANT_RULE, ref_ecc_deg=VIS_REF_ECC_DEG,
            )

    # ---- preprocess everyone once, collect full-data human summaries for pooled RT SDs ----
    participant_trials = {}
    all_full_human_summaries = []
    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            print(f"[KFOLD SKIP] {pp}: no visibility splines")
            continue
        trials_pp = preprocess_participant_trials(pp, pairs, dprime_models[pp])
        if len(trials_pp) < k * 2:
            print(f"[KFOLD SKIP] {pp}: too few trials ({len(trials_pp)}) for k={k} folds")
            continue
        participant_trials[pp] = trials_pp
        all_full_human_summaries.append(compute_human_summary_from_preprocessed(trials_pp))

    if rt_tp_sd_override is not None and rt_ta_sd_override is not None:
        rt_tp_sd, rt_ta_sd = rt_tp_sd_override, rt_ta_sd_override
        print(f"[KFOLD] using OVERRIDE pooled log-RT SDs: TP={rt_tp_sd:.4f}  TA={rt_ta_sd:.4f}")
    else:
        rt_tp_sd, rt_ta_sd = compute_global_rt_sds(all_full_human_summaries)
        print(f"[KFOLD] pooled log-RT SDs (full data): TP={rt_tp_sd:.4f}  TA={rt_ta_sd:.4f}")
        
    if dprime_sd_override is None:
        raise ValueError("Set dprime_sd_override (pooled SD of log-linear d'; 0.65 from the free-viewing data).")
    dprime_sd = float(dprime_sd_override)
    print(f"[KFOLD] using d' SD: {dprime_sd:.3f}")


    fold_rows = []
    final_rows = []

    for pp, trials_pp in participant_trials.items():
        # ---- stable shuffle + k-fold split, seeded per participant ----
        rng_split = np.random.default_rng(split_seed_offset + int(pp[2:]))
        idx = rng_split.permutation(len(trials_pp))
        folds = np.array_split(idx, k)

        print(f"[KFOLD] {pp}: {len(trials_pp)} trials -> {k} folds "
              f"(~{len(trials_pp)//k} trials/fold)")

        held_out_losses = []
        for f in range(k):
            test_idx = folds[f]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != f])

            train_trials = [trials_pp[i] for i in train_idx]
            test_trials  = [trials_pp[i] for i in test_idx]

            fit_result = _fit_grid_two_pass(
                train_trials, dprime_models[pp], eta_grid, theta_grid,
                n_repeats_fit, tnd, dt_override_s,
                eta_lo, eta_hi, theta_lo, theta_hi, rt_tp_sd, rt_ta_sd, dprime_sd
            )

            model_test = simulate_model_summary_from_preprocessed(
                trials_pp=test_trials, dprime_models_pp=dprime_models[pp],
                eta=float(fit_result["eta"]), theta=float(fit_result["theta"]),
                n_repeats=n_repeats_fit, tnd_s=tnd, dt_override_s=dt_override_s,
            )
            human_test = compute_human_summary_from_preprocessed(test_trials)
            held_out_loss = loss_summary(human_test, model_test, w_choice=1.0, w_rt=1.0,
                                          rt_tp_sd=rt_tp_sd, rt_ta_sd=rt_ta_sd,
                                          dprime_sd=dprime_sd)
            held_out_losses.append(held_out_loss)

            fold_rows.append({
                "participant": pp, "fold": f,
                "eta_fold": fit_result["eta"], "theta_fold": fit_result["theta"],
                "train_loss": fit_result["loss"], "held_out_loss": held_out_loss,
                "n_train": len(train_trials), "n_test": len(test_trials),
            })
            print(f"[KFOLD] {pp} fold {f+1}/{k}: eta={fit_result['eta']:.3f} "
                  f"theta={fit_result['theta']:.3f} held_out_loss={held_out_loss:.4f}")

        print(f"[KFOLD] {pp}: mean held-out loss = {np.mean(held_out_losses):.4f} "
              f"(SD={np.std(held_out_losses):.4f})")

        # ---- separate final fit on the COMPLETE dataset (all trials) ----
        final_fit = _fit_grid_two_pass(
            trials_pp, dprime_models[pp], eta_grid, theta_grid,
            n_repeats_fit, tnd, dt_override_s,
            eta_lo, eta_hi, theta_lo, theta_hi, rt_tp_sd, rt_ta_sd, dprime_sd
        )
        N_OBS = count_obs(compute_human_summary_from_preprocessed(trials_pp))
        aic, bic = information_criteria(final_fit["loss"], N_OBS, k=K_PARAMS)

        final_rows.append({
            "participant": pp, "eta": final_fit["eta"], "theta": final_fit["theta"],
            "loss": final_fit["loss"], "AIC": aic, "BIC": bic, "tnd": tnd,
            "kfold_mean_held_out_loss": float(np.mean(held_out_losses)),
            "kfold_sd_held_out_loss": float(np.std(held_out_losses)),
        })
        print(f"[KFOLD] {pp}: FINAL full-data fit eta={final_fit['eta']:.3f} "
              f"theta={final_fit['theta']:.3f}")

    kfold_df = pd.DataFrame(fold_rows)
    final_df = pd.DataFrame(final_rows)
    kfold_df.to_csv(out_csv_folds, index=False)
    final_df.to_csv(out_csv_final, index=False)
    print(f"[KFOLD] wrote {out_csv_folds} ({len(kfold_df)} fold-rows) "
          f"and {out_csv_final} ({len(final_df)} participants)")
    return kfold_df, final_df


def get_kfold_assignment(pp, trials_pp, k, split_seed_offset):
    """
    Reproduces the EXACT same k-fold split as fit_model_per_participant_kfold,
    for one participant. Returns a dict mapping trial_seed -> fold index (0..k-1),
    so a replay function can look up which fold each trial belongs to.
    """
    rng_split = np.random.default_rng(split_seed_offset + int(pp[2:]))
    idx = rng_split.permutation(len(trials_pp))
    folds = np.array_split(idx, k)

    trial_seed_to_fold = {}
    for f, fold_idx in enumerate(folds):
        for i in fold_idx:
            trial_seed_to_fold[trials_pp[i]["trial_seed"]] = f
    return trial_seed_to_fold


def run_kfold_pooled_replay(
    search_dir: str,
    visibility_dir: str,
    kfold_df: pd.DataFrame,          # from fit_model_per_participant_kfold
    output_csv: str = "replay_model_results_kfold_pooled.csv",
    k: int = 10,
    n_model_repeats: int = 1000,
    spline_s: float = 0.5,
    max_calib_drift_deg: Optional[float] = None,
    dt_override_s: Optional[float] = None,
    use_group_visibility: bool = False,
    split_seed_offset: int = 12345,
) -> pd.DataFrame:
    """
    Builds POOLED, genuinely out-of-sample predictions: every trial is
    predicted using the (eta, theta) fitted WITHOUT that trial (i.e., from
    the fold that left it out). Every trial in the dataset ends up with
    exactly one prediction, and none of them are in-sample.

    This is what Figures 8-10 / Table S2 should be built from going
    forward, per [supervisor]'s suggestion -- NOT the single final
    full-data fit (which is still used for Table 1 / reported eta,theta).
    """
    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)
    tnd = float(GLOBAL_TND_S)

    from visibility import build_group_mean_dprime_splines
    if use_group_visibility:
        group_splines = build_group_mean_dprime_splines(vis_by_pp, spline_s=spline_s)
        dprime_models = {pp: group_splines for pp in vis_by_pp.keys()}
    else:
        dprime_models: Dict[str, Dict[int, UnivariateSpline]] = {}
        for pp, files in vis_by_pp.items():
            base_model = build_dprime_splines_for_participant(files, spline_s=spline_s)
            dprime_models[pp] = make_visibility_null_model(
                base_model, mode=VISIBILITY_MODE, source_speed=VIS_NULL_SOURCE_SPEED,
                constant_rule=VIS_CONSTANT_RULE, ref_ecc_deg=VIS_REF_ECC_DEG,
            )

    # -- lookup: (participant, fold) -> (eta_fold, theta_fold), from kfold_df --
    fold_params = {
        (row["participant"], int(row["fold"])): (float(row["eta_fold"]), float(row["theta_fold"]))
        for _, row in kfold_df.iterrows()
    }

    all_rows = []
    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            print(f"[KFOLD-REPLAY SKIP] {pp}: no visibility splines")
            continue

        trials_pp = preprocess_participant_trials(pp, pairs, dprime_models[pp])
        if len(trials_pp) < k * 2:
            print(f"[KFOLD-REPLAY SKIP] {pp}: too few trials for k={k}")
            continue

        trial_seed_to_fold = get_kfold_assignment(pp, trials_pp, k, split_seed_offset)
        print(f"[KFOLD-REPLAY] {pp}: {len(trials_pp)} trials assigned across {k} folds")

        for csv_path, asc_path in sorted(pairs):
            df = pd.read_csv(csv_path)
            if max_calib_drift_deg is not None and "CalibrationDrift(deg)" in df.columns:
                df = df[df["CalibrationDrift(deg)"] <= max_calib_drift_deg].copy()

            required = ["Trial", "Gabor Positions", "Target Present", "Response", "Correct", "Reaction Time (s)"]
            for c in required:
                if c not in df.columns:
                    raise ValueError(f"Missing column '{c}' in {csv_path}")

            dt = dt_from_name(csv_path)
            speed_px_s = DT_TO_SPEED[dt]
            #alpha_use = ALPHA_BASE * (1 - ALPHA_SPEED_SLOPE * (speed_px_s / 400))
            #raw_speed = float(df["Speed (px/s)"].iloc[0]) if "Speed (px/s)" in df.columns else float(speed_px_s)

            if speed_px_s not in dprime_models[pp]:
                continue

            (screen_w, screen_h), eye_trials = parse_asc_events(asc_path)
            df["gabor_pos"] = df["Gabor Positions"].apply(safe_parse_positions)

            for _, r in df.iterrows():
                trial = safe_int(r.get("Trial"))
                target_present = safe_int(r.get("Target Present"))
                human_resp = safe_int(r.get("Response"))
                human_corr = safe_int(r.get("Correct"))
                human_rt = safe_float(r.get("Reaction Time (s)"))
                if None in (trial, target_present, human_resp, human_corr, human_rt):
                    continue

                obj = r.get("gabor_pos")
                if obj is None or not isinstance(obj, np.ndarray) or obj.ndim != 3:
                    continue
                Tstim = obj.shape[0]
                if Tstim < 2:
                    continue

                trial_seed = stable_trial_seed(pp, os.path.basename(csv_path), int(trial))
                if trial_seed not in trial_seed_to_fold:
                    continue   # shouldn't happen, but guard against mismatch
                fold = trial_seed_to_fold[trial_seed]
                eta_pp, theta_pp = fold_params[(pp, fold)]   # <-- this trial's OUT-OF-FOLD params

                dt_s = estimate_dt_from_positions(obj, speed_px_s) if speed_px_s > 0 else None
                if dt_s is None or dt_s <= 0:
                    dt_s = dt_from_duration(obj, 3.5)

                tr_eye = eye_trials.get(trial)
                gaze_cells = (np.full((Tstim, 2), np.nan, dtype=float) if tr_eye is None
                              else gaze_series_cells_from_fixations(tr_eye, Tstim, dt_s, screen_w, screen_h))

                rng_trial = np.random.default_rng(trial_seed)
                gaze_cells_used = perturb_gaze(gaze_cells, mode=GAZE_MODE, rng=rng_trial,
                                                dt_s=dt_override_s if dt_override_s is not None else dt_s)

                target_index = None
                if target_present == 1:
                    traj = safe_parse_trajectory(r.get("Target Trajectory"))
                    if traj is not None:
                        target_index = infer_target_index(obj, traj)

                model_resp = np.empty(n_model_repeats, dtype=int)
                model_rt = np.empty(n_model_repeats, dtype=float)
                ss = np.random.SeedSequence(trial_seed)
                child_seeds = ss.spawn(n_model_repeats)
                for k_i in range(n_model_repeats):
                    rng_k = np.random.default_rng(child_seeds[k_i])
                    resp, rt = run_replay_trial(
                        obj_xy_cells=obj, gaze_xy_cells=gaze_cells_used, speed_px_s=speed_px_s,
                        dprime_splines=dprime_models[pp], dt_s=dt_s, eta=eta_pp,
                        decision_theta_present=float(theta_pp), target_present=target_present,
                        target_index=target_index, rng=rng_k,
                        dt_override_s=dt_override_s,
                    )
                    model_resp[k_i] = resp
                    model_rt[k_i] = rt + tnd

                present_mask = (model_resp == 1)
                absent_mask = (model_resp == 0)

                all_rows.append({
                    "participant": pp, "fold": fold,   # <-- which fold this trial was held out from
                    "speed_px_s_used": speed_px_s, "trial": trial,
                    "human_target_present": target_present, "human_response": human_resp,
                    "human_correct": human_corr, "human_rt_s": float(human_rt),
                    "model_p_present": float(model_resp.mean()),
                    "model_rt_present_mean_s": float(np.mean(model_rt[present_mask])) if present_mask.any() else np.nan,
                    "model_rt_absent_mean_s": float(np.mean(model_rt[absent_mask])) if absent_mask.any() else np.nan,
                    "eta_used": eta_pp, "theta_used": theta_pp,
                })

        print(f"[KFOLD-REPLAY] {pp}: done")

    out = pd.DataFrame(all_rows)
    out.to_csv(output_csv, index=False)
    print(f"[KFOLD-REPLAY] wrote {output_csv}, {len(out)} rows (pooled, out-of-fold predictions)")
    return out







# ----------------------------------------------------------
# Everything below is code for debug runs (model recovery and visibility null model comparisons) 
# ----------------------------------------------------------

def information_criteria(loss, n_obs, k):
    """
    Computes AIC and BIC from SSE-like loss.
    """
    if loss <= 0:
        loss = 1e-12

    logL = -0.5 * n_obs * np.log(loss / n_obs)
    AIC = 2 * k - 2 * logL
    BIC = k * np.log(n_obs) - 2 * logL

    return AIC, BIC


def fit_eta_theta_weighted(pp, trials_pp, dprime_models_pp, eta_grid, theta_grid,
                            w_choice=1.0, w_rt=1.0, n_repeats_fit=10, tnd_s=GLOBAL_TND_S,
                            rt_tp_sd=1.0, rt_ta_sd=1.0, dprime_sd=1.0):   
    rng_split = np.random.default_rng(12345 + int(pp[2:]))
    idx = rng_split.permutation(len(trials_pp))
    n_train = int(0.7 * len(idx))
    train_trials = [trials_pp[i] for i in idx[:n_train]]
    human_train = compute_human_summary_from_preprocessed(train_trials)
    kin_cache = {i: precompute_trial_kinematics(tr, dprime_models_pp)
                 for i, tr in enumerate(train_trials)}
    best = {"loss": np.inf, "eta": None, "theta": None}
    for eta in eta_grid:
        for theta in theta_grid:
            model = simulate_model_summary_from_preprocessed_fast(
                trials_pp=train_trials, dprime_models_pp=dprime_models_pp,
                eta=float(eta), theta=float(theta), n_repeats=n_repeats_fit,
                tnd_s=tnd_s, kin_cache=kin_cache,
            )
            L = loss_summary(human_train, model, w_choice=w_choice, w_rt=w_rt,
                              rt_tp_sd=rt_tp_sd, rt_ta_sd=rt_ta_sd)   # <-- pass through
            if L < best["loss"]:
                best = {"loss": float(L), "eta": float(eta), "theta": float(theta)}
    return best["eta"], best["theta"], best["loss"]


def compute_loss_surface(pp, synth_trials, dprime_models_pp, eta_grid, theta_grid,
                          n_repeats=15, tnd_s=GLOBAL_TND_S):
    """
    Computes the full loss surface over (eta_grid x theta_grid) for one
    synthetic dataset, instead of just returning the argmin. Reuses the
    same train split and kinematics cache as the fitter for consistency.
    Returns loss_grid of shape (len(eta_grid), len(theta_grid)).
    """
    rng_split = np.random.default_rng(12345 + int(pp[2:]))
    idx = rng_split.permutation(len(synth_trials))
    n_train = int(0.7 * len(idx))
    train_trials = [synth_trials[i] for i in idx[:n_train]]
    human_train = compute_human_summary_from_preprocessed(train_trials)

    kin_cache = {i: precompute_trial_kinematics(tr, dprime_models_pp)
                 for i, tr in enumerate(train_trials)}

    loss_grid = np.full((len(eta_grid), len(theta_grid)), np.nan)
    for i, eta in enumerate(eta_grid):
        for j, theta in enumerate(theta_grid):
            model = simulate_model_summary_from_preprocessed_fast(
                trials_pp=train_trials, dprime_models_pp=dprime_models_pp,
                eta=float(eta), theta=float(theta), n_repeats=n_repeats,
                tnd_s=tnd_s, kin_cache=kin_cache,
            )
            loss_grid[i, j] = loss_summary(human_train, model)
    return loss_grid


def run_loss_surface_batch(search_dir, visibility_dir, participants, conditions,
                            eta_grid, theta_grid, n_repeats=15, spline_s=0.5,
                            out_npz="loss_surfaces.npz"):
    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(visibility_dir if False else search_dir)
    dprime_models = {pp: build_dprime_splines_for_participant(vis_by_pp[pp], spline_s=spline_s)
                      for pp in participants if pp in vis_by_pp}

    results = {}
    for pp in participants:
        if pp not in dprime_models or pp not in search_by_pp:
            print(f"[SKIP] {pp}: missing data")
            continue
        trials_real = preprocess_participant_trials(pp, search_by_pp[pp], dprime_models[pp])
        if len(trials_real) < 10:
            continue
        for (eta_true, theta_true) in conditions:
            synth_trials = make_synthetic_human_trials(
                trials_pp=trials_real, dprime_models_pp=dprime_models[pp],
                eta_true=float(eta_true), theta_true=float(theta_true), tnd_s=GLOBAL_TND_S,
            )
            print(f"[SURFACE] {pp} eta_true={eta_true} theta_true={theta_true} ...")
            loss_grid = compute_loss_surface(
                pp=pp, synth_trials=synth_trials, dprime_models_pp=dprime_models[pp],
                eta_grid=eta_grid, theta_grid=theta_grid, n_repeats=n_repeats,
            )
            key = f"{pp}_eta{eta_true}_theta{theta_true}"
            results[key] = loss_grid

    np.savez(out_npz, eta_grid=eta_grid, theta_grid=theta_grid, **results)
    print(f"[SURFACE] wrote {out_npz} with {len(results)} surfaces")
    return results

def make_synthetic_human_trials(
    trials_pp: List[dict],
    dprime_models_pp: Dict[int, UnivariateSpline],
    eta_true: float,
    theta_true: float,
    tnd_s: float = GLOBAL_TND_S,
    dt_override_s: Optional[float] = None,
) -> List[dict]:
    """
    Returns a copy of trials_pp where the human_resp/human_rt/human_corr fields
    are replaced with simulated responses from the model at (eta_true, theta_true)
    for purposes of model recovery analysis. 
    """
    out = []
    for tr in trials_pp:
        tr2 = dict(tr)  # shallow copy

        # one RNG per trial (stable)
        rng = np.random.default_rng(tr["trial_seed"])

        resp, rt = run_replay_trial(
            obj_xy_cells=tr["obj"],
            gaze_xy_cells=tr["gaze_cells"],
            speed_px_s=tr["speed_px_s"],
            dprime_splines=dprime_models_pp,
            dt_s = tr["dt_s"],          
            dt_override_s = dt_override_s,  
            eta=float(eta_true),
            decision_theta_present=float(theta_true),
            target_present=tr["target_present"],
            target_index=tr["target_index"],
            #alpha_search=tr["alpha_trial"],
            rng=rng,
        )

        rt = float(rt + tnd_s)

        tr2["human_resp"] = int(resp)
        tr2["human_rt"] = float(rt)

        # correctness wrt ground truth (target_present)
        tr2["human_corr"] = int((resp == 1 and tr["target_present"] == 1) or (resp == 0 and tr["target_present"] == 0))

        out.append(tr2)

    return out


def count_obs(human_by_speed):
    n = 0
    for s in SPEED_ORDER:
        h = human_by_speed[s]
        for key in ["hit", "fa", "rt_tp", "rt_ta"]:
            if not np.isnan(h[key]):
                n += 1
    return n



# -----------------------------------------
# Functions for model recovery analysis run 
# -----------------------------------------
def run_model_recovery(
    search_dir: str,
    visibility_dir: str,
    eta_true_grid: np.ndarray,
    theta_true_grid: np.ndarray,
    eta_fit_grid: np.ndarray,
    theta_fit_grid: np.ndarray,
    n_repeats_fit: int = 10,
    spline_s: float = 0.5,
    out_csv: str = "model_recovery.csv",
    rt_tp_sd: float = 0.2450,
    rt_ta_sd: float = 0.3569,
    dprime_sd: float = 0.694,
    dt_override_s: Optional[float] = None,      # <-- new
) -> pd.DataFrame:

    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)

    dprime_models = {pp: build_dprime_splines_for_participant(files, spline_s=spline_s)
                     for pp, files in vis_by_pp.items()}

    rows = []

    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            continue

        trials_real = preprocess_participant_trials(pp, pairs, dprime_models[pp])
        if len(trials_real) < 10:
            continue

        for eta_true in eta_true_grid:
            for theta_true in theta_true_grid:

                synth_trials = make_synthetic_human_trials(
                    trials_pp=trials_real,
                    dprime_models_pp=dprime_models[pp],
                    eta_true=float(eta_true),
                    theta_true=float(theta_true),
                    tnd_s=GLOBAL_TND_S,
                    dt_override_s=dt_override_s,            # <-- new
                )

                eta_hat, theta_hat, loss = fit_eta_theta_from_trials_fast(
                    pp=pp,
                    trials_pp=synth_trials,
                    dprime_models_pp=dprime_models[pp],
                    eta_grid=eta_fit_grid,
                    theta_grid=theta_fit_grid,
                    n_repeats_fit=n_repeats_fit,
                    tnd_s=GLOBAL_TND_S,
                    rt_tp_sd=rt_tp_sd,                      # <-- new
                    rt_ta_sd=rt_ta_sd,                      # <-- new
                    dprime_sd=dprime_sd,                    # <-- new
                    dt_override_s=dt_override_s,            # <-- new
                )

                rows.append(dict(
                    participant=pp,
                    eta_true=float(eta_true),
                    theta_true=float(theta_true),
                    eta_hat=float(eta_hat),
                    theta_hat=float(theta_hat),
                    train_loss=float(loss),
                    n_trials=len(synth_trials),
                ))

        print(f"[RECOVERY] done {pp}")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[RECOVERY] wrote {out_csv} rows={len(df)}")
    return df


def compute_human_summary_from_preprocessed(trials_subset: List[dict]) -> Dict[int, Dict[str, float]]:
    by_speed = {s: {"hit": np.nan, "fa": np.nan, "rt_tp": np.nan, "rt_ta": np.nan} for s in SPEED_ORDER}
    tmp = {s: {"tp_resp": [], "ta_resp": [], "tp_rt": [], "ta_rt": []} for s in SPEED_ORDER}
    by_speed = {s: {"hit": np.nan, "fa": np.nan, "rt_tp": np.nan, "rt_ta": np.nan,
                "n_pos": 0, "n_neg": 0} for s in SPEED_ORDER}

    for tr in trials_subset:
        # skip trials with missing behavioral data
        if (tr.get("human_resp") is None) or (tr.get("human_corr") is None) or (tr.get("human_rt") is None):
            continue

        s = tr["speed_px_s"]
        if s not in tmp:
            continue

        tp = int(tr["target_present"]) # 1=present
        resp = int(tr["human_resp"])  # 1=present
        corr = int(tr["human_corr"]) # 1=correct
        rt = float(tr["human_rt"])

        if tp == 1:
            tmp[s]["tp_resp"].append(resp)
            if corr == 1:
                tmp[s]["tp_rt"].append(rt)
        else:
            tmp[s]["ta_resp"].append(resp)
            if corr == 1:
                tmp[s]["ta_rt"].append(rt)

    for s in SPEED_ORDER:
        tp_resp = np.asarray(tmp[s]["tp_resp"], float)
        ta_resp = np.asarray(tmp[s]["ta_resp"], float)
        tp_rt   = np.asarray(tmp[s]["tp_rt"], float)
        ta_rt   = np.asarray(tmp[s]["ta_rt"], float)

        by_speed[s]["hit"]   = float(np.mean(tp_resp)) if tp_resp.size else np.nan
        by_speed[s]["fa"]    = float(np.mean(ta_resp)) if ta_resp.size else np.nan
        by_speed[s]["rt_tp"] = float(np.median(tp_rt))  if tp_rt.size   else np.nan
        by_speed[s]["rt_ta"] = float(np.median(ta_rt))  if ta_rt.size   else np.nan
        by_speed[s]["n_pos"] = int(tp_resp.size)
        by_speed[s]["n_neg"] = int(ta_resp.size)

    return by_speed


def simulate_model_summary_from_preprocessed(
    trials_pp: List[dict],
    dprime_models_pp: Dict[int, UnivariateSpline],
    eta: float,
    theta: float,
    n_repeats: int = 10,
    tnd_s: float = GLOBAL_TND_S,
    dt_override_s: Optional[float] = None,
) -> Dict[int, Dict[str, float]]:

    by_speed = {s: {"hit": np.nan, "fa": np.nan, "rt_tp": np.nan, "rt_ta": np.nan} for s in SPEED_ORDER}
    tmp = {s: {"tp_p": [], "ta_p": [], "tp_rt": [], "ta_rt": []} for s in SPEED_ORDER}

    for tr in trials_pp:
        speed_px_s = tr["speed_px_s"]
        if speed_px_s not in tmp:
            continue

        ss = SeedSequence(tr["trial_seed"])
        child = ss.spawn(n_repeats)

        model_resp = np.empty(n_repeats, dtype=int)
        model_rt = np.empty(n_repeats, dtype=float)

        for k in range(n_repeats):
            rng_k = np.random.default_rng(child[k])

            resp, rt = run_replay_trial(
                obj_xy_cells=tr["obj"],
                gaze_xy_cells=tr["gaze_cells"],
                speed_px_s=speed_px_s,
                dprime_splines=dprime_models_pp,
                dt_s=tr["dt_s"],
                eta=eta,
                decision_theta_present=theta,
                target_present=tr["target_present"],
                target_index=tr["target_index"],
                #alpha_search=tr["alpha_trial"],
                rng=rng_k,
                dt_override_s=dt_override_s,
            )

            model_resp[k] = resp
            model_rt[k] = rt + tnd_s

        p_present = float(np.mean(model_resp))
        present_mask = (model_resp == 1)
        absent_mask = (model_resp == 0)

        rt_present = float(np.mean(model_rt[present_mask])) if present_mask.any() else np.nan
        rt_absent = float(np.mean(model_rt[absent_mask])) if absent_mask.any() else np.nan

        if tr["target_present"] == 1:
            tmp[speed_px_s]["tp_p"].append(p_present)
            if not np.isnan(rt_present):
                tmp[speed_px_s]["tp_rt"].append(rt_present)
        else:
            tmp[speed_px_s]["ta_p"].append(p_present)
            if not np.isnan(rt_absent):
                tmp[speed_px_s]["ta_rt"].append(rt_absent)

    for s in SPEED_ORDER:
        tp_p = np.asarray(tmp[s]["tp_p"], float)
        ta_p = np.asarray(tmp[s]["ta_p"], float)
        tp_rt = np.asarray(tmp[s]["tp_rt"], float)
        ta_rt = np.asarray(tmp[s]["ta_rt"], float)

        by_speed[s]["hit"] = float(np.mean(tp_p)) if tp_p.size else np.nan
        by_speed[s]["fa"] = float(np.mean(ta_p)) if ta_p.size else np.nan
        by_speed[s]["rt_tp"] = float(np.median(tp_rt)) if tp_rt.size else np.nan
        by_speed[s]["rt_ta"] = float(np.median(ta_rt)) if ta_rt.size else np.nan

    return by_speed

def run_replay_trial_batch(kin, eta, decision_theta_present, target_present,
                            target_index, n_repeats, rng):
    """
    Simulates n_repeats replicates of ONE trial simultaneously, reusing a
    precomputed dstep. Returns (resp, rt) arrays of shape (n_repeats,).
    Numerically equivalent to calling run_replay_trial n_repeats times,
    but without the Python-level per-timestep, per-repeat loop.
    """
    dstep = kin["dstep"]                       # (T_use, N)
    dt_eff = kin["dt_eff"]
    T_use, N = dstep.shape

    theta_pos = float(decision_theta_present)
    theta_neg = -theta_pos
    noise_scale = np.sqrt(max(1e-6, float(eta)))

    x = rng.normal(0.0, noise_scale, size=(n_repeats, T_use, N))
    if target_present == 1 and target_index is not None:
        x[:, :, target_index] += dstep[np.newaxis, :, target_index]

    dllr = dstep[np.newaxis] * x - 0.5 * (dstep[np.newaxis] ** 2)
    logLR = np.cumsum(dllr, axis=1)             # (n_repeats, T_use, N)
    dv = np.max(logLR, axis=2)                  # (n_repeats, T_use)

    pos_cross = dv >= theta_pos
    neg_cross = dv <= theta_neg
    any_cross = pos_cross | neg_cross
    has_crossing = any_cross.any(axis=1)
    first_idx = np.argmax(any_cross, axis=1)     # 0 if no crossing (masked below)
    pos_at_first = np.take_along_axis(pos_cross, first_idx[:, None], axis=1)[:, 0]
    resp = np.where(has_crossing, pos_at_first.astype(int), (dv[:, -1] > TIMEOUT_DV_THRESHOLD).astype(int))
    #resp = np.where(has_crossing, pos_at_first.astype(int), (dv[:, -1] > 0).astype(int))
    stop_idx = np.where(has_crossing, first_idx, T_use - 1)
    rt = (stop_idx + 1) * dt_eff

    return resp.astype(int), rt.astype(float)

def precompute_trial_kinematics(tr, dprime_splines, max_time_s=3.5, dt_override_s=None):
    """
    Precomputes dstep (the deterministic per-object, per-timestep sensory
    drift) once per trial. dstep does NOT depend on eta or theta, so this
    result can be reused across the entire (eta, theta) grid search and
    across all repeats -- this is the main cost that was being redundantly
    recomputed before.
    """
    obj_xy_cells = tr["obj"]
    gaze_xy_cells = tr["gaze_cells"]
    speed_px_s = tr["speed_px_s"]
    dt_s = tr["dt_s"]
    #alpha_search = tr["alpha_trial"]

    T, N, _ = obj_xy_cells.shape
    if dt_s <= 0:
        dt_s = max_time_s / max(1, T - 1)
    dt_eff = dt_s if dt_override_s is None else float(dt_override_s)
    T_use = min(T, int(np.floor(max_time_s / dt_s)))
    if T_use <= 0:
        return None
    if speed_px_s not in dprime_splines:
        raise ValueError(f"Missing d' spline for speed_px_s={speed_px_s}")
    spl = dprime_splines[speed_px_s]

    dstep = np.empty((T_use, N), dtype=float)
    for t in range(T_use):
        gaze = gaze_xy_cells[t]
        if np.any(np.isnan(gaze)):
            gaze = np.array([GRID_SIZE_X / 2.0, GRID_SIZE_Y / 2.0], dtype=float)
        ecc_deg = ecc_deg_from_cells(obj_xy_cells[t], gaze)
        #d400 = np.maximum(alpha_search * spl(ecc_deg), 0.0)
        #dstep[t] = 0.4 * d400 * np.sqrt(dt_eff / 0.4)
        d400 = np.maximum(spl(ecc_deg), 0.0)
        dstep[t] = T_VIS_S * d400 * np.sqrt(dt_eff / T_VIS_S)
        assert dstep.shape == (T_use, N), dstep.shape

    return {"dstep": dstep, "dt_eff": dt_eff, "T_use": T_use, "N": N}

def simulate_model_summary_from_preprocessed_fast(
    trials_pp, dprime_models_pp, eta, theta, n_repeats=10,
    tnd_s=GLOBAL_TND_S, dt_override_s=None, kin_cache=None,
):
    by_speed = {s: {"hit": np.nan, "fa": np.nan, "rt_tp": np.nan, "rt_ta": np.nan} for s in SPEED_ORDER}
    tmp = {s: {"tp_p": [], "ta_p": [], "tp_rt": [], "ta_rt": []} for s in SPEED_ORDER}

    for i, tr in enumerate(trials_pp):
        speed_px_s = tr["speed_px_s"]
        if speed_px_s not in tmp:
            continue
        if kin_cache is not None and i in kin_cache:
            kin = kin_cache[i]
        else:
            kin = precompute_trial_kinematics(tr, dprime_models_pp, dt_override_s=dt_override_s)
            if kin_cache is not None:
                kin_cache[i] = kin
        if kin is None:
            continue

        rng = np.random.default_rng(np.random.SeedSequence(tr["trial_seed"]))
        model_resp, model_rt = run_replay_trial_batch(
            kin=kin, eta=eta, decision_theta_present=theta,
            target_present=tr["target_present"], target_index=tr["target_index"],
            n_repeats=n_repeats, rng=rng,
        )
        model_rt = model_rt + tnd_s

        p_present = float(np.mean(model_resp))
        present_mask, absent_mask = model_resp == 1, model_resp == 0
        rt_present = float(np.mean(model_rt[present_mask])) if present_mask.any() else np.nan
        rt_absent = float(np.mean(model_rt[absent_mask])) if absent_mask.any() else np.nan

        bucket = tmp[speed_px_s]
        if tr["target_present"] == 1:
            bucket["tp_p"].append(p_present)
            if not np.isnan(rt_present): bucket["tp_rt"].append(rt_present)
        else:
            bucket["ta_p"].append(p_present)
            if not np.isnan(rt_absent): bucket["ta_rt"].append(rt_absent)

    for s in SPEED_ORDER:
        tp_p, ta_p = np.asarray(tmp[s]["tp_p"]), np.asarray(tmp[s]["ta_p"])
        tp_rt, ta_rt = np.asarray(tmp[s]["tp_rt"]), np.asarray(tmp[s]["ta_rt"])
        by_speed[s]["hit"] = float(np.mean(tp_p)) if tp_p.size else np.nan
        by_speed[s]["fa"] = float(np.mean(ta_p)) if ta_p.size else np.nan
        by_speed[s]["rt_tp"] = float(np.median(tp_rt)) if tp_rt.size else np.nan
        by_speed[s]["rt_ta"] = float(np.median(ta_rt)) if ta_rt.size else np.nan
    return by_speed


# ---------------------------------------------------------------------------
# shared helper: coarse grid + refinement, same logic as the main fit
# ---------------------------------------------------------------------------
def _comparison_grid_search(
    trials,
    dprime_models_pp,
    human,
    *,
    eta_grid,
    theta_grid,
    eta_lo: float,
    eta_hi: float,
    theta_lo: float,
    theta_hi: float,
    n_repeats_fit: int,
    tnd: float,
    dt_override_s,
    rt_tp_sd: float,
    rt_ta_sd: float,
    dprime_sd: float,
    eta_half_width_refine: float = 0.072,
    n_refine: int = 12,
):
    """
    Coarse grid, then a refinement pass around the winner.
    An axis with a single value (a fixed parameter) is not refined.
    Returns dict(loss, eta, theta).
    """
    eta_grid = np.atleast_1d(np.asarray(eta_grid, dtype=float))
    theta_grid = np.atleast_1d(np.asarray(theta_grid, dtype=float))

    def _loss(eta, theta):
        model = simulate_model_summary_from_preprocessed(
            trials_pp=trials,
            dprime_models_pp=dprime_models_pp,
            eta=float(eta),
            theta=float(theta),
            n_repeats=n_repeats_fit,
            tnd_s=tnd,
            dt_override_s=dt_override_s,
        )
        return loss_summary(human, model, rt_tp_sd=rt_tp_sd, rt_ta_sd=rt_ta_sd, dprime_sd=dprime_sd)

    best = {"loss": np.inf, "eta": float(eta_grid[0]), "theta": float(theta_grid[0])}

    # coarse pass
    for eta in eta_grid:
        for theta in theta_grid:
            L = _loss(eta, theta)
            if L < best["loss"]:
                best = {"loss": L, "eta": float(eta), "theta": float(theta)}

    # refinement pass (only on free axes)
    if len(eta_grid) > 1:
        eta_ref = make_refined_grid(best["eta"], eta_half_width_refine, n_refine, lo=eta_lo, hi=eta_hi)
    else:
        eta_ref = eta_grid

    if len(theta_grid) > 1:
        step = (theta_grid[-1] / theta_grid[0]) ** (1.0 / (len(theta_grid) - 1))
        theta_ref = np.geomspace(
            max(theta_lo, best["theta"] / step),
            min(theta_hi, best["theta"] * step),
            n_refine,
        )
    else:
        theta_ref = theta_grid

    for eta in eta_ref:
        for theta in theta_ref:
            L = _loss(eta, theta)
            if L < best["loss"]:
                best = {"loss": L, "eta": float(eta), "theta": float(theta)}

    return best


# ---------------------------------------------------------------------------
# parameter null models: eta+theta vs eta-only vs theta-only
# ---------------------------------------------------------------------------
def fit_model_comparison_per_participant(
    search_dir: str,
    visibility_dir: str,
    eta_grid: np.ndarray,
    theta_grid: np.ndarray,
    eta_fixed: float,           # used when eta is NOT free (theta-only model)
    theta_fixed: float,         # used when theta is NOT free (eta-only model)
    n_repeats_fit: int = 10,
    spline_s: float = 0.5,
    out_csv: str = "model_comparison.csv",
    eta_lo: float = 0.04,
    eta_hi: float = 1.20,
    theta_lo: float = 0.08,
    theta_hi: float = 16.0,
    dt_override_s: Optional[float] = None,
    rt_tp_sd: float = 1.0,
    rt_ta_sd: float = 1.0,
    dprime_sd: float = 1.0,
) -> pd.DataFrame:
    """
    Fits three nested models per participant and compares via AIC/BIC:
      M2: eta + theta free  (k=2)
      M_eta: eta free, theta fixed  (k=1)
      M_theta: theta free, eta fixed  (k=1)

    eta_fixed / theta_fixed are the held values for the restricted models
    (group means from the full fit of the SAME experiment, on the same scale).
    All models use the same coarse grid + refinement as the main fit.
    """
    if VISIBILITY_MODE != "empirical":
        print(f"[COMPARE WARNING] config.VISIBILITY_MODE = {VISIBILITY_MODE!r}, expected 'empirical'")

    tnd = float(GLOBAL_TND_S)

    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)
    #search_by_pp = {k: v for k, v in search_by_pp.items() if k in ("kh1", "kh2")}   # TEMPORARY

    dprime_models: Dict[str, Dict[int, UnivariateSpline]] = {}
    for pp, files in vis_by_pp.items():
        base = build_dprime_splines_for_participant(files, spline_s=spline_s)
        dprime_models[pp] = make_visibility_null_model(
            base,
            mode=VISIBILITY_MODE,
            source_speed=VIS_NULL_SOURCE_SPEED,
            constant_rule=VIS_CONSTANT_RULE,
            ref_ecc_deg=VIS_REF_ECC_DEG,
        )

    common = dict(
        eta_lo=eta_lo, eta_hi=eta_hi, theta_lo=theta_lo, theta_hi=theta_hi,
        n_repeats_fit=n_repeats_fit, tnd=tnd, dt_override_s=dt_override_s,
        rt_tp_sd=rt_tp_sd, rt_ta_sd=rt_ta_sd, dprime_sd=dprime_sd,
    )

    rows = []

    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            print(f"[COMPARE SKIP] {pp}: no visibility splines")
            continue

        trials_pp = preprocess_participant_trials(pp, pairs, dprime_models[pp])
        if len(trials_pp) < 10:
            print(f"[COMPARE SKIP] {pp}: too few trials ({len(trials_pp)})")
            continue

        # identical train split to main fit - same seed logic
        rng_split = np.random.default_rng(12345 + int(pp[2:]))
        idx = rng_split.permutation(len(trials_pp))
        n_train = int(0.7 * len(idx))
        train_trials = [trials_pp[i] for i in idx[:n_train]]
        test_trials = [trials_pp[i] for i in idx[n_train:]]

        human_train = compute_human_summary_from_preprocessed(train_trials)
        human_test = compute_human_summary_from_preprocessed(test_trials)
        N_OBS = count_obs(human_train)   # identical for all three models

        def _fit(etas, thetas):
            b = _comparison_grid_search(
                train_trials, dprime_models[pp], human_train,
                eta_grid=etas, theta_grid=thetas, **common,
            )
            return b["eta"], b["theta"], b["loss"]

        def _test_loss(eta, theta):
            model = simulate_model_summary_from_preprocessed(
                trials_pp=test_trials,
                dprime_models_pp=dprime_models[pp],
                eta=float(eta),
                theta=float(theta),
                n_repeats=n_repeats_fit,
                tnd_s=tnd,
                dt_override_s=dt_override_s,
            )
            return loss_summary(human_test, model, rt_tp_sd=rt_tp_sd, rt_ta_sd=rt_ta_sd, dprime_sd=dprime_sd)

        # ---- M2: both free (k=2) ----
        e2, t2, L2 = _fit(eta_grid, theta_grid)
        aic2, bic2 = information_criteria(L2, N_OBS, k=2)
        tl2 = _test_loss(e2, t2)

        # ---- M_eta: only eta free, theta pinned (k=1) ----
        e_eta, _, L_eta = _fit(eta_grid, [theta_fixed])
        aic_eta, bic_eta = information_criteria(L_eta, N_OBS, k=1)
        tl_eta = _test_loss(e_eta, theta_fixed)

        # ---- M_theta: only theta free, eta pinned (k=1) ----
        _, t_th, L_th = _fit([eta_fixed], theta_grid)
        aic_th, bic_th = information_criteria(L_th, N_OBS, k=1)
        tl_th = _test_loss(eta_fixed, t_th)

        print(
            f"[COMPARE] {pp} | "
            f"M2: eta={e2:.3f} theta={t2:.3f} AIC={aic2:.2f} | "
            f"M_eta: eta={e_eta:.3f} AIC={aic_eta:.2f} | "
            f"M_theta: theta={t_th:.3f} AIC={aic_th:.2f}"
        )

        rows.append(dict(
            participant=pp,
            # --- M2 ---
            m2_eta=e2,        m2_theta=t2,
            m2_train_loss=L2, m2_test_loss=tl2,
            m2_AIC=aic2,      m2_BIC=bic2,
            # --- M_eta (theta fixed) ---
            meta_eta=e_eta,         meta_theta_fixed=theta_fixed,
            meta_train_loss=L_eta,  meta_test_loss=tl_eta,
            meta_AIC=aic_eta,       meta_BIC=bic_eta,
            # --- M_theta (eta fixed) ---
            mtheta_eta_fixed=eta_fixed, mtheta_theta=t_th,
            mtheta_train_loss=L_th,     mtheta_test_loss=tl_th,
            mtheta_AIC=aic_th,          mtheta_BIC=bic_th,
            # --- delta AIC (positive = M2 wins) ---
            delta_AIC_vs_eta=aic_eta - aic2,
            delta_AIC_vs_theta=aic_th - aic2,
            N_obs=N_OBS,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[COMPARE] wrote {out_csv}  n_pp={len(df)}")
    return df


# ---------------------------------------------------------------------------
# visibility null models
# ---------------------------------------------------------------------------
def fit_model_comparison_visibility(
    search_dir: str,
    visibility_dir: str,
    eta_grid: np.ndarray,
    theta_grid: np.ndarray,
    n_repeats_fit: int = 10,
    spline_s: float = 0.5,
    out_csv: str = "model_comparison_visibility.csv",
    null_mode: str = "constant",        # "group", "constant" or "mean_across_speeds"
    eta_lo: float = 0.04,
    eta_hi: float = 1.20,
    theta_lo: float = 0.08,
    theta_hi: float = 16.0,
    dt_override_s: Optional[float] = None,
    rt_tp_sd: float = 1.0,
    rt_ta_sd: float = 1.0,
    dprime_sd: float = 1.0,
) -> pd.DataFrame:
    """
    Compares two visibility models per participant:
      V_indiv : individual empirical d' splines (the current model)
      V_null  : null visibility, chosen by null_mode:
                "group"              shared group-mean splines per speed
                "constant"           no eccentricity dependence
                "mean_across_speeds" individual splines averaged over speeds
                                     (speed-independent visibility)

    eta and theta are re-fit under each visibility model, so any difference
    in loss/AIC reflects the information carried by the visibility model,
    not a parameter-count difference.
    """
    if VISIBILITY_MODE != "empirical":
        print(f"[VIS COMPARE WARNING] config.VISIBILITY_MODE = {VISIBILITY_MODE!r}, expected 'empirical'")

    tnd = float(GLOBAL_TND_S)
    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)

    # --- individual (empirical) splines ---
    dprime_models: Dict[str, Dict[int, UnivariateSpline]] = {}
    for pp, files in vis_by_pp.items():
        base = build_dprime_splines_for_participant(files, spline_s=spline_s)
        dprime_models[pp] = make_visibility_null_model(
            base,
            mode=VISIBILITY_MODE,
            source_speed=VIS_NULL_SOURCE_SPEED,
            constant_rule=VIS_CONSTANT_RULE,
            ref_ecc_deg=VIS_REF_ECC_DEG,
        )

    # --- null model depending on mode ---
    null_models: Dict[str, Dict[int, callable]] = {}

    if null_mode == "group":
        group_splines = build_group_mean_dprime_splines(vis_by_pp, spline_s=spline_s)
        print(f"[VIS COMPARE] null mode: group-mean splines for speeds: {sorted(group_splines.keys())}")
        for pp in vis_by_pp.keys():
            null_models[pp] = group_splines

    elif null_mode == "constant":
        for pp, files in vis_by_pp.items():
            base = build_dprime_splines_for_participant(files, spline_s=spline_s)
            null_models[pp] = make_visibility_null_model(
                base,
                mode="constant_from_speed0",
                constant_rule="mean",
            )

    elif null_mode == "mean_across_speeds":
        print("[VIS COMPARE] null mode: individual splines averaged across speeds")
        for pp, files in vis_by_pp.items():
            base = build_dprime_splines_for_participant(files, spline_s=spline_s)
            null_models[pp] = make_visibility_null_model(
                base,
                mode="mean_across_speeds",
            )

    else:
        raise ValueError(
            f"Unknown null_mode: {null_mode!r}. Use 'group', 'constant' or 'mean_across_speeds'."
        )

    common = dict(
        eta_grid=eta_grid, theta_grid=theta_grid,
        eta_lo=eta_lo, eta_hi=eta_hi, theta_lo=theta_lo, theta_hi=theta_hi,
        n_repeats_fit=n_repeats_fit, tnd=tnd, dt_override_s=dt_override_s,
        rt_tp_sd=rt_tp_sd, rt_ta_sd=rt_ta_sd, dprime_sd=dprime_sd,
    )

    rows = []

    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models or pp not in null_models:
            continue

        trials_pp = preprocess_participant_trials(pp, pairs, dprime_models[pp])
        if len(trials_pp) < 10:
            continue

        # same stable split as main fit
        rng_split = np.random.default_rng(12345 + int(pp[2:]))
        idx = rng_split.permutation(len(trials_pp))
        n_train = int(0.7 * len(idx))
        train_trials = [trials_pp[i] for i in idx[:n_train]]
        test_trials = [trials_pp[i] for i in idx[n_train:]]

        human_train = compute_human_summary_from_preprocessed(train_trials)
        human_test = compute_human_summary_from_preprocessed(test_trials)
        N_OBS = count_obs(human_train)

        def _test_loss(models_pp, b):
            model = simulate_model_summary_from_preprocessed(
                trials_pp=test_trials, dprime_models_pp=models_pp,
                eta=b["eta"], theta=b["theta"],
                n_repeats=n_repeats_fit, tnd_s=tnd, dt_override_s=dt_override_s,
            )
            return loss_summary(human_test, model, rt_tp_sd=rt_tp_sd, rt_ta_sd=rt_ta_sd, dprime_sd=dprime_sd)

        # V_indiv - refit eta/theta with individual visibility
        b_indiv = _comparison_grid_search(train_trials, dprime_models[pp], human_train, **common)
        aic_indiv, bic_indiv = information_criteria(b_indiv["loss"], N_OBS, k=2)
        tl_indiv = _test_loss(dprime_models[pp], b_indiv)

        # V_null - refit eta/theta with null visibility
        b_null = _comparison_grid_search(train_trials, null_models[pp], human_train, **common)
        aic_null, bic_null = information_criteria(b_null["loss"], N_OBS, k=2)
        tl_null = _test_loss(null_models[pp], b_null)

        print(
            f"[VIS COMPARE] {pp} | "
            f"V_indiv: eta={b_indiv['eta']:.3f} theta={b_indiv['theta']:.3f} AIC={aic_indiv:.2f} | "
            f"V_null: eta={b_null['eta']:.3f} theta={b_null['theta']:.3f} AIC={aic_null:.2f} | "
            f"dAIC={aic_null - aic_indiv:.2f}"
        )

        rows.append(dict(
            participant=pp,
            indiv_eta=b_indiv["eta"],         indiv_theta=b_indiv["theta"],
            indiv_train_loss=b_indiv["loss"], indiv_test_loss=tl_indiv,
            indiv_AIC=aic_indiv,              indiv_BIC=bic_indiv,
            null_eta=b_null["eta"],           null_theta=b_null["theta"],
            null_train_loss=b_null["loss"],   null_test_loss=tl_null,
            null_AIC=aic_null,                null_BIC=bic_null,
            delta_AIC=aic_null - aic_indiv,   # positive = individual visibility wins
            N_obs=N_OBS,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[VIS COMPARE] wrote {out_csv}  n_pp={len(df)}")
    return df
