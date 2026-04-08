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

from config import (
    GLOBAL_TND_S,
    SPEED_ORDER,
    DT_TO_SPEED,
    VISIBILITY_MODE,
    VIS_NULL_SOURCE_SPEED,
    VIS_CONSTANT_RULE,
    VIS_REF_ECC_DEG,
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
)

from visibility import (
    build_dprime_splines_for_participant,
    make_visibility_null_model,
    rates_to_dprime,
)

from replay import (
    run_replay_trial,
    infer_target_index,
)


# ------------------
# Fitting helpers
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


# ------------------
# Pre-processing
# ------------------
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
        alpha_use = 0.5 * (1 - 0.5 * (speed_px_s / 400))

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


            alpha_trial = alpha_use   # same for TP and TA

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
                alpha_trial=alpha_trial,
                trial_seed=trial_seed,
                
                
                # human fields for train/test summaries
                human_resp=human_resp,
                human_corr=human_corr,
                human_rt=human_rt,
            ))

    return trials_out


# ------------------
# Model recovery helpers
# ------------------
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


def make_synthetic_human_trials(
    trials_pp: List[dict],
    dprime_models_pp: Dict[int, UnivariateSpline],
    eta_true: float,
    theta_true: float,
    tnd_s: float = GLOBAL_TND_S,
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
            dt_override_s = None,       
            eta=float(eta_true),
            decision_theta_present=float(theta_true),
            target_present=tr["target_present"],
            target_index=tr["target_index"],
            alpha_search=tr["alpha_trial"],
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


# ------------------
# Loss function
# ------------------
def loss_summary(human_by_speed: Dict[int, Dict[str, float]],
                 model_by_speed: Dict[int, Dict[str, float]],
                 w_choice: float = 1.0,
                 w_rt: float = 1.0) -> float:

    loss = 0.0

    for s in SPEED_ORDER:
        h = human_by_speed[s]
        m = model_by_speed[s]

        # d'
        if not (np.isnan(h["hit"]) or np.isnan(h["fa"]) or
                np.isnan(m["hit"]) or np.isnan(m["fa"])):

            d_h = rates_to_dprime(h["hit"], h["fa"])
            d_m = rates_to_dprime(m["hit"], m["fa"])

            loss += w_choice * (d_h - d_m) ** 2

        # TP RT
        if not (np.isnan(h["rt_tp"]) or np.isnan(m["rt_tp"])):
            z = safe_log(h["rt_tp"]) - safe_log(m["rt_tp"])
            loss += w_rt * (z ** 2)

        # TA RT
        if not (np.isnan(h["rt_ta"]) or np.isnan(m["rt_ta"])):
            z = safe_log(h["rt_ta"]) - safe_log(m["rt_ta"])
            loss += w_rt * (z ** 2)

    return float(loss)


# ------------------
# Fitting loop
# ------------------
def fit_model_per_participant(
    search_dir: str,
    visibility_dir: str,
    eta_grid: np.ndarray,
    theta_grid: np.ndarray,
    n_repeats_fit: int = 10,
    spline_s: float = 0.5,
    out_csv: str = "fitted_params.csv",
    warmstart_csv: Optional[str] = None,
    eta_half_width: float = 0.15,
    theta_half_width: float = 6.0,
    theta_lo: float = 2.0,
    theta_hi: float = 12.0,
    theta_shift: float = 0.0,
    eta_lo: float = 0.05,
    eta_hi: float = 0.8,
    dt_override_s: Optional[float] = None,
) -> pd.DataFrame:
    
    """
    Fits (eta, theta) per participant using a two-pass grid search (coarse + refine),
    with a stable 70/30 train/test split per participant for cross valucation.

    """

    # -----------------------------------------
    # Model parameter availability
    # -----------------------------------------
    # number of fitted parameters (eta, theta)
    K_PARAMS = 2

    # Fixed global NDT
    tnd = float(GLOBAL_TND_S)
    print(f"\n===== FIXED global NDT: tnd={tnd:.3f}s =====")

    # -----------------------------------------
    # Warmstart map (optional)
    # -----------------------------------------
    warm_map: Dict[str, Tuple[float, float]] = {}
    if warmstart_csv is not None:
        warm_df = pd.read_csv(warmstart_csv)
        if ("participant" in warm_df.columns) and ("eta" in warm_df.columns) and ("theta" in warm_df.columns):
            warm_map = {
                str(r["participant"]).strip(): (float(r["eta"]), float(r["theta"]))
                for _, r in warm_df.iterrows()
            }

    # -----------------------------------------
    # Collect files + build d' splines once
    # -----------------------------------------
    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)

    dprime_models: Dict[str, Dict[int, UnivariateSpline]] = {}
    for pp, files in vis_by_pp.items():
        base_model = build_dprime_splines_for_participant(files, spline_s=spline_s)
        dprime_models[pp] = make_visibility_null_model(
            base_model,
            mode=VISIBILITY_MODE,
            source_speed=VIS_NULL_SOURCE_SPEED,
            constant_rule=VIS_CONSTANT_RULE,
            ref_ecc_deg=VIS_REF_ECC_DEG,
        )

    rows = []

    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            print(f"[FIT SKIP] {pp}: no visibility splines")
            continue

        trials_pp = preprocess_participant_trials(pp, pairs, dprime_models[pp])
        if len(trials_pp) < 10:
            print(f"[FIT SKIP] {pp}: too few preprocessed trials ({len(trials_pp)})")
            continue

        print(f"[FIT] {pp}: preprocessed {len(trials_pp)} trials")

        # -----------------------------------------
        # Train/test split (stable per participant)
        # -----------------------------------------
        rng_split = np.random.default_rng(12345 + int(pp[2:]))
        idx = rng_split.permutation(len(trials_pp))
        n_train = int(0.7 * len(idx))
        train_trials = [trials_pp[i] for i in idx[:n_train]]
        test_trials  = [trials_pp[i] for i in idx[n_train:]]

        human_train = compute_human_summary_from_preprocessed(train_trials)
        human_test  = compute_human_summary_from_preprocessed(test_trials)
        N_OBS = count_obs(human_train)

        # -----------------------------------------
        # Per-participant grid (centered if warmstart)
        # -----------------------------------------
        if pp in warm_map:
            eta0, theta0_old = warm_map[pp]
            eta_grid_pp = make_refined_grid(eta0, eta_half_width, 9, lo=eta_lo, hi=eta_hi)

            theta_center = float(theta0_old + theta_shift)
            theta_grid_pp = make_refined_grid(theta_center, theta_half_width, 9, lo=theta_lo, hi=theta_hi)
        else:
            eta_grid_pp = np.asarray(eta_grid, dtype=float)
            theta_grid_pp = np.asarray(theta_grid, dtype=float)


        best = {"loss": np.inf, "eta": None, "theta": None}

        # ---------- PASS 1: coarse grid ----------
        for eta in eta_grid_pp:
            for theta in theta_grid_pp:
                model = simulate_model_summary_from_preprocessed(
                    trials_pp=train_trials,
                    dprime_models_pp=dprime_models[pp],
                    eta=float(eta),
                    theta=float(theta),
                    n_repeats=n_repeats_fit,
                    tnd_s=tnd,
                    dt_override_s=dt_override_s,
                )
                L = loss_summary(human_train, model)
                if L < best["loss"]:
                    best = {"loss": float(L), "eta": float(eta), "theta": float(theta)}

        print(f"[FIT COARSE] {pp}: eta={best['eta']:.3f} theta={best['theta']:.3f} loss={best['loss']:.4f}")

        # ---------- PASS 2: refinement around winner ----------
        ETA_HALF_WIDTH_REFINE   = 0.15
        THETA_HALF_WIDTH_REFINE = 1.0
        N_REFINE = 12

        eta_ref = make_refined_grid(best["eta"], ETA_HALF_WIDTH_REFINE, N_REFINE, lo=eta_lo, hi=eta_hi)
        theta_ref = make_refined_grid(best["theta"], THETA_HALF_WIDTH_REFINE, N_REFINE, lo=theta_lo, hi=theta_hi)

        best2 = dict(best)

        for eta in eta_ref:
            for theta in theta_ref:
                model = simulate_model_summary_from_preprocessed(
                    trials_pp=train_trials,
                    dprime_models_pp=dprime_models[pp],
                    eta=float(eta),
                    theta=float(theta),
                    n_repeats=n_repeats_fit,
                    tnd_s=tnd,
                    dt_override_s=dt_override_s,
                )
                L = loss_summary(human_train, model, w_choice=1.0, w_rt=1.0)
                if L < best2["loss"]:
                    best2 = {"loss": float(L), "eta": float(eta), "theta": float(theta)}

        best = best2
        print(f"[FIT OK] {pp}: eta={best['eta']:.3f} theta={best['theta']:.3f} loss={best['loss']:.4f}")

        # information criteria 
        aic, bic = information_criteria(best["loss"], N_OBS, k=K_PARAMS)
        print(f"[FIT IC] {pp}: AIC={aic:.2f} BIC={bic:.2f}")

        # ===================== CROSS VALIDATION (TEST SET 30%) =====================
        model_test = simulate_model_summary_from_preprocessed(
            trials_pp=test_trials,
            dprime_models_pp=dprime_models[pp],
            eta=float(best["eta"]),
            theta=float(best["theta"]),
            n_repeats=n_repeats_fit,
            tnd_s=tnd,
            dt_override_s=dt_override_s,
        )

        test_loss = loss_summary(human_test, model_test, w_choice=1.0, w_rt=1.0)
        print(f"[FIT TEST] {pp}: test_loss={test_loss:.4f}")
        # ===============================================================

        rows.append({
            "participant": pp,
            "eta": best["eta"],
            "theta": best["theta"],
            "loss": best["loss"],
            "test_loss": test_loss,
            "AIC": aic,
            "BIC": bic,
            "tnd": tnd,  
        })

    if len(rows) == 0:
        raise RuntimeError("No participants were fit successfully (check file pairing / preprocessing).")

    fit_df = pd.DataFrame(
        rows,
        columns=["participant", "eta", "theta", "loss", "test_loss", "AIC", "BIC", "tnd"],
    )
    fit_df.to_csv(out_csv, index=False)
    print(f"[FIT] wrote {out_csv} (fixed global tnd={tnd:.3f}s) n_pp={len(fit_df)}")
    return fit_df


def fit_eta_theta_from_trials(
    pp: str,
    trials_pp: List[dict],
    dprime_models_pp: Dict[int, UnivariateSpline],
    eta_grid: np.ndarray,
    theta_grid: np.ndarray,
    n_repeats_fit: int = 10,
    tnd_s: float = GLOBAL_TND_S,
) -> Tuple[float, float, float]:
    """
    Fit eta, theta on a provided trial list (which already has human_* fields).
    Returns (eta_hat, theta_hat, best_loss) using the summary loss.
    """

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
) -> pd.DataFrame:

    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)

    # build d' models
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
                )

                eta_hat, theta_hat, loss = fit_eta_theta_from_trials(
                    pp=pp,
                    trials_pp=synth_trials,
                    dprime_models_pp=dprime_models[pp],
                    eta_grid=eta_fit_grid,
                    theta_grid=theta_fit_grid,
                    n_repeats_fit=n_repeats_fit,
                    tnd_s=GLOBAL_TND_S,
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
                alpha_search=tr["alpha_trial"],
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
