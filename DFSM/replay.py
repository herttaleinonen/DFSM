#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:46:29 2026

@author: herttaleinonen
"""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

from config import (
    DEG_PER_CELL,
    GRID_SIZE_X,
    GRID_SIZE_Y,
    DT_TO_SPEED,
    GLOBAL_TND_S,
    MODEL_SAMPLING_RATE,
    GAZE_MODE,
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
    perturb_gaze,
    _gaze_change_flags,
    collect_visibility_files,
    collect_search_pairs,
    stable_trial_seed,
    dt_from_name,
)

from visibility import (
    build_dprime_splines_for_participant,
    make_visibility_null_model,
)


# =========================
# Helpers
# =========================
def ecc_deg_from_cells(obj_xy_cells_t: np.ndarray, gaze_xy_cells_t: np.ndarray) -> np.ndarray:
    ecc_cells = np.linalg.norm(obj_xy_cells_t - gaze_xy_cells_t[None, :], axis=1)
    return ecc_cells * DEG_PER_CELL


def min_target_eccentricity(obj_xy_cells, gaze_xy_cells, target_index):
    """Return minimum eccentricity (deg) between gaze and target over trial."""
    if target_index is None:
        return np.nan

    T = min(obj_xy_cells.shape[0], gaze_xy_cells.shape[0])
    dmin = np.inf

    for t in range(T):
        gaze = gaze_xy_cells[t]
        if np.any(np.isnan(gaze)):
            continue

        ecc = np.linalg.norm(obj_xy_cells[t, target_index] - gaze) * DEG_PER_CELL
        if ecc < dmin:
            dmin = ecc

    return float(dmin) if np.isfinite(dmin) else np.nan


def time_to_first_vis_d400(
    obj_xy_cells: np.ndarray,
    gaze_xy_cells: np.ndarray,
    target_index: int,
    *,
    spl: "UnivariateSpline",
    alpha_search: float,
    dt_s: float,
    thresh: float,
) -> float:
    """First time (s) when target's d400 >= thresh. Returns NaN if never reached. 
       Recorded for debug purposes only.
    """
    T, N, _ = obj_xy_cells.shape
    if target_index is None or target_index < 0 or target_index >= N:
        return np.nan

    for t in range(T):
        gaze = gaze_xy_cells[t]
        if np.any(np.isnan(gaze)):
            continue
        ecc = ecc_deg_from_cells(obj_xy_cells[t], gaze)
        d400 = float(np.maximum(alpha_search * spl(ecc[target_index]), 0.0))
        if d400 >= thresh:
            return float((t + 1) * dt_s)  # matches t_sec convention
    return np.nan


def infer_target_index(gabor_pos: np.ndarray, target_traj: np.ndarray) -> int:
    """
    gabor_pos:    [T, N, 2] positions in grid coords
    target_traj:  [T, 2] target trajectory in same coords
    Returns the object index j that best matches the target trajectory.
    """
    T, N, _ = gabor_pos.shape
    T2 = min(T, target_traj.shape[0])

    dists = np.zeros(N, dtype=float)
    for j in range(N):
        dj = np.linalg.norm(gabor_pos[:T2, j, :] - target_traj[:T2, :], axis=1)
        dists[j] = float(np.mean(dj))

    return int(np.argmin(dists))


# =========================
# The main update loop for a single trial
# =========================
def run_replay_trial(
    obj_xy_cells: np.ndarray,
    gaze_xy_cells: np.ndarray,
    speed_px_s: int,
    dprime_splines: Dict[int, UnivariateSpline],
    dt_s: float,
    eta: float,
    decision_theta_present: float,
    target_present: int,
    target_index: Optional[int],
    max_time_s: float = 3.5,
    alpha_search: float = 1.0,
    dt_override_s: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[int, float]:

    if rng is None:
        rng = np.random.default_rng()

    T, N, _ = obj_xy_cells.shape

    if dt_s <= 0:
        dt_s = max_time_s / max(1, T - 1)

    dt_eff = dt_s if dt_override_s is None else float(dt_override_s)
    T_use = min(T, int(np.floor(max_time_s / dt_s)))
    if T_use <= 0:
        return 0, 0.0

    if speed_px_s not in dprime_splines:
        raise ValueError(f"Missing d' spline for speed_px_s={speed_px_s}")

    theta_pos = float(decision_theta_present)
    theta_neg = -theta_pos

    spl = dprime_splines[speed_px_s]
    logLR = np.zeros(N, dtype=float)

    eta_safe = max(1e-6, float(eta))
    noise_scale = np.sqrt(eta_safe)

    for t in range(T_use):
        gaze = gaze_xy_cells[t]
        if np.any(np.isnan(gaze)):
            gaze = np.array([GRID_SIZE_X / 2.0, GRID_SIZE_Y / 2.0], dtype=float)

        ecc_deg = ecc_deg_from_cells(obj_xy_cells[t], gaze)
        d400 = np.maximum(alpha_search * spl(ecc_deg), 0.0)

        # Convert 400 ms d' into per-step evidence strength
        dstep = 0.4 * d400 * np.sqrt(dt_eff / 0.4)

        # sensory sample
        x = rng.normal(loc=0.0, scale=noise_scale, size=N)

        # if target is present, shift the target channel toward the target hypothesis
        if (target_present == 1) and (target_index is not None):
            x[target_index] += dstep[target_index]

        # object-wise LLR increment
        dllr = dstep * x - 0.5 * (dstep ** 2)
        logLR += dllr

        dv = np.max(logLR)
        t_sec = (t + 1) * dt_eff

        if dv >= theta_pos:
            return 1, t_sec
        if dv <= theta_neg:
            return 0, t_sec

    resp = 1 if dv > 0 else 0
    return resp, T_use * dt_eff


# =========================
# Update loop for saccade prediction 
# =========================
def replay_trace_trial(
    obj_xy_cells: np.ndarray,
    gaze_xy_cells: np.ndarray,
    speed_px_s: int,
    dprime_splines: Dict[int, UnivariateSpline],
    dt_s: float,
    eta: float,
    target_present: int,
    target_index: Optional[int],
    max_time_s: float = 3.5,
    alpha_search: float = 1.0,
    dt_override_s: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Returns one row per model time step with evidence variables.
    """
    if rng is None:
        rng = np.random.default_rng()

    T, N, _ = obj_xy_cells.shape
    if dt_s <= 0:
        dt_s = max_time_s / max(1, T - 1)

    dt_eff = dt_s if dt_override_s is None else float(dt_override_s)
    T_use = min(T, int(np.floor(max_time_s / dt_s)))
    if T_use <= 1:
        return pd.DataFrame()

    if speed_px_s not in dprime_splines:
        raise ValueError(f"Missing d' spline for speed_px_s={speed_px_s}")

    spl = dprime_splines[speed_px_s]
    logLR = np.zeros(N, dtype=float)

    eta_safe = max(1e-6, float(eta))
    noise_scale = np.sqrt(eta_safe)

    rows = []

    for t in range(T_use):
        gaze = gaze_xy_cells[t]
        if np.any(np.isnan(gaze)):
            gaze = np.array([GRID_SIZE_X / 2.0, GRID_SIZE_Y / 2.0], dtype=float)

        ecc_deg = ecc_deg_from_cells(obj_xy_cells[t], gaze)
        d400 = np.maximum(alpha_search * spl(ecc_deg), 0.0)

        # Convert 400 ms d' into per-step evidence strength
        dstep = 0.4 * d400 * np.sqrt(dt_eff / 0.4)

        # sensory sample
        x = rng.normal(loc=0.0, scale=noise_scale, size=N)

        # if target is present, shift the target channel toward the target hypothesis
        if (target_present == 1) and (target_index is not None):
            x[target_index] += dstep[target_index]

        # object-wise LLR increment
        dllr = dstep * x - 0.5 * (dstep ** 2)
        logLR += dllr

        # summary evidence variables
        order = np.argsort(logLR)
        max_idx = int(order[-1])
        max_loglr = float(logLR[max_idx])
        second_loglr = float(logLR[order[-2]]) if N >= 2 else np.nan
        margin = float(max_loglr - second_loglr) if N >= 2 else np.nan
        abs_dv = float(abs(max_loglr))

        target_loglr = np.nan
        target_ecc = np.nan
        target_d400 = np.nan
        if (target_present == 1) and (target_index is not None):
            target_loglr = float(logLR[target_index])
            target_ecc = float(ecc_deg[target_index])
            target_d400 = float(d400[target_index])

        rows.append({
            "t": int(t),
            "t_sec": float((t + 1) * dt_eff),
            "max_loglr": max_loglr,
            "abs_dv": abs_dv,
            "second_loglr": second_loglr,
            "margin": margin,
            "winner_idx": max_idx,
            "target_loglr": target_loglr,
            "target_ecc_deg": target_ecc,
            "target_d400": target_d400,
        })

    return pd.DataFrame(rows)


def build_saccade_prediction_table(
    search_dir: str,
    visibility_dir: str,
    fitted_params_csv: str,
    output_csv: str = "saccade_prediction_table.csv",
    spline_s: float = 0.5,
    dt_override_s: Optional[float] = MODEL_SAMPLING_RATE,
    max_calib_drift_deg: Optional[float] = None,
    gaze_mode: str = "real",
) -> pd.DataFrame:
    """
    Builds one row per 50 ms bin per trial.
    Main DV: fix_change_next = whether gaze changes at the next sample.
    """

    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)

    fit_df = pd.read_csv(fitted_params_csv)
    fit_map = {
        str(r["participant"]).strip(): (float(r["eta"]), float(r["theta"]))
        for _, r in fit_df.iterrows()
    }

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

    all_rows = []

    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            print(f"[SACCADE SKIP] {pp}: no visibility files")
            continue
        if pp not in fit_map:
            print(f"[SACCADE SKIP] {pp}: no fitted params")
            continue

        eta_pp, theta_pp = fit_map[pp]

        for csv_path, asc_path in sorted(pairs):
            df = pd.read_csv(csv_path)

            if max_calib_drift_deg is not None and "CalibrationDrift(deg)" in df.columns:
                df = df[df["CalibrationDrift(deg)"] <= max_calib_drift_deg].copy()

            required = ["Trial", "Gabor Positions", "Target Present", "Response", "Correct", "Reaction Time (s)"]
            if any(c not in df.columns for c in required):
                print(f"[SACCADE SKIP FILE] missing columns in {csv_path}")
                continue

            dt_tag = dt_from_name(csv_path)
            if dt_tag is None:
                continue
            speed_px_s = DT_TO_SPEED[dt_tag]
            alpha_use = 0.5 * (1 - 0.5 * (speed_px_s / 400))

            if speed_px_s not in dprime_models[pp]:
                continue

            (screen_w, screen_h), eye_trials = parse_asc_events(asc_path)
            df["gabor_pos"] = df["Gabor Positions"].apply(safe_parse_positions)

            kept = 0
            skipped = 0

            for _, r in df.iterrows():
                trial = safe_int(r.get("Trial"))
                target_present = safe_int(r.get("Target Present"))
                human_resp = safe_int(r.get("Response"))
                human_corr = safe_int(r.get("Correct"))
                human_rt = safe_float(r.get("Reaction Time (s)"))

                if trial is None or target_present is None or human_resp is None or human_corr is None or human_rt is None:
                    skipped += 1
                    continue

                obj = r.get("gabor_pos")
                if obj is None or not isinstance(obj, np.ndarray) or obj.ndim != 3 or obj.shape[0] < 2:
                    skipped += 1
                    continue

                dt_s = estimate_dt_from_positions(obj, speed_px_s) if speed_px_s > 0 else None
                if dt_s is None or dt_s <= 0:
                    dt_s = dt_from_duration(obj, 3.5)

                tr_eye = eye_trials.get(trial)
                if tr_eye is None:
                    gaze_cells = np.full((obj.shape[0], 2), np.nan, dtype=float)
                else:
                    gaze_cells = gaze_series_cells_from_fixations(tr_eye, obj.shape[0], dt_s, screen_w, screen_h)

                trial_seed = stable_trial_seed(pp, os.path.basename(csv_path), int(trial))
                rng_trial = np.random.default_rng(trial_seed)
                gaze_cells_used = perturb_gaze(gaze_cells, mode=gaze_mode, rng=rng_trial, dt_s=dt_override_s if dt_override_s is not None else dt_s)

                target_index = None
                if target_present == 1:
                    traj = safe_parse_trajectory(r.get("Target Trajectory"))
                    if traj is not None:
                        target_index = infer_target_index(obj, traj)

                trace = replay_trace_trial(
                    obj_xy_cells=obj,
                    gaze_xy_cells=gaze_cells_used,
                    speed_px_s=speed_px_s,
                    dprime_splines=dprime_models[pp],
                    dt_s=dt_s,
                    eta=eta_pp,
                    target_present=target_present,
                    target_index=target_index,
                    alpha_search=alpha_use,
                    dt_override_s=dt_override_s,
                    rng=np.random.default_rng(trial_seed),
                )

                if len(trace) < 2:
                    skipped += 1
                    continue

                # align gaze/event coding to same time base as trace
                T_use = len(trace)
                gaze_use = gaze_cells_used[:T_use].copy()
                gaze_change = _gaze_change_flags(gaze_use)

                # fixation age in bins/time
                fix_age_bins = np.zeros(T_use, dtype=int)
                age = 0
                for t in range(T_use):
                    if t == 0:
                        age = 0
                    else:
                        if gaze_change[t] == 1:
                            age = 0
                        else:
                            age += 1
                    fix_age_bins[t] = age

                """
                fix_change_next = np.zeros(T_use, dtype=int)
                fix_change_next[:-1] = gaze_change[1:]
                fix_change_next[-1] = 0
                """
                # next-200ms event: did gaze change in the next 4 bins?
                fix_change_next_200 = np.zeros(T_use, dtype=int)
                
                for t in range(T_use):
                    future_changes = []
                    for k in [1, 2, 3, 4]:
                        if t + k < T_use:
                            future_changes.append(gaze_change[t + k])
                    fix_change_next_200[t] = int(np.any(future_changes)) if future_changes else 0

                trace["participant"] = pp
                trace["trial"] = int(trial)
                trace["search_csv"] = os.path.basename(csv_path)
                trace["asc_file"] = os.path.basename(asc_path)
                trace["speed_px_s"] = int(speed_px_s)
                trace["target_present"] = int(target_present)
                trace["human_response"] = int(human_resp)
                trace["human_correct"] = int(human_corr)
                trace["human_rt_s"] = float(human_rt)
                trace["eta_used"] = float(eta_pp)
                trace["theta_used"] = float(theta_pp)
                trace["alpha_used"] = float(alpha_use)

                trace["fix_change_now"] = gaze_change.astype(int)
                #trace["fix_change_next"] = fix_change_next.astype(int)
                trace["fix_change_next_200"] = fix_change_next_200.astype(int)
                trace["fix_age_bins"] = fix_age_bins.astype(int)
                trace["fix_age_s"] = trace["fix_age_bins"] * float(dt_override_s if dt_override_s is not None else dt_s)

                trace["log_abs_dv"] = np.log(np.maximum(trace["abs_dv"].values, 1e-6))
                trace["log_margin"] = np.log(np.maximum(trace["margin"].fillna(0).values, 1e-6))

                trace = trace.iloc[:-1].copy()

                all_rows.append(trace)
                kept += 1

            print(f"[SACCADE OK] {pp} speed={speed_px_s}: {os.path.basename(csv_path)} kept={kept} skipped={skipped}")

    out = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    out.to_csv(output_csv, index=False)
    print(f"\n[SACCADE] wrote {output_csv} rows={len(out)}")
    return out


# =========================
# Run everything + build final CSV
# =========================
def run_full_replay(
    search_dir: str,
    visibility_dir: str,
    output_csv: str = "replay_model_results.csv",
    n_model_repeats: int = 50,
    spline_s: float = 0.5,
    tnd_s: float = GLOBAL_TND_S,
    max_calib_drift_deg: Optional[float] = None,
    fitted_params_csv: Optional[str] = None,
    dt_override_s: Optional[float] = None,
    debug: bool = False,
) -> pd.DataFrame:

    vis_by_pp = collect_visibility_files(visibility_dir)
    search_by_pp = collect_search_pairs(search_dir)

    if fitted_params_csv is None:
        raise ValueError("fitted_params_csv must be provided to run per-participant fitted replay.")

    fit_df = pd.read_csv(fitted_params_csv)

    if "tnd" in fit_df.columns:
        tnd_s = float(np.nanmedian(fit_df["tnd"].values))
        print(f"[REPLAY] Using global tnd_s from fitted_params_csv: {tnd_s:.3f}s")

    fit_map = {
        str(r["participant"]).strip(): (float(r["eta"]), float(r["theta"]))
        for _, r in fit_df.iterrows()
    }

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

    all_rows = []
    printed_ecc_debug = set()

    for pp, pairs in sorted(search_by_pp.items()):
        if pp not in dprime_models:
            print(f"[SKIP] {pp}: no visibility files found")
            continue

        if pp not in fit_map:
            raise ValueError(f"No fitted parameters found for participant {pp}")

        eta_pp, theta_pp = fit_map[pp]

        if debug and pp.strip().lower() == "kh1":
            eccs = np.array([0, 3, 6, 9, 12, 15, 18], float)
            print("\n=== d' vs eccentricity (kh1) ===", flush=True)
            for s in [0, 100, 200, 300, 400]:
                if s in dprime_models[pp]:
                    vals = dprime_models[pp][s](eccs)
                    print(f"speed {s:3d}:", np.round(vals, 3), flush=True)
                else:
                    print(f"speed {s:3d}: missing spline", flush=True)
            print("eccs:", eccs, flush=True)
            print("===============================\n", flush=True)

        for csv_path, asc_path in sorted(pairs):
            df = pd.read_csv(csv_path)

            if max_calib_drift_deg is not None and "CalibrationDrift(deg)" in df.columns:
                df = df[df["CalibrationDrift(deg)"] <= max_calib_drift_deg].copy()

            required = ["Trial", "Gabor Positions", "Target Present", "Response", "Correct", "Reaction Time (s)"]
            for c in required:
                if c not in df.columns:
                    raise ValueError(f"Missing column '{c}' in {csv_path}")

            dt = dt_from_name(csv_path)
            if dt is None:
                raise ValueError(f"Cannot infer dt# from filename: {csv_path}")

            speed_px_s = DT_TO_SPEED[dt]
            alpha_use = 0.5 * (1 - 0.5 * (speed_px_s / 400))
            raw_speed = float(df["Speed (px/s)"].iloc[0]) if "Speed (px/s)" in df.columns else float(speed_px_s)

            if speed_px_s not in dprime_models[pp]:
                print(f"[SKIP] {pp} speed {speed_px_s}: no visibility spline")
                continue

            (screen_w, screen_h), eye_trials = parse_asc_events(asc_path)
            df["gabor_pos"] = df["Gabor Positions"].apply(safe_parse_positions)

            skipped = 0
            kept = 0

            for _, r in df.iterrows():
                trial = safe_int(r.get("Trial"))
                target_present = safe_int(r.get("Target Present"))
                human_resp = safe_int(r.get("Response"))
                human_corr = safe_int(r.get("Correct"))
                human_rt = safe_float(r.get("Reaction Time (s)"))

                if trial is None or target_present is None or human_resp is None or human_corr is None or human_rt is None:
                    skipped += 1
                    continue

                obj = r.get("gabor_pos")
                if obj is None or not isinstance(obj, np.ndarray) or obj.ndim != 3:
                    skipped += 1
                    continue

                Tstim = obj.shape[0]
                if Tstim < 2:
                    skipped += 1
                    continue

                dt_s = estimate_dt_from_positions(obj, speed_px_s) if speed_px_s > 0 else None
                if dt_s is None or dt_s <= 0:
                    dt_s = dt_from_duration(obj, 3.5)

                tr_eye = eye_trials.get(trial)
                if tr_eye is None:
                    gaze_cells = np.full((Tstim, 2), np.nan, dtype=float)
                else:
                    gaze_cells = gaze_series_cells_from_fixations(tr_eye, Tstim, dt_s, screen_w, screen_h)

                trial_seed = stable_trial_seed(pp, os.path.basename(csv_path), int(trial))
                rng_trial = np.random.default_rng(trial_seed)

                gaze_cells_used = perturb_gaze(
                    gaze_cells,
                    mode=GAZE_MODE,
                    rng=rng_trial,
                    dt_s=dt_override_s if dt_override_s is not None else dt_s,
                )

                target_index = None
                if target_present == 1:
                    traj = safe_parse_trajectory(r.get("Target Trajectory"))
                    if traj is not None:
                        target_index = infer_target_index(obj, traj)

                min_target_ecc_real = min_target_eccentricity(obj, gaze_cells, target_index)
                min_target_ecc_used = min_target_eccentricity(obj, gaze_cells_used, target_index)

                key = (pp, GAZE_MODE)
                if debug and (key not in printed_ecc_debug) and (target_present == 1) and (target_index is not None):
                    print("GAZE_MODE:", GAZE_MODE, "| pp:", pp, "| trial:", trial)
                    print("min_ecc_real:", min_target_ecc_real)
                    print("min_ecc_used:", min_target_ecc_used)
                    printed_ecc_debug.add(key)

                model_resp = np.empty(n_model_repeats, dtype=int)
                model_rt = np.empty(n_model_repeats, dtype=float)

                ss = np.random.SeedSequence(trial_seed)
                child_seeds = ss.spawn(n_model_repeats)

                for k in range(n_model_repeats):
                    rng_k = np.random.default_rng(child_seeds[k])

                    resp, rt = run_replay_trial(
                        obj_xy_cells=obj,
                        gaze_xy_cells=gaze_cells_used,
                        speed_px_s=speed_px_s,
                        dprime_splines=dprime_models[pp],
                        dt_s=dt_s,
                        eta=eta_pp,
                        decision_theta_present=float(theta_pp),
                        target_present=target_present,
                        target_index=target_index,
                        alpha_search=alpha_use,
                        rng=rng_k,
                        dt_override_s=dt_override_s,
                    )

                    model_resp[k] = resp
                    model_rt[k] = rt + float(tnd_s)

                present_mask = (model_resp == 1)
                absent_mask = (model_resp == 0)

                time_first = {}
                if (target_present == 1) and (target_index is not None):
                    for th in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
                        time_first[th] = time_to_first_vis_d400(
                            obj_xy_cells=obj,
                            gaze_xy_cells=gaze_cells_used,
                            target_index=target_index,
                            spl=dprime_models[pp][speed_px_s],
                            alpha_search=alpha_use,
                            dt_s=dt_s,
                            thresh=th,
                        )

                row = {
                    "Task Type": r.get("Task Type", "search"),
                    "Participant ID": r.get("Participant ID", pp),
                    "participant": pp,
                    "speed_px_s_raw": raw_speed,
                    "speed_px_s_used": speed_px_s,
                    "trial": trial,
                    "dt_s_est": float(dt_s),
                    "human_target_present": target_present,
                    "human_response": human_resp,
                    "human_correct": human_corr,
                    "human_rt_s": float(human_rt),
                    "model_p_present": float(model_resp.mean()),
                    "model_rt_mean_s": float(model_rt.mean()),
                    "model_rt_median_s": float(np.median(model_rt)),
                    "model_rt_present_mean_s": float(np.mean(model_rt[present_mask])) if present_mask.any() else np.nan,
                    "model_rt_absent_mean_s": float(np.mean(model_rt[absent_mask])) if absent_mask.any() else np.nan,
                    "alpha_used": alpha_use,
                    "theta_present_used": float(theta_pp),
                    "eta_used": float(eta_pp),
                    "theta_fitted": float(theta_pp),
                    "tnd_used": float(tnd_s),
                    "min_target_ecc_deg_real": float(min_target_ecc_real),
                    "min_target_ecc_deg_used": float(min_target_ecc_used),
                    "search_csv": os.path.basename(csv_path),
                    "asc_file": os.path.basename(asc_path),
                }

                for th in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
                    val = time_first.get(th, np.nan)
                    row[f"time_to_first_vis_d400_{th:.1f}_s"] = float(val) if np.isfinite(val) else np.nan

                for extra in ["FixOnTargetTime(s)", "LastFixIndex", "CalibrationDrift(deg)"]:
                    if extra in r:
                        row[extra] = r[extra]

                all_rows.append(row)
                kept += 1

            print(f"[OK] {pp} speed={speed_px_s}: {os.path.basename(csv_path)} kept={kept} skipped={skipped}")

    out = pd.DataFrame(all_rows)
    out.to_csv(output_csv, index=False)
    print(f"\nWrote {output_csv} rows={len(out)}")
    return out


