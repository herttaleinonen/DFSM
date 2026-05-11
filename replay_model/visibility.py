#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:43:11 2026

@author: herttaleinonen
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import UnivariateSpline

from config import VT_TO_SPEED
from parsing_and_gaze import vt_from_name


# =========================
# Visibility -> d'(400ms)
# =========================
def rates_to_dprime(hit, fa, eps=1e-4):
    H = np.clip(hit, eps, 1 - eps)
    F = np.clip(fa, eps, 1 - eps)
    return norm.ppf(H) - norm.ppf(F)


def compute_dprime_from_group(group: pd.DataFrame) -> pd.Series:
    """
    Takes a subset of visibility trials (one participant × one speed × one ecc bin).

    Computes hit rate HH, false alarm rate FF.

    Applies the SDT transform: norm.ppf(H) - norm.ppf(F) → d′
    """
    targets = group[group["stim_type"] == 1]
    distractors = group[group["stim_type"] == 0]

    h = int(np.sum(targets["response"] == 1))
    f = int(np.sum(distractors["response"] == 1))
    nT = int(len(targets))
    nD = int(len(distractors))

    # log-linear correction
    H = (h + 0.5) / (nT + 1.0) if nT > 0 else 0.5
    F = (f + 0.5) / (nD + 1.0) if nD > 0 else 0.5
    dprime = float(norm.ppf(H) - norm.ppf(F))

    return pd.Series({"dprime": dprime, "H": H, "F": F, "nT": nT, "nD": nD})


def build_dprime_splines_for_participant(
    visibility_files: List[str],
    spline_s: float = 0.5,
    ecc_bin_edges: Optional[List[float]] = None,
    ecc_bin_labels: Optional[List[float]] = None,
) -> Dict[int, UnivariateSpline]:
    """
    Build speed->spline(ecc_deg)->d'(400ms) for one participant from all their visibility vt files.
    Uses stim_speed_px_s directly.
    """
    dfs = []
    for path in visibility_files:
        df = pd.read_csv(path)
        if "stim_speed_px_s" not in df.columns:
            vt = vt_from_name(path)
            if vt is None:
                raise ValueError(f"No stim_speed_px_s and cannot infer vt# from {path}")
            df["stim_speed_px_s"] = VT_TO_SPEED[vt]
        dfs.append(df)

    vis = pd.concat(dfs, ignore_index=True)

    if ecc_bin_edges is None:
        ecc_bin_edges = [0, 4.5, 9, 14, 18, 25]
    if ecc_bin_labels is None:
        ecc_bin_labels = [3, 6, 12, 16, 20]

    vis["ecc_bin"] = pd.cut(
        vis["ecc_deg_actual"],
        bins=ecc_bin_edges,
        labels=ecc_bin_labels,
        include_lowest=True,
    ).astype(float)

    dp_table = (
        vis.groupby(["stim_speed_px_s", "ecc_bin"])
        .apply(compute_dprime_from_group)
        .reset_index()
        .rename(columns={"stim_speed_px_s": "speed_px_s"})
    )

    splines: Dict[int, UnivariateSpline] = {}
    for speed in sorted(dp_table["speed_px_s"].unique()):
        sub = dp_table[dp_table["speed_px_s"] == speed].sort_values("ecc_bin")
        ecc = sub["ecc_bin"].values.astype(float)
        dp = sub["dprime"].values.astype(float)

        spl = UnivariateSpline(ecc, dp, k=2, s=spline_s)
        splines[int(speed)] = spl

    return splines


# =========================
# Visibility null-models
# =========================
def constant_dprime_fn(c: float):
    """
    Returns a callable that behaves like a spline but always returns
    the same d' for any eccentricity input.
    """
    c = float(c)

    def f(ecc):
        ecc = np.asarray(ecc, dtype=float)
        return np.full_like(ecc, c, dtype=float)

    return f


def make_visibility_null_model(
    dprime_models_pp: Dict[int, callable],
    mode: str = "empirical",
    source_speed: int = 0,
    constant_rule: str = "mean",
    ref_ecc_deg: float = 0.0,
) -> Dict[int, callable]:
    """
    Transform a participant's visibility model.

    mode:
      - 'empirical'            : leave as-is
      - 'freeze_speed_at_0'    : use the 0 deg/s spline for all speeds,
                                 but still vary with eccentricity
      - 'constant_from_speed0' : use one constant d' everywhere,
                                 derived from the 0 deg/s spline

    constant_rule (used only for constant_from_speed0):
      - 'mean'     : mean d' sampled from source_speed spline across ecc range
      - 'foveal'   : d' at ref_ecc_deg (default 0 deg)
      - 'median'   : median sampled d'
    """
    if mode == "empirical":
        return dprime_models_pp

    if source_speed not in dprime_models_pp:
        raise ValueError(f"source_speed={source_speed} not found in dprime model")

    src = dprime_models_pp[source_speed]

    if mode == "freeze_speed_at_0":
        # same ecc function for every speed
        return {s: src for s in dprime_models_pp.keys()}

    if mode == "constant_from_speed0":
        sample_ecc = np.linspace(0.0, 25.0, 200)
        src_vals = np.asarray(src(sample_ecc), dtype=float)

        if constant_rule == "mean":
            c = float(np.mean(src_vals))
        elif constant_rule == "median":
            c = float(np.median(src_vals))
        elif constant_rule == "foveal":
            c = float(np.asarray(src(ref_ecc_deg)).reshape(-1)[0])
        else:
            raise ValueError(f"Unknown constant_rule: {constant_rule}")

        const_fn = constant_dprime_fn(c)
        return {s: const_fn for s in dprime_models_pp.keys()}

    raise ValueError(f"Unknown visibility null mode: {mode}")

