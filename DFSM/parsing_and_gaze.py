#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 23:35:48 2026

@author: herttaleinonen
"""

import os
import glob
import re
import ast
import zlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from config import (
    CELL_SIZE_PX,
    GRID_SIZE_X,
    GRID_SIZE_Y,
    GRID_OFFSET_X,
    GRID_OFFSET_Y,
)


# =========================
# Filename parsing
# =========================
def participant_from_name(fname: str) -> str:
    m = re.search(r"(kh\d+)", os.path.basename(fname).lower())
    if not m:
        raise ValueError(f"Could not extract participant (kh#) from {fname}")
    return m.group(1)


def dt_from_name(fname: str) -> Optional[str]:
    m = re.search(r"(dt\d)", os.path.basename(fname).lower()) # dt = dynamic search task
    return m.group(1) if m else None


def vt_from_name(fname: str) -> Optional[str]:
    m = re.search(r"(vt\d)", os.path.basename(fname).lower()) # vt = visibility task
    return m.group(1) if m else None


def is_missing(x) -> bool:
    """True for NaN/None/empty-string."""
    if x is None:
        return True
    if isinstance(x, float) and np.isnan(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def safe_int(x) -> Optional[int]:
    if is_missing(x):
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def safe_float(x) -> Optional[float]:
    if is_missing(x):
        return None
    try:
        return float(x)
    except Exception:
        return None


def safe_parse_positions(cell: str) -> Optional[np.ndarray]:
    """Parse Gabor Positions cell -> np.ndarray or None."""
    if is_missing(cell):
        return None
    try:
        arr = ast.literal_eval(cell)
        a = np.array(arr, dtype=np.float32)
        # Expected shape [T, N, 2]
        if a.ndim != 3 or a.shape[2] != 2:
            return None
        return a
    except Exception:
        return None


def safe_parse_trajectory(cell: str) -> Optional[np.ndarray]:
    """Parse Target Trajectory cell -> np.ndarray [T,2] or None."""
    if is_missing(cell):
        return None
    try:
        arr = ast.literal_eval(cell)
        a = np.array(arr, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 2:
            return None
        return a
    except Exception:
        return None


# =========================
# ASC parsing (events-based)
# =========================
@dataclass
class Fix:
    t0: int
    t1: int
    x_px: float
    y_px: float


@dataclass
class TrialEye:
    trial: int
    onset_ms: Optional[int]
    offset_ms: Optional[int]
    fixes: List[Fix]


GAZE_COORDS_RE = re.compile(
    r"MSG\s+\d+\s+GAZE_COORDS\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)"
)
TRIAL_RE = re.compile(r"MSG\s+(\d+)\s+TRIALID\s+(\d+)")
ONSET_RE = re.compile(r"MSG\s+(\d+)\s+stimulus_onset")
OFFSET_RE = re.compile(r"MSG\s+(\d+)\s+stimulus_offset")
EFIX_RE = re.compile(r"EFIX\s+R\s+(\d+)\s+(\d+)\s+\d+\s+([\d\.]+)\s+([\d\.]+)")


def parse_asc_events(path: str) -> Tuple[Tuple[int, int], Dict[int, TrialEye]]:
    """
    Human eye movements → gaze(t)
    Extracts fixations and onset/offset per trial.
    """
    screen_w, screen_h = 1920, 1200
    trials: Dict[int, TrialEye] = {}

    cur_trial: Optional[int] = None
    cur_onset: Optional[int] = None
    cur_offset: Optional[int] = None
    cur_fixes: List[Fix] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = GAZE_COORDS_RE.search(line)
            if m:
                xmax = float(m.group(3))
                ymax = float(m.group(4))
                screen_w = int(round(xmax + 1))
                screen_h = int(round(ymax + 1))
                continue

            m = TRIAL_RE.search(line)
            if m:
                if cur_trial is not None:
                    trials[cur_trial] = TrialEye(cur_trial, cur_onset, cur_offset, cur_fixes)
                cur_trial = int(m.group(2))
                cur_onset = None
                cur_offset = None
                cur_fixes = []
                continue

            m = ONSET_RE.search(line)
            if m and cur_trial is not None:
                cur_onset = int(m.group(1))
                continue

            m = OFFSET_RE.search(line)
            if m and cur_trial is not None:
                cur_offset = int(m.group(1))
                continue

            m = EFIX_RE.search(line)
            if m and cur_trial is not None:
                t0 = int(m.group(1))
                t1 = int(m.group(2))
                x = float(m.group(3))
                y = float(m.group(4))
                cur_fixes.append(Fix(t0, t1, x, y))
                continue

    if cur_trial is not None:
        trials[cur_trial] = TrialEye(cur_trial, cur_onset, cur_offset, cur_fixes)

    return (screen_w, screen_h), trials


# =========================
# Coordinate transforms
# =========================
def eyelink_px_to_centered_px(
    x_px: float, y_px: float, screen_w: int, screen_h: int
) -> Tuple[float, float]:
    return x_px - screen_w / 2.0, y_px - screen_h / 2.0


def centered_px_to_grid_cells(x_c: float, y_c: float) -> Tuple[float, float]:
    gx = (x_c - GRID_OFFSET_X) / CELL_SIZE_PX
    gy = (y_c - GRID_OFFSET_Y) / CELL_SIZE_PX
    return gx, gy


def eyelink_px_to_grid_cells(
    x_px: float, y_px: float, screen_w: int, screen_h: int
) -> Tuple[float, float]:
    x_c, y_c = eyelink_px_to_centered_px(x_px, y_px, screen_w, screen_h)
    return centered_px_to_grid_cells(x_c, y_c)


# =========================
# dt estimation + gaze series
# =========================
def estimate_dt_from_positions(obj_xy_cells: np.ndarray, speed_px_s: int) -> Optional[float]:
    if speed_px_s <= 0:
        return None
    d = np.linalg.norm(np.diff(obj_xy_cells, axis=0), axis=2)  # [T-1,N] cells
    d_flat = d.reshape(-1)
    d_flat = d_flat[d_flat > 1e-6]
    if len(d_flat) == 0:
        return None
    disp_cells = float(np.median(d_flat))
    disp_px = disp_cells * CELL_SIZE_PX
    return float(disp_px / float(speed_px_s))


def dt_from_duration(obj_xy_cells: np.ndarray, duration_s: float = 3.5) -> float:
    T = obj_xy_cells.shape[0]
    return duration_s / max(1, (T - 1))


def gaze_series_cells_from_fixations(
    tr_eye: TrialEye,
    Tstim: int,
    dt_s: float,
    screen_w: int,
    screen_h: int,
) -> np.ndarray:
    gaze_cells = np.full((Tstim, 2), np.nan, dtype=float)
    if tr_eye.onset_ms is None:
        return gaze_cells

    t_stim_ms = tr_eye.onset_ms + (np.arange(Tstim) * dt_s * 1000.0)

    fixes = tr_eye.fixes
    j = 0
    last = None
    for k, tk in enumerate(t_stim_ms):
        while j < len(fixes) and fixes[j].t1 < tk:
            j += 1
        if j < len(fixes) and fixes[j].t0 <= tk <= fixes[j].t1:
            gx, gy = eyelink_px_to_grid_cells(
                fixes[j].x_px, fixes[j].y_px, screen_w, screen_h
            )
            gaze_cells[k] = (gx, gy)
            last = gaze_cells[k].copy()
        elif last is not None:
            gaze_cells[k] = last
    return gaze_cells


def _gaze_change_flags(gaze_cells: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """
    Returns change[t] = 1 if gaze at t differs from gaze at t-1, else 0.
    First sample is 0 by definition.
    """
    T = gaze_cells.shape[0]
    change = np.zeros(T, dtype=int)
    for t in range(1, T):
        g0 = gaze_cells[t - 1]
        g1 = gaze_cells[t]
        if np.any(np.isnan(g0)) or np.any(np.isnan(g1)):
            change[t] = 0
        else:
            change[t] = int(np.linalg.norm(g1 - g0) > tol)
    return change


# =========================
# File collection
# =========================
def collect_visibility_files(vis_dir: str) -> Dict[str, List[str]]:
    files = glob.glob(os.path.join(vis_dir, "visibility_kh*_vt*_*.csv"))
    by_pp: Dict[str, List[str]] = {}
    for f in files:
        pp = participant_from_name(f)
        by_pp.setdefault(pp, []).append(f)
    return by_pp


def collect_search_pairs(search_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Pairs:
      results_khX_dtY_...csv
    with:
      khX_dtY_yyyy_mm_dd_hh_mm(.asc)
    by extracting a shared stem from the CSV filename.
    """
    csvs = glob.glob(os.path.join(search_dir, "results_kh*_dt*_*.csv"))

    asc_files = glob.glob(os.path.join(search_dir, "*.asc")) + glob.glob(
        os.path.join(search_dir, "*.ASC")
    )
    asc_map = {os.path.basename(a).lower(): a for a in asc_files}

    by_pp: Dict[str, List[Tuple[str, str]]] = {}
    paired = 0
    missing = 0

    stem_re = re.compile(
        r"(kh\d+_dt\d+_\d{4}_\d{2}_\d{2}_\d{2}_\d{2})(?:_\d{2})?",
        re.IGNORECASE,
    )

    for csv in csvs:
        base = os.path.basename(csv).lower()
        m = stem_re.search(base)
        if not m:
            print("[PAIR SKIP] Could not extract stem from:", os.path.basename(csv))
            missing += 1
            continue

        stem = m.group(1)
        candidate1 = f"{stem}.asc"
        candidate2 = f"{stem}.ASC".lower()

        asc = asc_map.get(candidate1) or asc_map.get(candidate2)
        if not asc:
            matches = [a for k, a in asc_map.items() if k.startswith(stem)]
            if matches:
                asc = matches[0]

        if not asc:
            print("[PAIR SKIP] Missing ASC for:", os.path.basename(csv), " | stem:", stem)
            missing += 1
            continue

        pp = stem.split("_")[0]
        by_pp.setdefault(pp, []).append((csv, asc))
        paired += 1

    print(
        f"[PAIRING] Search CSVs: {len(csvs)} | Paired with ASC: {paired} | Missing ASC: {missing}"
    )
    return by_pp


# =========================
# Debug
# =========================
def debug_inventory(search_dir, visibility_dir):

    search_csvs = glob.glob(os.path.join(search_dir, "results_kh*_dt*_*.csv"))
    vis_csvs = glob.glob(os.path.join(visibility_dir, "visibility_kh*_vt*_*.csv"))

    print("\n=== DEBUG INVENTORY ===")
    print("Search dir:", search_dir)
    print("Visibility dir:", visibility_dir)
    print("Found search CSVs:", len(search_csvs))
    print("Found visibility CSVs:", len(vis_csvs))

    for f in search_csvs[:3]:
        print("  search:", os.path.basename(f))
    for f in vis_csvs[:3]:
        print("  vis:", os.path.basename(f))

    def pp(fname):
        m = re.search(r"(kh\d+)", os.path.basename(fname).lower())
        return m.group(1) if m else None

    search_pp = sorted({pp(f) for f in search_csvs if pp(f)})
    vis_pp = sorted({pp(f) for f in vis_csvs if pp(f)})

    print("Participants in search:", search_pp[:10], ("..." if len(search_pp) > 10 else ""))
    print(
        "Participants in visibility:",
        vis_pp[:10],
        ("..." if len(vis_pp) > 10 else ""),
    )

    common = sorted(set(search_pp) & set(vis_pp))
    print("Participants in BOTH:", common[:10], ("..." if len(common) > 10 else ""))
    print("=======================\n")


def stable_trial_seed(pp: str, csv_basename: str, trial: int) -> int:
    seed_str = f"{pp}|{csv_basename}|{trial}"
    return zlib.crc32(seed_str.encode("utf-8")) & 0xFFFFFFFF


# =========================
# Gaze null-models
# =========================
def perturb_gaze(gaze_cells, mode, rng, dt_s=None):
    """
    Returns gaze_cells_used: [T,2] in grid coords.

    Operates at a fixation cadence (250 ms).
    Forces the first fixation to start at screen centre.

    Modes:
      - real: observed gaze (as reconstructed)
      - shuffle_time: shuffle fixation blocks (preserves fixation-like structure)
      - shift_time: circular shift of fixation blocks
      - center: constant central gaze
      - corner_tl: constant top-left
      - random: random fixation locations, held for ~250 ms each
      - ideal_coverage: coverage-maximizing fixation locations, held for ~250 ms each
    """
    T = gaze_cells.shape[0]

    # ---------- shared cadence ----------
    fix_dur_s = 0.25  # 250 ms
    if dt_s is None or dt_s <= 0:
        frames_per_fix = 5  # fallback (typical dt ~0.05s)
    else:
        frames_per_fix = int(np.clip(round(fix_dur_s / dt_s), 3, 8))
    n_fixes = int(np.ceil(T / frames_per_fix))
    # ----------------------------------

    center_xy = np.array([GRID_SIZE_X / 2.0, GRID_SIZE_Y / 2.0], dtype=float)

    def expand_fix_centers(fix_centers):
        """Expand [n_fixes,2] fixation centers to per-frame gaze [T,2]."""
        g = np.zeros((T, 2), dtype=float)
        for fi, c in enumerate(fix_centers):
            t0 = fi * frames_per_fix
            t1 = min(T, (fi + 1) * frames_per_fix)
            g[t0:t1, :] = c
        return g

    if mode == "real":
        return gaze_cells

    # ----- fixed policies (not superhuman) -----
    if mode == "center":
        g = np.zeros_like(gaze_cells, dtype=float)
        g[:, 0] = center_xy[0]
        g[:, 1] = center_xy[1]
        return g

    if mode == "corner_tl":
        g = np.zeros_like(gaze_cells, dtype=float)
        g[:, 0] = 0.0
        g[:, 1] = 0.0
        return g

    # ----- build fixation-block representation of REAL gaze -----
    # for shuffle/shift at fixation cadence
    real_fix = np.zeros((n_fixes, 2), dtype=float)
    for fi in range(n_fixes):
        t0 = fi * frames_per_fix
        t1 = min(T, (fi + 1) * frames_per_fix)
        block = gaze_cells[t0:t1, :]
        chosen = None
        for row in block:
            if not np.any(np.isnan(row)):
                chosen = row
                break
        if chosen is None:
            chosen = center_xy
        real_fix[fi] = chosen

    # Force start-at-center for synthetic reorderings 
    real_fix_centered = real_fix.copy()
    real_fix_centered[0] = center_xy

    if mode == "shuffle_time":
        if n_fixes <= 1:
            return expand_fix_centers(real_fix_centered)
        idx_rest = rng.permutation(np.arange(1, n_fixes))
        idx = np.concatenate(([0], idx_rest))
        return expand_fix_centers(real_fix_centered[idx])

    if mode == "shift_time":
        if n_fixes <= 1:
            return expand_fix_centers(real_fix_centered)
        shift = int(rng.integers(0, n_fixes - 1))  # shift only the rest
        rest = np.roll(real_fix_centered[1:], shift=shift, axis=0)
        fix_centers = np.vstack([center_xy[None, :], rest])
        return expand_fix_centers(fix_centers)
    
    if mode == "random_from_real_hist":
        # Sample fixations from the participant's empirical fixation distribution (with replacement).
        # Preserves WHERE ppl tend to look, destroys scanpath order/structure.
        fix_centers = np.zeros((n_fixes, 2), dtype=float)
        fix_centers[0] = center_xy
    
        if n_fixes > 1:
            pool = real_fix_centered[1:]  # exclude forced first-center
            if len(pool) == 0:
                pool = np.array([center_xy], dtype=float)
    
            idx = rng.integers(0, len(pool), size=n_fixes - 1)
            fix_centers[1:] = pool[idx]
    
        return expand_fix_centers(fix_centers)

    # ----- random fixation policy (human-like cadence), start at center -----
    if mode == "random":
        fix_centers = np.zeros((n_fixes, 2), dtype=float)
        fix_centers[0] = center_xy
        if n_fixes > 1:
            fix_centers[1:, 0] = rng.uniform(0, GRID_SIZE_X, size=n_fixes - 1)
            fix_centers[1:, 1] = rng.uniform(0, GRID_SIZE_Y, size=n_fixes - 1)
        return expand_fix_centers(fix_centers)

    # ----- coverage-maximizing fixation policy, start at center -----
    """
    Picks a point whose nearest neighbor among recent fixations is as far away as possible.
    """
    if mode == "ideal_coverage":
        step = 3
        memory_fix = 20

        xs = np.arange(0, GRID_SIZE_X, step, dtype=float)
        ys = np.arange(0, GRID_SIZE_Y, step, dtype=float)
        cand = np.array([(x, y) for x in xs for y in ys], dtype=float)  # [M,2]
        #M = cand.shape[0]

        fix_centers = np.zeros((n_fixes, 2), dtype=float)
        fix_centers[0] = center_xy

        for fi in range(1, n_fixes):
            t0 = max(0, fi - memory_fix)
            visited = fix_centers[t0:fi]  # [m,2]

            diff = cand[:, None, :] - visited[None, :, :]
            dist2 = np.sum(diff * diff, axis=2)
            min_dist2 = np.min(dist2, axis=1)

            best = np.flatnonzero(min_dist2 == np.max(min_dist2))
            fix_centers[fi] = cand[int(best[int(rng.integers(0, len(best)))])]

        return expand_fix_centers(fix_centers)

    raise ValueError(mode)
