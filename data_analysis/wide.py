#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 15:37:16 2026

@author: herttaleinonen

TRIAL-LEVEL DATASET (search tasks + visibility tasks): behavior + eye tracking (ASC) merged by participant+task+trial.

Input (all under ./data):
- Behavior dynamic: results_kh1_dt1....csv   (trial-level; has Target Present etc.)
- Eye tracking:     kh1_dt1....asc           (has TRIALID markers, "stimulus onset" msgs + EFIX lines)

Output:
- data/dt_trials_with_eye.csv

Run:
  cd project
  python3 wide.py
"""

import re
from pathlib import Path
import pandas as pd
import math

# --------------------------------
# Pixel to degree conversion
# --------------------------------
SCREEN_WIDTH_CM = 48
SCREEN_RESOLUTION_X = 1920
VIEWING_DISTANCE_CM = 53

CM_PER_PIXEL = SCREEN_WIDTH_CM / SCREEN_RESOLUTION_X

DEG_PER_PIXEL = 2 * math.degrees(math.atan(CM_PER_PIXEL / (2 * VIEWING_DISTANCE_CM)))

DATA_DIR = Path("data")
OUT_WIDE = DATA_DIR / "wide.csv"

# Expected tasks 
DT_TASKS = [f"dt{i}" for i in range(1, 6)]
VT_TASKS = [f"vt{i}" for i in range(1, 6)]
ALL_TASKS = DT_TASKS + VT_TASKS

# filename patterns
DT_RE = re.compile(r"(kh\d+).*?_dt(\d+)", re.IGNORECASE)   
VT_RE = re.compile(r"(kh\d+).*?_vt(\d+)", re.IGNORECASE)
ASC_RE = re.compile(r"(kh\d+).*?_dt(\d+).*\.asc$", re.IGNORECASE)

TRIALID_RE = re.compile(r"\bTRIALID\b\s*(.*)", re.IGNORECASE)

UNWANTED_COLS = {
    "Gabor Positions",
    "Target Trajectory",
    "FixOnTargetTime(s)",
    "LastFixIndex",
    "Speed (px/s)",
}

def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]  
    return df

def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in lower_map:
            return lower_map[key]
    return None

def _trial_colname(task: str, trial: int, field: str) -> str:
    return f"{task}_t{trial:03d}_{field}"

# ----------------------------
# Load Dynamic search task behavior (trial-level) -> long standardized
# ----------------------------
def load_dt_behavior_long(data_dir: Path) -> pd.DataFrame:
    files = list(data_dir.rglob("results_*_dt*.csv"))
    if not files:
        raise FileNotFoundError(f"No dynamic CSVs found under {data_dir} matching results_*_dt*.csv")

    rows = []
    for fp in files:
        m = DT_RE.search(fp.name)
        if not m:
            continue
        participant = m.group(1).lower()
        task = f"dt{int(m.group(2))}"

        df = pd.read_csv(fp, sep=None, engine="python")
        df = _clean_cols(df)
        df = df.drop(columns=[c for c in df.columns if c in UNWANTED_COLS], errors="ignore")

        trial_col = _first_existing_col(df, ["Trial", "trial"])
        if not trial_col:
            raise ValueError(f"Missing Trial column in {fp.name}. Columns: {list(df.columns)[:20]}")
        trial = pd.to_numeric(df[trial_col], errors="coerce").astype("Int64")

        tp_col = _first_existing_col(df, ["Target Present", "target_present", "target present"])
        resp_col = _first_existing_col(df, ["Response", "response"])
        corr_col = _first_existing_col(df, ["Correct", "correct"])
        rt_col   = _first_existing_col(df, ["Reaction Time (s)", "rt_s", "RT", "rt"])

        rows.append(pd.DataFrame({
            "participant": participant,
            "task": task,
            "trial": trial,
            "target_present": pd.to_numeric(df[tp_col], errors="coerce").astype("Int64") if tp_col else pd.Series([pd.NA]*len(df), dtype="Int64"),
            "response": pd.to_numeric(df[resp_col], errors="coerce").astype("Int64") if resp_col else pd.Series([pd.NA]*len(df), dtype="Int64"),
            "correct": pd.to_numeric(df[corr_col], errors="coerce").astype("Int64") if corr_col else pd.Series([pd.NA]*len(df), dtype="Int64"),
            "rt_s": pd.to_numeric(df[rt_col], errors="coerce") if rt_col else pd.Series([pd.NA]*len(df), dtype="float"),
        }))

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["participant", "task", "trial"])
    return out

# ----------------------------
# Load Visibility task behavior (trial-level) -> long standardized
# ----------------------------
def load_vt_behavior_long(data_dir: Path) -> pd.DataFrame:
    files = list(data_dir.rglob("visibility_*_vt*.csv"))
    if not files:
        raise FileNotFoundError(f"No visibility CSVs found under {data_dir} matching visibility_*_vt*.csv")

    rows = []
    for fp in files:
        m = VT_RE.search(fp.name)
        if not m:
            continue
        participant = m.group(1).lower()
        task = f"vt{int(m.group(2))}"

        df = pd.read_csv(fp, sep=None, engine="python")
        df = _clean_cols(df)

        # --- trial ---
        trial_col = _first_existing_col(df, ["trial", "Trial"])
        if not trial_col:
            raise ValueError(f"Missing trial column in {fp.name}. Columns: {list(df.columns)[:20]}")
        trial = pd.to_numeric(df[trial_col], errors="coerce").astype("Int64")

        # --- VT target present/absent (stim_type: 0/1) ---
        stim_col = _first_existing_col(df, ["stim_type", "Stim_type", "Stim Type", "stim type"])
        target_present = (
            pd.to_numeric(df[stim_col], errors="coerce").astype("Int64")
            if stim_col else pd.Series([pd.NA] * len(df), dtype="Int64")
        )

        # --- eccentricity (intended) ---
        ecc_col = _first_existing_col(df, ["ecc_deg_intended", "ecc_deg", "eccentricity"])
        ecc_deg = (
            pd.to_numeric(df[ecc_col], errors="coerce")
            if ecc_col else pd.Series([pd.NA] * len(df), dtype="float")
        )

        # --- response/correct/rt ---
        resp_col = _first_existing_col(df, ["response", "Response"])
        corr_col = _first_existing_col(df, ["correct", "Correct"])
        rt_col   = _first_existing_col(df, ["rt", "RT", "rt_s"])

        rows.append(pd.DataFrame({
            "participant": participant,
            "task": task,
            "trial": trial,
            "target_present": target_present,   # <-- filled from stim_type
            "ecc_deg": ecc_deg,                
            "response": pd.to_numeric(df[resp_col], errors="coerce").astype("Int64") if resp_col else pd.Series([pd.NA]*len(df), dtype="Int64"),
            "correct": pd.to_numeric(df[corr_col], errors="coerce").astype("Int64") if corr_col else pd.Series([pd.NA]*len(df), dtype="Int64"),
            "rt_s": pd.to_numeric(df[rt_col], errors="coerce") if rt_col else pd.Series([pd.NA]*len(df), dtype="float"),
        }))

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["participant", "task", "trial"])
    return out


# ----------------------------
# EyeLink ASC parsing -> per-trial eye metrics for dt
# ----------------------------
def parse_trial_id(trialid_tail: str):
    ints = re.findall(r"\d+", trialid_tail)
    if ints:
        return int(ints[-1])
    tail = trialid_tail.strip()
    return tail if tail else None

def parse_efix_line(line: str) -> dict | None:
    parts = line.strip().split()
    if not parts or parts[0] != "EFIX":
        return None
    idx = 1
    if idx < len(parts) and not re.fullmatch(r"-?\d+(\.\d+)?", parts[idx]):
        idx += 1  # skip eye label

    def f(i):
        try:
            return float(parts[i])
        except Exception:
            return None

    start = f(idx); idx += 1
    end   = f(idx); idx += 1
    dur   = f(idx); idx += 1
    x     = f(idx); idx += 1
    y     = f(idx); idx += 1
    pupil = f(idx) if idx < len(parts) else None

    return {"start_time": start, "end_time": end, "fix_dur": dur, "x": x, "y": y, "pupil": pupil}

def load_fixations_long(data_dir: Path) -> pd.DataFrame:
    asc_files = list(data_dir.rglob("kh*_dt*.asc")) + list(data_dir.rglob("kh*_dt*.ASC"))
    rows = []
    for fp in asc_files:
        m = ASC_RE.search(fp.name)
        if not m:
            continue
        participant = m.group(1).lower()
        task = f"dt{int(m.group(2))}"

        current_trial = None
        with fp.open("r", encoding="utf-8", errors="ignore") as f:
            for raw in f:
                line = raw.strip()
                tm = TRIALID_RE.search(line)
                if tm:
                    current_trial = parse_trial_id(tm.group(1))
                    continue
                if line.startswith("EFIX"):
                    rec = parse_efix_line(line)
                    if rec:
                        rec.update({"participant": participant, "task": task, "trial": current_trial})
                        rows.append(rec)

    fix = pd.DataFrame(rows)
    if fix.empty:
        return fix

    for c in ["start_time", "end_time", "fix_dur", "x", "y", "pupil"]:
        fix[c] = pd.to_numeric(fix[c], errors="coerce")
    fix = fix.dropna(subset=["participant", "task", "trial"])
    fix["trial"] = pd.to_numeric(fix["trial"], errors="coerce").astype("Int64")
    return fix

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def convex_hull(points):
    """Monotone chain convex hull. points: list[(x,y)] -> hull list[(x,y)] CCW."""
    pts = sorted(set(points))
    if len(pts) <= 1:
        return pts

    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]

def polygon_area(poly):
    """Shoelace formula; poly list[(x,y)]. Returns area."""
    if len(poly) < 3:
        return 0.0
    area = 0.0
    for i in range(len(poly)):
        j = (i + 1) % len(poly)
        area += poly[i][0] * poly[j][1] - poly[j][0] * poly[i][1]
    return abs(area) / 2.0

def convex_hull_area(points):
    """points list[(x,y)] -> convex hull area in same units^2."""
    if points is None or len(points) < 3:
        return 0.0
    hull = convex_hull(points)
    return polygon_area(hull)

def mean_distance_from_center_deg(points):
    """
    points: list of (x,y) in pixels
    returns mean distance from screen centre in DEGREES
    """
    if not points:
        return float("nan")

    cx = SCREEN_RESOLUTION_X / 2
    cy = 1080 / 2   

    dists = [
        math.sqrt((x - cx)**2 + (y - cy)**2)
        for (x, y) in points
    ]

    return float(pd.Series(dists).mean() * DEG_PER_PIXEL)

def compute_eye_metrics_per_trial(fix: pd.DataFrame) -> pd.DataFrame:
    if fix.empty:
        return pd.DataFrame(columns=[
            "participant","task","trial",
            "fix_count","mean_fix_dur","fix_path_length_deg","fix_dispersion_deg2"
        ])

    fix = fix.sort_values(["participant", "task", "trial", "start_time"], kind="mergesort").copy()

    # 1) fix_count + mean_fix_dur
    basic = (
        fix.groupby(["participant","task","trial"], dropna=False)
           .agg(
               fix_count=("fix_dur","size"),
               mean_fix_dur=("fix_dur","mean"),
           )
           .reset_index()
    )

    # 2) scanpath length
    fix["x_prev"] = fix.groupby(["participant","task","trial"])["x"].shift(1)
    fix["y_prev"] = fix.groupby(["participant","task","trial"])["y"].shift(1)
    dx = fix["x"] - fix["x_prev"]
    dy = fix["y"] - fix["y_prev"]
    fix["step_dist_px"] = (dx**2 + dy**2) ** 0.5

    path = (
        fix.groupby(["participant","task","trial"], dropna=False)["step_dist_px"]
           .sum(min_count=1)
           .fillna(0.0)
           .mul(DEG_PER_PIXEL)               # convert here
           .rename("fix_path_length_deg")    # store only degrees
           .reset_index()
    )

    # 3) fixation dispersion
    disp = (
        fix.groupby(["participant","task","trial"], dropna=False)
           .apply(lambda g: convex_hull_area(list(zip(g["x"], g["y"]))))
           .mul(DEG_PER_PIXEL ** 2)
           .rename("fix_dispersion_deg2")
           .reset_index()
    )
    # 4) distance from centre
    centerbias = (
        fix.groupby(["participant","task","trial"], dropna=False)
           .apply(lambda g: mean_distance_from_center_deg(list(zip(g["x"], g["y"]))))
           .rename("fix_center_dist_deg")
           .reset_index()
    )
    
    return basic.merge(path, on=["participant","task","trial"], how="outer") \
                .merge(disp, on=["participant","task","trial"], how="outer") \
                .merge(centerbias, on=["participant","task","trial"], how="outer")


# ----------------------------
# Wide conversion (forces ALL tasks)
# ----------------------------
def wide_from_long_trials(df_long: pd.DataFrame, fields: list[str], tasks: list[str]) -> pd.DataFrame:
    participants = sorted(df_long["participant"].dropna().unique().tolist())
    out = pd.DataFrame({"participant": participants}).set_index("participant")

    for task in tasks:
        sub = df_long[df_long["task"] == task].copy()

        # If no data for this task at all, skip (or create no columns)
        if sub.empty:
            continue

        max_trial = pd.to_numeric(sub["trial"], errors="coerce").max()
        if pd.isna(max_trial):

            raise ValueError(f"Task {task}: trial column could not be parsed to numbers. Check that Trial values are numeric.")

        max_trial = int(max_trial)

        for t in range(1, max_trial + 1):
            st = sub[sub["trial"] == t].set_index("participant")
            for field in fields:
                col = _trial_colname(task, t, field)
                out[col] = st[field] if field in st.columns else pd.NA

    return out.reset_index()

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError("Can't find ./data. Run from your project root.")

    dt_long = load_dt_behavior_long(DATA_DIR)
    vt_long = load_vt_behavior_long(DATA_DIR)

    fix_long = load_fixations_long(DATA_DIR)
    eye_trial = compute_eye_metrics_per_trial(fix_long)

    # merge eye metrics into dt trial rows
    if not eye_trial.empty:
        dt_long = dt_long.merge(eye_trial, on=["participant","task","trial"], how="left", validate="m:1")
    else:
        dt_long["fix_count"] = pd.NA
        dt_long["mean_fix_dur"] = pd.NA
        dt_long["fix_path_length_px"] = pd.NA

    # Wide blocks
    dt_fields = [
    "target_present","response","correct","rt_s",
    "fix_count","mean_fix_dur",
    "fix_path_length_deg","fix_dispersion_deg2",
    "fix_center_dist_deg",
    ]

    vt_fields = ["target_present", "ecc_deg", "response", "correct", "rt_s"]

    dt_wide = wide_from_long_trials(dt_long, dt_fields, DT_TASKS)
    vt_wide = wide_from_long_trials(vt_long, vt_fields, VT_TASKS)

    wide = dt_wide.merge(vt_wide, on="participant", how="outer")
    wide.to_csv(OUT_WIDE, index=False)

    # Quick diagnostics
    print("Saved:", OUT_WIDE.resolve())
    print("Participants:", wide.shape[0], "Columns:", wide.shape[1])
    print("Detected DT tasks:", sorted(dt_long["task"].unique().tolist()))
    print("Detected VT tasks:", sorted(vt_long["task"].unique().tolist()))

if __name__ == "__main__":
    main()
