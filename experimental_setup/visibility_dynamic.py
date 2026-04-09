#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 12:34:21 2025

@author: herttaleinonen


    Run visibility mapping and record EyeLink during the stimulus window.
    Responses are right/left arrow on keyboard or keys 1 and 3 on Cedrus.


    - Dynamic Gaussian noise background during stimulus (gabor/motion window).
    - Static noise during fixation/drift and during response.
    
    - Eccentricities are: 3, 6, 12, 16, 20 deg.
    - No eccentricity clamping. Instead, gabor is auto-scaled per trial to prevent clipping.
    
"""

import os
import random
import math
import csv
import numpy as np
from psychopy import visual, core, event

# =================== USER SETTINGS ===================
MOVE_ANGLE_DEG = 10.0          # total angular sweep over stimulus (deg of polar angle)
ROTATION_DIR   = 'random'      # 'ccw', 'cw', or 'random'

EXPERIMENT_SPEED_PX_S = 100.0  # <-- CHANGE THIS PER EXPERIMENT (px/sec)

NOISE_GRAIN = 3                # grain size (px)
NOISE_BANK_N = 30              # number of precomputed noise frames
NOISE_FRAME_SKIP = 2           # noise updates every N frames during stimulus (2 = slower)
# =====================================================

# --- debug overlay toggle helper ---
DEBUG_ECC_OVERLAY = False
DEBUG_TOGGLE_KEY  = 'd'

def _maybe_toggle_debug():
    global DEBUG_ECC_OVERLAY
    keys = event.getKeys(keyList=[DEBUG_TOGGLE_KEY])
    if keys:
        DEBUG_ECC_OVERLAY = not DEBUG_ECC_OVERLAY
        print(f"[DEBUG] Eccentricity overlay {'ON' if DEBUG_ECC_OVERLAY else 'OFF'}")

# ---------- Optional Cedrus support ----------
try:
    import pyxid2
except Exception:
    pyxid2 = None

def _cedrus_open():
    if pyxid2 is None:
        return None
    try:
        devs = pyxid2.get_xid_devices()
        if not devs:
            return None
        dev = devs[0]
        if hasattr(dev, "reset_timer"):
            dev.reset_timer()
        else:
            if hasattr(dev, "reset_base_timer"): dev.reset_base_timer()
            if hasattr(dev, "reset_rt_timer"):   dev.reset_rt_timer()
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return dev
    except Exception as e:
        print(f"[Cedrus] init failed: {e}")
        return None

def _cedrus_flush(dev, dur=0.12):
    if not dev:
        return
    try:
        t0 = core.getTime()
        while (core.getTime() - t0) < dur:
            if hasattr(dev, "poll_for_response"):
                dev.poll_for_response()
            while dev.has_response():
                dev.get_next_response()
            core.wait(0.005)
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
    except Exception as e:
        print(f"[Cedrus] flush failed: {e}")

def _cedrus_any_pressed(dev):
    if not dev:
        return False
    try:
        if hasattr(dev, "poll_for_response"):
            dev.poll_for_response()
        any_pressed = False
        while dev.has_response():
            r = dev.get_next_response()
            if r and r.get("pressed", False):
                any_pressed = True
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return any_pressed
    except Exception as e:
        print(f"[Cedrus] poll failed: {e}")
        return False

def _cedrus_get_choice(dev):
    if not dev:
        return None
    try:
        if hasattr(dev, "poll_for_response"):
            dev.poll_for_response()
        choice = None
        while dev.has_response():
            r = dev.get_next_response()
            if not r or not r.get("pressed", False):
                continue
            k = r.get("key", None)
            if k == 3:
                choice = 'target'
            elif k == 1:
                choice = 'distractor'
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return choice
    except Exception as e:
        print(f"[Cedrus] read failed: {e}")
        return None

# --------------------------------------------

def run_dynamic_visibility_trials(
    win, el_tracker, screen_width, screen_height,
    participant_id, timestamp,
    speed_px_s: float = EXPERIMENT_SPEED_PX_S
):
    screen_width_pix  = int(screen_width)
    screen_height_pix = int(screen_height)

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"visibility_{participant_id}_{timestamp}.csv")

    try:
        from config import (
            cell_size, target_orientation, target_sf, num_trials, orientations
        )
    except Exception:
        cell_size = 35.0
        target_orientation = 45
        target_sf = 1.0 / (cell_size / 2.0)
        num_trials = 150
        orientations = (170, 135, 120, 80, 20)

    px_per_deg    = float(cell_size)

    distances_deg = [3, 6, 12, 16, 20]

    distractor_oris = list(orientations)

    fixation_duration = 0.5
    stimulus_duration = 0.4

    def deg2px(x_deg, y_deg):
        return (x_deg * px_per_deg, y_deg * px_per_deg)

    # ---- Noise generation ----
    def generate_noise(screen_width, screen_height, grain_size=NOISE_GRAIN):
        h_grains = int(np.ceil(screen_height / grain_size))
        w_grains = int(np.ceil(screen_width  / grain_size))
        small = np.random.normal(loc=0, scale=0.3, size=(h_grains, w_grains))
        small = np.clip(small, -1, 1)
        noise = np.repeat(np.repeat(small, grain_size, axis=0), grain_size, axis=1)
        return noise[:screen_height, :screen_width]

    # Precompute dynamic noise bank (used only during stimulus)
    noise_bank = [generate_noise(screen_width_pix, screen_height_pix, grain_size=NOISE_GRAIN)
                  for _ in range(NOISE_BANK_N)]
    noise_frames = [
        visual.ImageStim(
            win, image=img, size=(screen_width_pix, screen_height_pix),
            units="pix", interpolate=False
        )
        for img in noise_bank
    ]
    bank_i = 0

    # 50/50 target present/absent schedule
    n_present = int(num_trials) // 2
    present_schedule = [1]*n_present + [0]*(int(num_trials) - n_present)
    random.shuffle(present_schedule)

    trials = []
    for tp in present_schedule:
        ecc = random.choice(distances_deg)  
        if tp:
            ori = int(target_orientation)
            stim_type = 'target'
        else:
            ori = int(random.choice(distractor_oris))
            stim_type = 'distractor'
        if ROTATION_DIR == 'random':
            rot_dir = random.choice(['ccw','cw'])
        else:
            rot_dir = ROTATION_DIR
        trials.append({'ecc': ecc, 'ori': ori, 'type': stim_type, 'rot_dir': rot_dir})

    cedrus = _cedrus_open()
    if cedrus:
        print("[Cedrus] Connected. GREEN(key 3)=Yes, RED(key 1)=No.")
    else:
        print("[Cedrus] Not found (or pyxid2 missing). Keyboard only.")

    fixation = visual.TextStim(win, text='+', height=1, color='black', units='deg')

    gabor = visual.GratingStim(
        win, tex='sin', mask='gauss',
        size=int(cell_size),
        sf=target_sf,
        ori=0,
        units='pix',
        phase=0.25, contrast=1.0
    )

    ecc_circle_intended = visual.Circle(
        win, radius=1.0, edges=180, units='deg',
        lineColor='white', fillColor=None, lineWidth=1.5, opacity=0.18
    )
    ecc_circle_actual = visual.Circle(
        win, radius=1.0, edges=180, units='deg',
        lineColor='white', fillColor=None, lineWidth=1.5, opacity=0.35
    )

    # ===================== Instruction screen (EXAMPLES) =====================
    inst = visual.TextStim(
        win,
        text=("In this task, a single object will briefly appear on noise at different eccentricities.\n"
              "If it is the TARGET object (90° tilt), press GREEN button.\n"
              "If it is a DISTRACTOR object (any other tilt), or you did not see it, press RED button.\n"
              "Between trials focus your eyes to the cross shown in the middle of the screen.\n\n"
              "Press any button to start."),
        color='white', height=30, wrapWidth=screen_width_pix * 0.85, units='pix',
        pos=(0, screen_height_pix * 0.24)
    )

    row_y_deg  = -2.0
    gap_deg    = 1.8
    extra_deg  = 5.0

    example_distractors = list(distractor_oris)
    n_d = len(example_distractors)

    gabor_size_d = (cell_size / px_per_deg)
    row_w_deg   = n_d * gabor_size_d + (n_d - 1) * gap_deg
    start_x_deg = -row_w_deg / 2 + gabor_size_d / 2

    example_stims = []
    for i_d, ori in enumerate(example_distractors):
        g_ex = visual.GratingStim(
            win, tex='sin', mask='gauss',
            size=int(cell_size),
            sf=target_sf,
            ori=int(ori), phase=0.25,
            units='pix', contrast=1.0
        )
        x_deg_i = start_x_deg + i_d * (gabor_size_d + gap_deg)
        g_ex.pos = deg2px(x_deg_i, row_y_deg)
        example_stims.append(g_ex)

    tgt_x_deg = start_x_deg + (n_d - 1) * (gabor_size_d + gap_deg) + gabor_size_d/2 + extra_deg
    tgt_ex = visual.GratingStim(
        win, tex='sin', mask='gauss',
        size=int(cell_size),
        sf=target_sf,
        ori=int(target_orientation), phase=0.25,
        units='pix', contrast=1.0
    )
    tgt_ex.pos = deg2px(tgt_x_deg, row_y_deg)
    example_stims.append(tgt_ex)

    lab_d = visual.TextStim(
        win, text="Distractors", color='white', height=0.8, units='deg',
        pos=((start_x_deg + (start_x_deg + (n_d - 1) * (gabor_size_d + gap_deg))) / 2.0,
             row_y_deg - 1.8)
    )
    lab_t = visual.TextStim(
        win, text="Target (90°)", color='white', height=0.8, units='deg',
        pos=(tgt_x_deg, row_y_deg - 1.8)
    )

    inst.draw()
    for s in example_stims: s.draw()
    lab_d.draw(); lab_t.draw()
    win.flip()
    # ========================================================================

    if cedrus:
        while True:
            if _cedrus_any_pressed(cedrus):
                break
            keys = event.getKeys(keyList=['return', 'enter', 'escape'])
            if 'escape' in keys:
                return filename
            if keys:
                break
            core.wait(0.01)
        _cedrus_flush(cedrus)
    else:
        event.waitKeys(keyList=['return', 'enter'])
    event.clearEvents(eventType='keyboard')

    def measure_fixation_drift(trial_idx, duration=0.5, bg=None):
        if not el_tracker:
            t0 = core.Clock()
            while t0.getTime() < duration:
                if bg: bg.draw()
                fixation.draw()
                win.flip()
            return ""
        try:
            el_tracker.setOfflineMode()
            el_tracker.sendMessage(f'FIXCHECK_START {trial_idx}')
            el_tracker.startRecording(1, 1, 1, 1)
            core.wait(0.1)
        except Exception:
            t0 = core.Clock()
            while t0.getTime() < duration:
                if bg: bg.draw()
                fixation.draw()
                win.flip()
            return ""
        samples = []
        clk_fix = core.Clock()
        while clk_fix.getTime() < duration:
            if bg: bg.draw()
            fixation.draw()
            _maybe_toggle_debug()
            win.flip()
            s = el_tracker.getNewestSample()
            eye = s.getRightEye() if (s and s.isRightSample()) else (s.getLeftEye() if (s and s.isLeftSample()) else None)
            if eye is not None:
                rx, ry = eye.getGaze()
                if (rx is not None and ry is not None and rx > -1e5 and ry > -1e5):
                    gx = float(rx) - (screen_width_pix / 2.0)
                    gy = (screen_height_pix / 2.0) - float(ry)
                    dist_deg = float(np.hypot(gx, gy)) / float(px_per_deg)
                    samples.append(dist_deg)
            core.wait(0.005)
        try:
            el_tracker.sendMessage('FIXCHECK_END')
            el_tracker.stopRecording()
        except Exception:
            pass
        return round(float(np.median(samples)), 3) if samples else ""

    # ---------------------------- Run trials ----------------------------
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            'trial',
            'ecc_deg_intended', 'ecc_deg_actual',
            'angle_start_deg', 'angle_end_deg', 'orientation_deg',
            'stim_type',
            'x_start_deg', 'y_start_deg',
            'x_end_deg', 'y_end_deg',
            'rot_dir', 'moved_angle_deg',
            'gabor_px', 'stim_speed_px_s',
            'response', 'correct', 'rt',
            'CalibrationDrift(deg)'
        ])

        for i, tr in enumerate(trials, start=1):
            stim_num = 1 if tr['type'] == 'target' else 0

            # --- STATIC noise for fixation + response ---
            static_noise_img = generate_noise(screen_width_pix, screen_height_pix, grain_size=NOISE_GRAIN)
            static_noise_stim = visual.ImageStim(
                win, image=static_noise_img,
                size=(screen_width_pix, screen_height_pix),
                units='pix', interpolate=False
            )

            drift_deg = measure_fixation_drift(i, duration=fixation_duration, bg=static_noise_stim)
            event.clearEvents(eventType='keyboard')
            if cedrus:
                _cedrus_flush(cedrus)

            ecc_intended = float(tr['ecc'])   # 3/6/12/16/20 
            ecc_actual = ecc_intended         # no clamping

            angle_start = random.uniform(0, 360)
            sign = +1 if tr['rot_dir'] == 'ccw' else -1
            angle_end = (angle_start + sign * MOVE_ANGLE_DEG) % 360
            moved_angle = abs((angle_end - angle_start + 540) % 360 - 180)

            rad_s = math.radians(angle_start)
            rad_e = math.radians(angle_end)

            x_start_deg = ecc_actual * math.cos(rad_s)
            y_start_deg = ecc_actual * math.sin(rad_s)
            x_end_deg   = ecc_actual * math.cos(rad_e)
            y_end_deg   = ecc_actual * math.sin(rad_e)

            # ----------------- AUTO-SCALE GABOR TO PREVENT CLIPPING -----------------
            ecc_px = ecc_actual * px_per_deg
            stim_px_base = float(cell_size)
            half_stim_px_base = stim_px_base / 2.0

            half_w = screen_width_pix / 2.0
            half_h = screen_height_pix / 2.0

            cosr = abs(math.cos(rad_s))
            sinr = abs(math.sin(rad_s))
            eps = 1e-6

            max_r_x = (half_w - 1) / max(cosr, eps)
            max_r_y = (half_h - 1) / max(sinr, eps)
            max_r   = min(max_r_x, max_r_y)

            allowed_half_size_px = max_r - ecc_px
            allowed_half_size_px = max(allowed_half_size_px, 5.0)

            scale_factor = min(1.0, allowed_half_size_px / half_stim_px_base)
            stim_px_actual = stim_px_base * scale_factor

            gabor.size = stim_px_actual
            # -----------------------------------------------------------------------

            clock = core.Clock()
            if el_tracker:
                el_tracker.setOfflineMode()
                el_tracker.sendCommand('clear_screen 0')
                el_tracker.sendMessage(f'TRIALID {i}')
                el_tracker.startRecording(1, 1, 1, 1)
                core.wait(0.1)

            theta_accum_rad = 0.0
            last_t = 0.0
            first_frame = True
            noise_frame_counter = 0

            while clock.getTime() < stimulus_duration:
                t_now = clock.getTime()
                dt = max(0.0, t_now - last_t)
                last_t = t_now

                if ecc_px > 1e-6:
                    theta_accum_rad += (speed_px_s * dt) / ecc_px

                angle_now = (angle_start + sign * math.degrees(theta_accum_rad)) % 360
                rad = math.radians(angle_now)
                x_deg = ecc_actual * math.cos(rad)
                y_deg = ecc_actual * math.sin(rad)

                # dynamic noise only here
                noise_frames[bank_i].draw()
                noise_frame_counter += 1
                if noise_frame_counter % NOISE_FRAME_SKIP == 0:
                    bank_i = (bank_i + 1) % len(noise_frames)

                fixation.draw()
                gabor.ori = tr['ori']
                gabor.pos = deg2px(x_deg, y_deg)
                gabor.draw()

                if DEBUG_ECC_OVERLAY:
                    ecc_circle_intended.radius = float(ecc_intended)
                    ecc_circle_intended.draw()
                    ecc_circle_actual.radius = float(ecc_actual)
                    ecc_circle_actual.draw()

                _maybe_toggle_debug()
                win.flip()

                if first_frame and el_tracker:
                    el_tracker.sendMessage('stimulus_onset')
                    first_frame = False

            if el_tracker:
                el_tracker.sendMessage('stimulus_offset')
                el_tracker.stopRecording()

            static_noise_stim.draw()
            visual.TextStim(win, text='?', height=1, color='black', units='deg').draw()
            win.flip()

            if cedrus:
                _cedrus_flush(cedrus)

            response = None
            rt = None
            while response is None:
                if cedrus:
                    choice = _cedrus_get_choice(cedrus)
                    if choice in ('target', 'distractor'):
                        response = choice
                        rt = clock.getTime()
                        break
                keys = event.getKeys(keyList=['left', 'right', 'escape'], timeStamped=clock)
                if keys:
                    key, key_time = keys[0]
                    if key == 'escape':
                        return filename
                    response = 'target' if key == 'right' else 'distractor'
                    rt = key_time
                    break
                core.wait(0.005)

            resp_num = 1 if response == 'target' else 0
            correct  = int(resp_num == stim_num)

            writer.writerow([
                i,
                ecc_intended, ecc_actual,
                round(angle_start, 2), round(angle_end, 2), tr['ori'],
                stim_num,
                round(x_start_deg, 2), round(y_start_deg, 2),
                round(x_end_deg, 2), round(y_end_deg, 2),
                tr['rot_dir'], round(moved_angle, 2),
                round(stim_px_actual, 3), float(speed_px_s),
                resp_num, correct, round(rt, 4),
                drift_deg
            ])

    return filename
