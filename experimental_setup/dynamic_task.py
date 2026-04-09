#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 11:28:41 2025

@author: herttaleinonen


    Provides run_dynamic_trials() for calling from main.py after EyeLink setup.
    Displays dynamic Gabor arrays on a Gaussian noise background,
    flashes a fixation cross between trials for 0.5s,
    collects responses during the trial duration window,
    provides feedback between trials, incorrect/correct based on gaze response.
    
"""

import os
import csv
import random
import numpy as np
from psychopy import visual, core, event
from config import (
    grid_size_x, grid_size_y, cell_size, DIAGONAL_SCALE,
    num_trials, trial_duration, feedback_duration, timeout_feedback_text,
    orientations, spatial_frequencies, target_orientation,
    target_sf, transition_steps, movement_delay
)

# ---------- central fixation gate ----------
def wait_for_central_fixation(win, el_tracker, screen_width, screen_height,
                              deg_thresh=2.499, hold_ms=200,
                              cross_height=40, cross_color='black',
                              max_wait_s=None):
    """
    Blocks next trial from starting until gaze is within `deg_thresh` of screen center for >= `hold_ms`.
    Returns (ok, drift_deg) where drift_deg is the median distance (deg) during the final hold.

    If `max_wait_s` is None → wait indefinitely. If timed out → returns (False, "").
    If no tracker → shows cross for 0.5 s and returns (True, "").
    """
    cross = visual.TextStim(win, text='+', color=cross_color, height=cross_height, units='pix')

    # No tracker → just show for a moment and continue
    if not el_tracker:
        cross.draw(); win.flip(); core.wait(0.5)
        return True, ""

    # Start a short dedicated recording for the gate
    try:
        el_tracker.setOfflineMode()
        el_tracker.sendMessage('FIXGATE_START')
        el_tracker.startRecording(1, 1, 1, 1)
        core.wait(0.1)
    except Exception:
        # If recording fails, show cross for 0.5s
        cross.draw(); win.flip(); core.wait(0.5)
        return True, ""

    ok = False
    hold_clock = core.Clock()
    total_clock = core.Clock()
    inside_since = None
    inside_samples_deg = []

    # px/deg from your config: 1 deg ≈ `cell_size` pixels
    px_per_deg = float(cell_size)
    if px_per_deg <= 0:
        px_per_deg = 1.0  # safety

    try:
        while True:
            # Optional timeout
            if (max_wait_s is not None) and (total_clock.getTime() >= max_wait_s):
                break

            # Draw cross
            cross.draw()
            win.flip()

            # Read newest sample
            s = el_tracker.getNewestSample()
            eye = s.getRightEye() if (s and s.isRightSample()) else (s.getLeftEye() if (s and s.isLeftSample()) else None)

            now = core.getTime()
            if eye is not None:
                rx, ry = eye.getGaze()  # screen px, (0,0)=top-left
                if (rx is not None) and (ry is not None) and (rx > -1e5) and (ry > -1e5):
                    # Convert to centered px (x right, y up), then to deg
                    cx = float(rx) - (screen_width / 2.0)
                    cy = (screen_height / 2.0) - float(ry)
                    dist_deg = float(np.hypot(cx, cy) / px_per_deg)

                    if dist_deg <= deg_thresh:
                        if inside_since is None:
                            inside_since = now
                            inside_samples_deg = [dist_deg]
                            hold_clock.reset()
                        else:
                            inside_samples_deg.append(dist_deg)

                        # Has the dwell lasted long enough?
                        if (hold_clock.getTime() * 1000.0) >= float(hold_ms):
                            ok = True
                            break
                    else:
                        inside_since = None
                        inside_samples_deg = []

            core.wait(0.005)  # be gentle with CPU/GPU
    finally:
        try:
            el_tracker.sendMessage('FIXGATE_END')
            el_tracker.stopRecording()
        except Exception:
            pass

    if not ok:
        return False, ""

    drift_deg = round(float(np.median(inside_samples_deg)), 3) if inside_samples_deg else ""
    return True, drift_deg


# --------- Optional Cedrus–response box setup ---------
try:
    import pyxid2
except Exception:
    pyxid2 = None

def _cedrus_open():
    """Return first Cedrus device or None. Never raises."""
    if pyxid2 is None:
        return None
    try:
        devs = pyxid2.get_xid_devices()
        if not devs:
            return None
        dev = devs[0]
        if hasattr(dev, "reset_base_timer"): dev.reset_base_timer()
        if hasattr(dev, "reset_rt_timer"):   dev.reset_rt_timer()
        if hasattr(dev, "clear_response_queue"): dev.clear_response_queue()
        return dev
    except Exception as e:
        print(f"[Cedrus] init failed: {e}")
        return None
    
def _cedrus_flush(dev, dur=0.12):
    """Drain any pending Cedrus events for ~dur seconds."""
    if not dev:
        return
    try:
        t0 = core.getTime()
        while (core.getTime() - t0) < dur:
            if hasattr(dev, "poll_for_response"):
                dev.poll_for_response()
            while dev.has_response():
                dev.get_next_response()  # pop everything
            core.wait(0.005)
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
    except Exception as e:
        print(f"[Cedrus] flush failed: {e}")


def _cedrus_any_pressed(dev) -> bool:
    """True if *any* Cedrus key press event is available."""
    if not dev:
        return False
    try:
        if hasattr(dev, "poll_for_response"):
            dev.poll_for_response()
        if not dev.has_response():
            return False

        pressed_seen = False
        while dev.has_response():
            r = dev.get_next_response()
            if bool(r.get("pressed", False)):
                pressed_seen = True

        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return pressed_seen
    except Exception as e:
        print(f"[Cedrus] poll failed: {e}")
        return False
    
    
def _cedrus_get_choice(dev):
    """
    Returns 'target' for GREEN (key 3), 'distractor' for RED (key 1), or None.
    """
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


# -------- Helper functions --------

def gaze_pix_to_grid(px, py, grid_offset_x, grid_offset_y, cell_size):
    gx = (px - grid_offset_x) / cell_size
    gy = (py - grid_offset_y) / cell_size
    return gx, gy

noise_grain = 3  # pixel x pixel

def generate_noise(screen_width, screen_height, grain_size=noise_grain):
    h_grains = int(np.ceil(screen_height / grain_size))
    w_grains = int(np.ceil(screen_width / grain_size))
    small = np.random.normal(loc=0, scale=0.3, size=(h_grains, w_grains))
    small = np.clip(small, -1, 1)
    noise = np.repeat(np.repeat(small, grain_size, axis=0), grain_size, axis=1)
    return noise[:screen_height, :screen_width]

def grid_to_pixel(x, y, offset_x, offset_y, cell_size):
    return (offset_x + x * cell_size, offset_y + y * cell_size)

def get_valid_moves(x, y, last_move):
    all_moves = [
        (4, 0), (-4, 0), (0, 4), (0, -4),
        (DIAGONAL_SCALE, DIAGONAL_SCALE), (-DIAGONAL_SCALE, DIAGONAL_SCALE),
        (DIAGONAL_SCALE, -DIAGONAL_SCALE), (-DIAGONAL_SCALE, -DIAGONAL_SCALE),
    ]
    return [
        (dx, dy) for dx, dy in all_moves
        if (dx, dy) != (-last_move[0], -last_move[1])
        and 2 <= x + dx < grid_size_x - 2
        and 2 <= y + dy < grid_size_y - 2
    ]

# -------- Trial loop --------
def run_dynamic_trials(win, el_tracker, screen_width, screen_height, participant_id, timestamp):
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"results_{participant_id}_{timestamp}.csv")

    grid_pixel_width = grid_size_x * cell_size
    grid_pixel_height = grid_size_y * cell_size
    grid_offset_x = -grid_pixel_width / 2
    grid_offset_y = -grid_pixel_height / 2

    noise_bank = [generate_noise(screen_width, screen_height, grain_size=3) for _ in range(30)]
    noise_frames = [
        visual.ImageStim(
            win, image=img, size=(screen_width, screen_height),
            units="pix", interpolate=False
        )
        for img in noise_bank
    ]
    bank_i = 0

    ema_alpha = 0.5
    fix_radius_px = max(80, int(cell_size * 1.2))
    min_fix_frames = 1
    capture_radius_px = max(int(cell_size * 3.0), 160)
    gauss_k = 3.0
    def target_accept_radius_px():
        sigma = cell_size / 6.0
        return max(capture_radius_px, gauss_k * sigma)

    cedrus = _cedrus_open()
    if cedrus:
        print("[Cedrus] Connected. Any button will start/respond.")
    else:
        print("[Cedrus] Not found (or pyxid2 missing). Keyboard only.")

    instruction_text = visual.TextStim(
        win,
        text=("In this task you will see 10 objects, and among them you have to find a target object.\n"
              "The target object is tilted 90°.\n"
              "Press the GREEN button as soon as you see the target.\n"
              "Press the RED button, If you do not find the target.\n"
              "Try to be as accurate as possible.\n"
              "Between trials focus your eyes to the cross shown in the middle of the screen.\n"
              "\n"
              "Press any button to start."),
        color='white', height=30, wrapWidth=screen_width * 0.8, units='pix'
    )

    row_y  = -int(screen_height * 0.22)
    gap    = int(cell_size * 0.6)
    extra  = int(cell_size * 5.0)
    tgt_dy = 0
    ori_list = list(orientations)
    n_d = len(ori_list)
    row_w_d = n_d * cell_size + (n_d - 1) * gap
    start_x = -row_w_d // 2 + cell_size // 2

    example_stims = []
    rng = np.random.default_rng(0)
    for i, ori in enumerate(ori_list):
        sf = rng.choice(spatial_frequencies)
        g = visual.GratingStim(win, tex='sin', mask='gauss', size=cell_size,
                               sf=sf, ori=ori, phase=0.25, units='pix')
        g.pos = (start_x + i * (cell_size + gap), row_y)
        example_stims.append(g)

    tgt_x = start_x + (n_d - 1) * (cell_size + gap) + cell_size // 2 + extra
    tgt_y = row_y + tgt_dy
    tgt = visual.GratingStim(win, tex='sin', mask='gauss', size=cell_size,
                             sf=target_sf, ori=target_orientation, phase=0.25, units='pix')
    tgt.pos = (tgt_x, tgt_y)
    example_stims.append(tgt)

    lab_d = visual.TextStim(win, text="Distractors", color='white', height=22, units='pix',
                            pos=((start_x + (start_x + (n_d - 1) * (cell_size + gap))) / 2.0,
                                 row_y - int(0.12 * screen_height)))
    lab_t = visual.TextStim(win, text="Target (90°)", color='white', height=22, units='pix',
                            pos=(tgt_x, tgt_y - int(0.12 * screen_height)))

    instruction_text.draw()
    for s in example_stims: s.draw()
    lab_d.draw(); lab_t.draw()
    win.flip()

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
        if hasattr(cedrus, "clear_response_queue"):
            cedrus.clear_response_queue()
    else:
        event.waitKeys(keyList=['return', 'enter'])
    _cedrus_flush(cedrus)
    event.clearEvents(eventType='keyboard')

    
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Task Type", "Participant ID", "Trial", "Target Present", "Response", "Correct",
            "Reaction Time (s)", "Num Gabors", "Gabor Positions", "Target Trajectory",
            "Speed (px/s)", "FixOnTargetTime(s)", "LastFixIndex", 'CalibrationDrift(deg)'
        ])

        progress_text = visual.TextStim(
            win, text="", color='white', height=28,
            pos=(0, -int(screen_height * 0.18)), units='pix'
        )
        
        percent_text = visual.TextStim(
            win, text="", color='white', height=24,
            pos=(0, -int(screen_height * 0.25)), units='pix'
        )
        
        wait_text = visual.TextStim(
            win,
            text="Waiting for response...",
            color="white",
            height=32,
            units="pix"
        )

        correct_so_far = 0
        
        # exact 50/50 target-present schedule
        n_present = num_trials // 2
        present_schedule = [1]*n_present + [0]*(num_trials - n_present)
        random.shuffle(present_schedule)

        # -------- Trials --------
        for trial in range(num_trials):
            
            # wait for central fixation gate before each trial
            ok, drift_deg = wait_for_central_fixation(
                win, el_tracker, screen_width, screen_height,
                deg_thresh=2.499, hold_ms=200,  # you can tweak these
                cross_height=40, cross_color='black',
                max_wait_s=None                  # None = wait indefinitely
            )
            if not ok:
                # If max_wait_s is set and it times out, bail out cleanly
                print(f"[FIXGATE] Trial {trial+1}: timed out waiting for central fixation.")
                return filename

            if cedrus:
                _cedrus_flush(cedrus)
            event.clearEvents(eventType='keyboard')

            num_gabors = 10
            target_present = bool(present_schedule[trial])

            positions = [(random.randint(2, grid_size_x - 3), random.randint(2, grid_size_y - 3))
                         for _ in range(num_gabors)]
            target_index = random.randint(0, num_gabors - 1) if target_present else None

            n_distractors = num_gabors - (1 if target_present else 0)
            distractor_oris = random.choices(orientations, k=n_distractors)
            distractor_sfs  = random.choices(spatial_frequencies, k=n_distractors)
            
            gabors = []
            d_idx = 0
            for i in range(num_gabors):
                if target_present and (i == target_index):
                    g = visual.GratingStim(win, tex="sin", mask="gauss", size=cell_size,
                                           sf=target_sf, ori=target_orientation, phase=0.25, units='pix')
                else:
                    g = visual.GratingStim(win, tex="sin", mask="gauss", size=cell_size,
                                           sf=distractor_sfs[d_idx], ori=distractor_oris[d_idx],
                                           phase=0.25, units='pix')
                    d_idx += 1
            
                px, py = grid_to_pixel(positions[i][0], positions[i][1], grid_offset_x, grid_offset_y, cell_size)
                g.pos = (px, py)
                gabors.append(g)

            current_steps = [random.randint(0, transition_steps // 2) for _ in range(num_gabors)]
            targets = positions[:]
            last_moves = [random.choice([(4, 0), (0, 4), (0, -4),
                                         (DIAGONAL_SCALE, DIAGONAL_SCALE),
                                         (DIAGONAL_SCALE, -DIAGONAL_SCALE)]) for _ in range(num_gabors)]
            for i in range(num_gabors):
                x, y = positions[i]
                valid = get_valid_moves(x, y, last_moves[i])
                move = random.choice(valid) if valid else (0, 0)
                targets[i] = (x + move[0], y + move[1])
                last_moves[i] = move

            trial_clock = core.Clock()
            response = None
            rt = None
            gabor_trajectory = []
            target_trajectory = []

            # Don't accept responses in the first 0.5 s of the trial
            min_rt = 0.5
            resp_open = False

            el_tracker.setOfflineMode()
            el_tracker.sendCommand('clear_screen 0')
            el_tracker.sendMessage(f'TRIALID {trial + 1}')
            el_tracker.startRecording(1, 1, 1, 1)
            core.wait(0.1)
            el_tracker.sendMessage('stimulus_onset')

            dist_px = 0.0
            prev_px = None

            cluster_cx = 0.0; cluster_cy = 0.0
            cluster_len = 0
            committed_this_cluster = False
            last_committed_fix_idx = None
            last_fix_on_target_time = None
            ema_x = screen_width / 2.0
            ema_y = screen_height / 2.0

            while trial_clock.getTime() < trial_duration:
                noise_frames[bank_i].draw()
                bank_i = (bank_i + 1) % len(noise_frames)

                frame_positions = []
                for i in range(num_gabors):
                    if current_steps[i] >= transition_steps:
                        x, y = positions[i]
                        valid = get_valid_moves(x, y, last_moves[i])
                        move = random.choice(valid) if valid else (0, 0)
                        targets[i] = (x + move[0], y + move[1])
                        last_moves[i] = move
                        current_steps[i] = 0

                    t = current_steps[i] / transition_steps
                    interp_x = positions[i][0] + (targets[i][0] - positions[i][0]) * t
                    interp_y = positions[i][1] + (targets[i][1] - positions[i][1]) * t
                    gabors[i].pos = grid_to_pixel(interp_x, interp_y, grid_offset_x, grid_offset_y, cell_size)
                    frame_positions.append((round(interp_x, 2), round(interp_y, 2)))

                    current_steps[i] += 1
                    if current_steps[i] >= transition_steps:
                        positions[i] = targets[i]

                gabor_trajectory.append(frame_positions)

                if target_index is not None:
                    target_x, target_y = frame_positions[target_index]
                    target_trajectory.append((target_x, target_y))
                    tx = target_x * cell_size + grid_offset_x
                    ty = target_y * cell_size + grid_offset_y
                    if prev_px is not None:
                        dx = tx - prev_px[0]; dy = ty - prev_px[1]
                        dist_px += (dx*dx + dy*dy) ** 0.5
                    prev_px = (tx, ty)

                s = el_tracker.getNewestSample()
                eye = None
                if s and s.isRightSample():
                    eye = s.getRightEye()
                elif s and s.isLeftSample():
                    eye = s.getLeftEye()
                now_t = trial_clock.getTime()
                
                if eye is not None:
                    rx, ry = eye.getGaze()
                    if (rx is not None and ry is not None and rx > -1e5 and ry > -1e5):
                        ema_x = float(np.clip(ema_alpha*rx + (1-ema_alpha)*ema_x, 0, screen_width-1))
                        ema_y = float(np.clip(ema_alpha*ry + (1-ema_alpha)*ema_y, 0, screen_height-1))
                        gx = ema_x - (screen_width / 2.0)
                        gy = (screen_height / 2.0) - ema_y
                        dx = gx - cluster_cx; dy = gy - cluster_cy
                        inside = (dx*dx + dy*dy) <= (fix_radius_px*fix_radius_px)
                        if inside:
                            cluster_len += 1
                            w = 1.0 / max(1, cluster_len)
                            cluster_cx = (1 - w)*cluster_cx + w*gx
                            cluster_cy = (1 - w)*cluster_cy + w*gy
                            if (not committed_this_cluster) and (cluster_len >= min_fix_frames):
                                centers_pix = np.array([g.pos for g in gabors], dtype=float)
                                if centers_pix.size > 0:
                                    d2 = (centers_pix[:,0]-cluster_cx)**2 + (centers_pix[:,1]-cluster_cy)**2
                                    j = int(np.argmin(d2))
                                    current_idx = None
                                    if np.sqrt(d2[j]) <= 1.25 * target_accept_radius_px():
                                        current_idx = j
                                    if current_idx is not None:
                                        last_committed_fix_idx = current_idx
                                        if target_index is not None and current_idx == target_index:
                                            last_fix_on_target_time = now_t
                                committed_this_cluster = True
                        else:
                            cluster_len = 1
                            cluster_cx = gx
                            cluster_cy = gy
                            committed_this_cluster = False

                for g in gabors: g.draw()
                win.flip()

                # --- open the response window after min_rt ---
                if (not resp_open) and (now_t >= min_rt):
                    # Flush any accidental early presses
                    if cedrus:
                        _cedrus_flush(cedrus)
                    event.clearEvents(eventType='keyboard')
                    resp_open = True

                # Only accept responses after the minimum RT (0.5s)
                if resp_open:
                    # Cedrus first
                    choice = _cedrus_get_choice(cedrus) if cedrus else None
                    if choice is not None:
                        response = choice
                        rt = now_t  # trial time in seconds
                        break
                    
                    # Keyboard: g / r
                    keys = event.getKeys(keyList=['g','r','escape'], timeStamped=trial_clock)
                    if keys:
                        k, t0 = keys[0]
                        if k == 'escape':
                            if el_tracker:
                                el_tracker.sendMessage('stimulus_offset')
                                el_tracker.stopRecording()
                            return filename
                        if k == 'g':
                            response = 'target'; rt = t0; break
                        if k == 'r':
                            response = 'distractor'; rt = t0; break

                    # Keyboard: space / escape (if you still want this)
                    keys = event.getKeys(keyList=['space', 'escape'], timeStamped=trial_clock)
                    if keys:
                        k, t0 = keys[0]
                        if k == 'escape':
                            if el_tracker:
                                el_tracker.sendMessage('stimulus_offset')
                                el_tracker.stopRecording()
                            return filename
                        if k == 'space':
                            response = 'space'
                            rt = t0
                            break

                core.wait(movement_delay)


            el_tracker.sendMessage('stimulus_offset')
            el_tracker.stopRecording()
            
            # ---------------------- allow late responses ----------------------
            if response is None:   # only if participant didn't respond during the stimulus
                post_resp_clock = core.Clock()
                max_post_resp   = 20.0   # seconds allowed after stimulus offset
                dead_time_post  = 0.5   # responses during the first 0.5s ignored
                resp_open_post  = False
            
                # Show waiting message
                wait_text.draw()
                win.flip()
            
                while post_resp_clock.getTime() < max_post_resp:
                    t_now = post_resp_clock.getTime()
            
                    # Open the post-stimulus response window after dead_time_post
                    if (not resp_open_post) and (t_now >= dead_time_post):
                        # flush any early “too fast” responses
                        if cedrus:
                            _cedrus_flush(cedrus)
                        event.clearEvents(eventType='keyboard')
                        resp_open_post = True
            
                    # Before dead_time_post: do not accept responses
                    if not resp_open_post:
                        core.wait(0.01)
                        continue
            
                    # --- Now responses are allowed ---
            
                    # Cedrus first
                    choice = _cedrus_get_choice(cedrus) if cedrus else None
                    if choice is not None:
                        response = choice
                        # RT is trial_duration + time since post window started
                        rt = trial_duration + t_now
                        break
            
                    # Keyboard fallback (g / r / escape)
                    keys = event.getKeys(keyList=['g','r','escape'],
                                         timeStamped=post_resp_clock)
                    if keys:
                        k, t0 = keys[0]
                        if k == 'escape':
                            return filename
                        if k == 'g':
                            response = 'target'
                            rt = trial_duration + t0   # t0 is relative to post_resp_clock
                            break
                        if k == 'r':
                            response = 'distractor'
                            rt = trial_duration + t0
                            break
            
                    core.wait(0.01)
            # ------------------------------------------------------------------


            elapsed = trial_clock.getTime()
            speed_px_per_sec = dist_px / elapsed if elapsed > 0 else 0.0

            if response is None or response == "timeout":
                response = "timeout"
                rt = ""
                is_correct = False
                feedback_text = timeout_feedback_text
            else:
                if target_present:
                    is_correct = (response == 'target')
                else:
                    is_correct = (response == 'distractor')
                feedback_text = "Correct" if is_correct else "Incorrect"
            
            if is_correct:
                correct_so_far += 1

            percent_correct = 100 * (correct_so_far / (trial + 1))

            
            last_fix_on_target_time = None
            last_committed_fix_idx = None

            feedback = visual.TextStim(win, text=feedback_text, color="white", height=40, units='pix')
            
            progress_text.text = f"{trial + 1}/{num_trials}"
            percent_text.text = f"Accuracy: {percent_correct:.1f}%"
            
            feedback.draw()
            progress_text.draw()
            percent_text.draw()
            
            win.flip()
            core.wait(feedback_duration)

            
            event.clearEvents(eventType='keyboard')
            
            if cedrus:
                _cedrus_flush(cedrus)

            if response == 'target':
                resp_num = 1
            elif response == 'distractor':
                resp_num = 0
            else:
                resp_num = ""
            
            writer.writerow([
                "dynamic task", participant_id, trial + 1, int(target_present),
                resp_num,
                int(is_correct), rt,
                num_gabors, gabor_trajectory,
                target_trajectory if target_present else "",
                round(speed_px_per_sec, 2),
                round(last_fix_on_target_time, 4) if (target_present and last_fix_on_target_time is not None) else "",
                last_committed_fix_idx if (target_present and last_committed_fix_idx is not None) else "",
                drift_deg
            ])
