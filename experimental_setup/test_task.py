#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 14:11:20 2025

@author: herttaleinonen


    Task for calibration check: 4 black squares on gray background.
    Participant moves a white '+' with their eyes; each square 'captures' with a short dwell.
    No timeout: runs until button press (ESC/any Cedrus button). Completion is evaluated at button press (all 4 captured or not). 
"""

import os
import csv
import numpy as np
from psychopy import visual, core, event

# ---------- Optional Cedrus support ----------
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

def _cedrus_any_pressed(dev) -> bool:
    """True if any Cedrus key press event is queued (drains batch)."""
    if not dev:
        return False
    try:
        if hasattr(dev, "poll_for_response"):
            dev.poll_for_response()
        if not dev.has_response():
            return False
        pressed_seen = False
        # Drain all currently queued events and detect any 'pressed'
        while dev.has_response():
            r = dev.get_next_response()
            if bool(r.get("pressed", False)):   # default False to ignore releases
                pressed_seen = True
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
        return pressed_seen
    except Exception as e:
        print(f"[Cedrus] poll failed: {e}")
        return False

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
                dev.get_next_response()
            core.wait(0.005)
        if hasattr(dev, "clear_response_queue"):
            dev.clear_response_queue()
    except Exception as e:
        print(f"[Cedrus] flush failed: {e}")
# --------------------------------------------

def run_square_test(win, el_tracker,
                    screen_width, screen_height,
                    participant_id, timestamp):

    sw, sh = screen_width, screen_height
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"results_{participant_id}_{timestamp}.csv")

    # Open Cedrus (optional)
    cedrus = _cedrus_open()
    if cedrus:
        print("[Cedrus] Connected. Any button will start/end.")
    else:
        print("[Cedrus] Not found (or pyxid2 missing). Keyboard fallback enabled.")

    # ---- visuals ----
    bg = visual.Rect(win, width=sw, height=sh, pos=(0, 0), units='pix',
                     fillColor=[0.2, 0.2, 0.2], lineColor=None)

    try:
        from config import cell_size
        sq_sz = int(max(2 * cell_size, 120))
    except Exception:
        sq_sz = 120

    half = sq_sz // 2
    offset_x = int(sw * 0.25)
    offset_y = int(sh * 0.25)
    square_positions = [(-offset_x,  offset_y),  # top-left
                        ( offset_x,  offset_y),  # top-right
                        (-offset_x, -offset_y),  # bottom-left
                        ( offset_x, -offset_y)]  # bottom-right

    squares = [
        visual.Rect(win, width=sq_sz, height=sq_sz, units='pix',
                    pos=pos, fillColor='black', lineColor=None)
        for pos in square_positions
    ]

    gaze_cross = visual.TextStim(win, text='+', color='white', height=40, units='pix')

    # Instruction text varies if Cedrus is present
    inst_text = (
        "Gaze test:\n\n"
        "Move your eyes to place the white '+' on each of the four black squares.\n"
        "Hold briefly to capture—squares turn green when captured.\n"
        "Order doesn't matter.\n\n"
        + ("Press any button to start."
           if cedrus else
           "Press Enter to start. Press ESC to end at any time.")
    )
    inst = visual.TextStim(win, text=inst_text,
                           color='white', height=30, wrapWidth=sw*0.8, units='pix')

    done_hint = visual.TextStim(
        win,
        text=("All squares captured — press any button to end."
              if cedrus else
              "All squares captured — press ESC to end."),
        color='white', height=28, pos=(0, -sh*0.35), units='pix'
    )

    # ---- Instruction screen + start gate ----
    inst.draw(); win.flip()

    if cedrus:
        # Non-blocking start gate: Cedrus OR ESC fallback
        while True:
            if _cedrus_any_pressed(cedrus):
                break
            keys = event.getKeys(keyList=['return', 'enter', 'escape'])
            if 'escape' in keys:
                return filename
            if keys:  # allow Enter fallback
                break
            core.wait(0.01)
        _cedrus_flush(cedrus)
    else:
        event.waitKeys(keyList=['return', 'enter'])
    event.clearEvents(eventType='keyboard')

    # ---- gaze params (EMA in screen coords) ----
    ema_alpha = 0.5
    ema_x, ema_y = sw/2.0, sh/2.0

    dwell_req_frames = 8  # ~130 ms @ 60 Hz
    in_counts = [0, 0, 0, 0]
    captured = [False, False, False, False]
    capture_times = [None, None, None, None]
    capture_order = []

    # ---- CSV header ----
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Task Type','Participant ID','Completed','TotalTime(s)',
            'CaptureOrder','CaptureTimes(s)','SquarePositions(px,px)'
        ])

        clk = core.Clock()
        # debounce so the same start press can't immediately end
        cedrus_arm_time = clk.getTime() + 0.12

        # EyeLink start
        if el_tracker:
            el_tracker.setOfflineMode()
            el_tracker.sendCommand('clear_screen 0')
            el_tracker.sendMessage('TRIALID 1')
            el_tracker.startRecording(1,1,1,1)
            core.wait(0.1)
            el_tracker.sendMessage('stimulus_onset')

        # ---- main loop: no timeout; ends on Cedrus (any) or ESC ----
        while True:
            # gaze sample & smoothing (screen coords)
            if el_tracker:
                s = el_tracker.getNewestSample()
                eye = None
                if s and s.isRightSample():
                    eye = s.getRightEye()
                elif s and s.isLeftSample():
                    eye = s.getLeftEye()

                if eye is not None:
                    rx, ry = eye.getGaze()
                    if (rx is not None and ry is not None and rx > -1e5 and ry > -1e5):
                        ema_x = float(np.clip(ema_alpha*rx + (1-ema_alpha)*ema_x, 0, sw-1))
                        ema_y = float(np.clip(ema_alpha*ry + (1-ema_alpha)*ema_y, 0, sh-1))

            # convert to centered coords for drawing/hit-testing
            gx = ema_x - (sw/2.0)
            gy = (sh/2.0) - ema_y
            gaze_cross.pos = (gx, gy)

            # hit test against squares; dwell to capture
            for i, (cx, cy) in enumerate(square_positions):
                if captured[i]:
                    continue
                inside = (abs(gx - cx) <= half) and (abs(gy - cy) <= half)
                if inside:
                    in_counts[i] += 1
                    if in_counts[i] >= dwell_req_frames:
                        captured[i] = True
                        capture_times[i] = clk.getTime()
                        capture_order.append(i)
                        squares[i].fillColor = 'green'
                else:
                    in_counts[i] = 0

            # draw
            bg.draw()
            for sq in squares:
                sq.draw()
            if all(captured):
                done_hint.draw()
            gaze_cross.draw()
            win.flip()

            # --- END CONDITIONS ---
            # Cedrus: any button ends (after arm time)
            if cedrus and clk.getTime() >= cedrus_arm_time and _cedrus_any_pressed(cedrus):
                break
            # Keyboard fallback: ESC ends
            if event.getKeys(keyList=['escape']):
                break

            core.wait(0.01)

        total_time = clk.getTime()

        # EyeLink stop
        if el_tracker:
            el_tracker.sendMessage('stimulus_offset')
            el_tracker.stopRecording()

        # feedback (short)
        completed = int(all(captured))
        fb_text = " " if completed else "Test ended."
        visual.TextStim(win, text=fb_text, color='white', height=40, units='pix').draw()
        win.flip(); core.wait(1.0)
        event.clearEvents(eventType='keyboard')
        if cedrus:
            _cedrus_flush(cedrus)

        # CSV row
        writer.writerow([
            'gaze cross test',
            participant_id,
            completed,
            round(total_time, 3),
            capture_order,
            [round(t, 3) if t is not None else '' for t in capture_times],
            square_positions
        ])

    return filename
