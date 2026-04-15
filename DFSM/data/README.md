Data contains search- and visibility task's data per participant.

# search-folder

Contains all participants’ (N = 16) behavioural data (CSV files) and eye movement data (ASC files) from the visual search task.

## CSV Columns

- **Task Type**  
  Task condition identifier.

- **Participant ID**  
  Unique participant identifier.

- **Trial**  
  Trial number (1–100).

- **Target Present**  
  Target presence (1 = target present, 0 = target absent).

- **Target orientation**  
  Orientation of the target (0 = 90°).

- **Response**  
  Participant response (1 = target present, 0 = target absent).

- **Correct**  
  Response accuracy (1 = correct, 0 = incorrect).

- **Reaction Time (s)**  
  Reaction time (in seconds).

- **Num Gabors**  
  Number of Gabor stimuli (typically 10).

- **Gabor Positions**  
  Positions of all Gabors over time. Stored as a list of positions at each time step (shape: time × items × 2).

- **Target Trajectory**  
  Positions of the target Gabor over time (time × 2). Empty if target is absent.

- **Speed (px/s)**  
  Empirically recorded stimulus speed (in pixels per second).

- **FixOnTargetTime(s)**  
  Not used (empty column).

- **LastFixIndex**  
  Not used (empty column).

- **CalibrationDrift(deg)**  
  Median fixation deviation during the pre-trial fixation period (in degrees of visual angle).



## ASC-files

Raw eye-tracking data recorded with EyeLink during the visual search task.

These files are converted from `.EDF` format and contain time-stamped eye movement events, calibration information, and experiment messages. The ASC files include:

- **Calibration and validation logs**  
  Information about calibration quality (e.g., average error in degrees, gain, offsets).

- **MSG events (messages)**  
  Time-stamped markers sent from the experiment script, including:
  - `TRIALID`: start of a trial
  - `stimulus_onset`: stimulus presentation begins
  - `stimulus_offset`: stimulus presentation ends
  - `FIXGATE_START` / `FIXGATE_END`: fixation periods between trials

- **EFIX events (fixations)**  
  Main data used in analysis. Each line has the format:  
  `EFIX R t_start t_end duration x y pupil`  

  Where:  
  - `t_start`, `t_end` = fixation start and end time (ms)  
  - `duration` = fixation duration (ms)  
  - `x`, `y` = gaze position (in screen pixel coordinates)  
  - `pupil` = pupil size (arbitrary units)

- **ESACC events (saccades)**  
  Saccade start/end times and movement properties (e.g., amplitude, velocity).

- **SBLINK / EBLINK events (blinks)**  
  Blink start and end times.

### Coordinate system

- Gaze positions (`x`, `y`) are in **screen pixel coordinates**.  
- Origin is the **top-left corner** of the screen:  
  - x increases to the right  
  - y increases downward  

- Screen resolution is specified in lines such as:  
  `MSG ... GAZE_COORDS 0.00 0.00 1919.00 1199.00`

### Example structure

A trial contains:

1. `TRIALID` → trial begins  
2. `stimulus_onset` → stimulus appears  
3. Sequence of:
   - `EFIX` (fixations)  
   - `ESACC` (saccades)  
4. `stimulus_offset` → stimulus disappears  

### Notes

- The files contain **continuous event streams**, not pre-segmented trials.  
  Trial structure must be reconstructed using `MSG` markers.

- Fixations (`EFIX`) are the primary input for gaze analyses.






# visibility-folder

All participants’ (N=16) behavioural data from the visibility task (CSV files).

## Columns

- **trial**  
  Trial number.

- **ecc_deg_intended, ecc_deg_actual**  
  Intended and actual eccentricity of the Gabor stimulus (in degrees of visual angle).

- **angle_start_deg, angle_end_deg**  
  Polar angle (in degrees) of the Gabor position relative to fixation at stimulus onset and offset.

- **orientation_deg**  
  Orientation of the Gabor stimulus (in degrees; target = 90°).

- **stim_type**  
  Stimulus category (1 = target, 0 = distractor).

- **x_start_deg, y_start_deg**  
  Gabor position at stimulus onset, in Cartesian coordinates (degrees of visual angle).

- **x_end_deg, y_end_deg**  
  Gabor position at the end of the motion interval (degrees of visual angle).

- **rot_dir**  
  Direction of motion along the circular trajectory (`cw` = clockwise, `ccw` = counterclockwise).

- **moved_angle_deg**  
  Angular distance travelled along the circular trajectory (in degrees).

- **gabor_px**  
  Size of the Gabor stimulus (in pixels).

- **stim_speed_px_s**  
  Stimulus speed (in pixels per second; values: 0, 100, 200, 300 or 400).

- **response**  
  Participant response (1 = target, 0 = distractor).

- **correct**  
  Response accuracy (1 = correct, 0 = incorrect).

- **rt**  
  Reaction time (in seconds).

- **CalibrationDrift(deg)**  
  Median fixation deviation during the pre-trial fixation period (in degrees of visual angle).
