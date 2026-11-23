"""Global settings for stat_patrol. Edit here, then run any stage script."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
FISH_IMAGES_DIR = PROJECT_DIR.parent / "fish_images"
OUTPUT_DIR = PROJECT_DIR / "output"

# ---------------------------------------------------------------------------
# Input for the single-frame scripts. Relative to PROJECT_DIR.
# ---------------------------------------------------------------------------
SOURCE = "videos/5/GX010985.MP4"
FRAME_INDEX = 750

# ---------------------------------------------------------------------------
# Experiment run (run_experiment.py)
# ---------------------------------------------------------------------------
EXPERIMENT_DIR = "videos/40"   # one experiment = one folder of clips
VIDEO = "GX011097"            # one video inside EXPERIMENT_DIR; results -> output/<experiment>/<video>/
# Table inside EXPERIMENT_DIR naming which clips to use and the relevant part
# of each, in seconds: column A video, B start, C end (header row optional).
# Frames are sampled only between start and end. First file found is used.
CLIPS_TABLE_NAMES = ("TIMES.xlsx", "TIMES.csv")

# True: only print the footprint calculation at the start and stop.
PRE_CALCULATIONS = False
# POST EXPERIMENT - estimate fish count and biomass from measurements that
# already exist. Nothing is re-measured.
#   input : output/<experiment>/<video>/detections.csv + frames.csv
#           (the folder of VIDEO)
#   steps : 1. reference = whole fish, measured ok, not cut_off
#              -> K_mean, L_mean, W_mean (mean of individual weights)
#           2. whole fish count 1 (also cut_off / failed ones);
#              partial fish count K / K_mean (failed -> 0, > 1 capped at 1)
#           3. count per frame = n_whole + sum of partial fractions;
#              mean over all sampled frames (empty frames count as 0)
#           4. population = mean count * POND_AREA_M2 / footprint, for two
#              footprints: on the floor and in the fish plane
#              (H - K_HEIGHT * L_mean); biomass = population * W_mean
#           5. error against the actual count (= experiment folder name)
#   output: output/<experiment>/<video>/summary.xlsx
#           (sheets: summary, frames, detections, per_clip)
# True  = only this step. False = measure first; the summary is still written
# at the end of the run. Needs POND_AREA_M2 for population and biomass.
POST_EXPERIMENT = False

N_FRAMES = 40                  # random frames sampled per clip
RANDOM_SEED = 0                # same seed -> same frames (allows resuming)

# Overlay images (boxes, masks, midlines) saved to output/<exp>/<video>/overlays/
#   None  - save none
#   "ALL" - save every processed frame (~6 MB each)
#   int N - save the N frames with the most detections, per video
#   "SELECTED" - per video: the first 3 and last 3 sampled frames (time
#                order) + the 2 frames in between with the most detections (8)
SAVE_OVERLAYS = "ALL"

# ---------------------------------------------------------------------------
# Robust runs (robust_runs.py): every video N times with different seeds
# ---------------------------------------------------------------------------
# Experiment folders (one or a list). Every video of each TIMES table is run.
ROBUST_DIRS = ["videos/10"]
ROBUST_N_RUNS = 3               # runs per video, each with its own random seed
# Videos with avoidance degree (TIMES column D) <= this enter the per-fish-
# count average; videos without a value in D are left out (warning).
ROBUST_AVOID_THRESHOLD = 0
# Overlays per run, as SAVE_OVERLAYS (None / "ALL" / "SELECTED" / int).
# "ALL" costs ~6 MB per frame per run (x2 with overlays_counted).
ROBUST_SAVE_OVERLAYS = "SELECTED"
# Night enhancement is set per experiment by robust_runs.py itself:
# on from 40 fish up (NIGHT_FROM_FISH there), NIGHT_ENHANCE above is ignored.
# Output: output/robust/<experiment>/<video>/seed_<seed>/ (+ seeds.txt,
# robust_<video>.xlsx) and output/robust/<experiment>/robust_<experiment>.xlsx

# ---------------------------------------------------------------------------
# Models (the existing ones in fish_images)
# ---------------------------------------------------------------------------
YOLO_WEIGHTS = FISH_IMAGES_DIR / "model.pt"
SAM_WEIGHTS = FISH_IMAGES_DIR / "sam2_b.pt"
CONF_THRESHOLD = 0.25          # detector confidence threshold

# Brightening for dark (night) videos: gamma on the Lightness channel,
# applied AFTER the existing preprocessing (fishmeasure apply_filters:
# CLAHE + sharpen), so it reaches YOLO and SAM. L' = 255 * (L/255) ** gamma.
#   1.0 = off (the original pipeline)    < 1 = brighter    > 1 = darker
#   e.g. gamma 0.7: L 50 -> 82, 100 -> 132, 200 -> 215, 255 stays 255
# Calibrate per video with brighten_preview.py (BRIGHTEN_PREVIEW_GAMMAS).
# Use a new output folder when changing it: the value is not stored in the CSVs.
BRIGHTEN_GAMMA = 1.0
# gammas compared by brighten_preview.py on SOURCE / FRAME_INDEX
BRIGHTEN_PREVIEW_GAMMAS = [1.0, 0.85, 0.7, 0.55, 0.4]

# Night enhancement for the dark night videos (run_experiment.py only;
# robust_runs.py switches it on by itself from 40 fish up).
# True replaces the original preprocessing with:
#   denoise (bilateral) -> CLAHE on L -> wide-radius sharpen -> gamma NIGHT_GAMMA
# (count_frame.night_filters). BRIGHTEN_GAMMA is ignored while it is on.
# On 6 night frames: whole/partial detections 6/8 -> 12/23 (sharpen_check.py).
# Feeds both YOLO and SAM. Keep it False for day videos (not tested there).
# Use a new output folder when switching it: it is not stored in the CSVs.
NIGHT_ENHANCE = False
NIGHT_GAMMA = 0.85

# Exact class names in model.pt. Matched with ==, never with "in":
# "fish" is a substring of "partial fish" (bug 1a in fish_images' HANDOVER).
WHOLE_CLASS = "fish"
PARTIAL_CLASS = "partial fish"

NUM_KEYPOINTS = 9

# ---------------------------------------------------------------------------
# Camera and calibration
# ---------------------------------------------------------------------------
CAMERA_HEIGHT_MM = 524.20      # lens to floor
POND_AREA_M2 = 1.0899            # tank floor area in m^2 - fill in

# Wall correction. The tank is small relative to the footprint, so on some
# patrol lines part of the frame is tank wall, not floor. Each entry is
# (share of the patrol, fraction of the footprint that is floor):
#   4/7 of the lines (inner) saw a full footprint of floor,
#   3/7 of the lines (outer) saw only 2/3 of it.
# Effective footprint = footprint * sum(share * fraction) / sum(share).
# Used in post_experiment.py -> wall_factor() / estimate().
WALL_CORRECTION = True
WALL_SEGMENTS=[]
WALL_SEGMENTS.append([(2 / 7, 1.0),(2 / 7, 2/3), (3 / 7, 1 / 2)]) # ORIGINAL - 0.69
WALL_SEGMENTS.append([(0.261, 0.725),(0.261, 0.976), (0.317, 0.717),(0.158,0.792)])#

# Centre-window method: one extra summary column per window (None = off).
# A number or a list of numbers; each is the central fraction of the frame
# width and height that is used (0.5 -> x and y from 25% to 75%).
# Every detection's midline is cut at the window border and the part inside
# is measured like K (K_in). Contribution to the frame count:
#   whole fish, midline entirely inside -> 1
#   whole fish, partly inside           -> K_in / its own K
#   partial fish                        -> K_in / K_mean (all reference fish)
#   never more than 1
#   failed detection (no midline) touching the window -> 0, reported in
#   summary.xlsx (summary row 'Failed detections touching window' and the
#   note column of the detections sheet)
# Weight = W_mean of all reference whole fish of the video (not only the window).
# No wall factor. Footprint = full footprint * (window area / frame area on
# the floor, through the POLY lens model; ~0.244 for 0.5).
# Needs the keypoints column in detections.csv (saved by runs from now on;
# older measurements must be re-measured for this method).
CENTER_WINDOW = [0.5, 0.7]

# Height correction d = K_HEIGHT * L. Fitted on the 39 calibration fish with
# the hand keypoints, exactly as error_budget_stage4_weight.py does it
# (that script uses leave-one-out; this is the full-pool value).
K_HEIGHT = 0.32318500028124814

# Length-weight relation, log10 W[g] = A + B * log10 L[mm].
# Fitted on fish.csv without fish 3, as in error_budget_stage4_weight.py.
LW_LOG10_A = -5.278911318676155
LW_B = 3.2380140830751816
