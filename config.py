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
EXPERIMENT_DIR = "videos/5"   # one experiment = one folder of clips
VIDEO = "GX010985"            # one video inside EXPERIMENT_DIR; results -> output/<experiment>/<video>/
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
#              partial fish count K / K_mean (failed -> 0, > 1 warned, kept)
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
SAVE_OVERLAYS = "ALL"

# ---------------------------------------------------------------------------
# Models (the existing ones in fish_images)
# ---------------------------------------------------------------------------
YOLO_WEIGHTS = FISH_IMAGES_DIR / "model.pt"
SAM_WEIGHTS = FISH_IMAGES_DIR / "sam2_b.pt"
CONF_THRESHOLD = 0.25          # detector confidence threshold

# Brighten every frame by this many percent BEFORE the existing preprocessing
# (fishmeasure apply_filters: CLAHE + sharpen) and therefore before YOLO and
# SAM. Each pixel value is multiplied by (1 + BRIGHTEN_PERCENT/100) and capped
# at 255. 0 = off (the original pipeline). Applied in count_frame.brighten().
BRIGHTEN_PERCENT = 0

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
#WALL_SEGMENTS = [(2 / 7, 1.0),(2 / 7, 2/3), (3 / 7, 1 / 2)] # ORIGINAL - 0.69
WALL_SEGMENTS = [(0.261, 0.725),(0.261, 0.976), (0.317, 0.717),(0.158,0.792)] #

# Height correction d = K_HEIGHT * L. Fitted on the 39 calibration fish with
# the hand keypoints, exactly as error_budget_stage4_weight.py does it
# (that script uses leave-one-out; this is the full-pool value).
K_HEIGHT = 0.32318500028124814

# Length-weight relation, log10 W[g] = A + B * log10 L[mm].
# Fitted on fish.csv without fish 3, as in error_budget_stage4_weight.py.
LW_LOG10_A = -5.278911318676155
LW_B = 3.2380140830751816
