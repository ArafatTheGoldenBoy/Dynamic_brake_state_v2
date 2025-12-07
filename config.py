import math

IMG_W, IMG_H = 960, 540          # display pane size for primary/telephoto views
FOV_X_DEG = 60.0                 # front cam horizontal FOV
TELEPHOTO_IMG_W = 640
TELEPHOTO_IMG_H = 360
TELEPHOTO_FOV_X_DEG = 25.0
TELEPHOTO_STRIDE_DEFAULT = 3
TELEPHOTO_CACHE_MAX_AGE_S = 0.6
TELEPHOTO_DIGITAL_ZOOM_DEFAULT = 1.5
TELEPHOTO_DIGITAL_ZOOM_MAX = 3.5
TELEPHOTO_ZOOM_TOP_BIAS = 0.35
TL_PRIMARY_CROP_FRAC = 0.20
TL_PRIMARY_SHORT_RANGE_M = 50.0
TL_STATE_SMOOTHING_FRAMES = 5
DT = 0.02                        # 50 Hz (set 0.02 for lower latency)
FX = (IMG_W / 2.0) / math.tan(math.radians(FOV_X_DEG / 2.0))

# Pinhole range estimation + gating defaults
OBJ_HEIGHT_M = {
    'person': 1.7,
    'bicycle': 1.6,
    'car': 1.5,
    'motorcycle': 1.4,
    'motorbike': 1.4,
    'bus': 3.1,
    'truck': 3.0,
    'train': 3.5,
    'traffic light': 0.4,
    'trafficlight': 0.4,
    'stop sign': 0.75,
    'stopsign': 0.75,
}
CENTER_BAND_FRAC = 0.35
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'train', 'motorcycle', 'motorbike', 'bicycle'}
PEDESTRIAN_CLASSES = {'person'}

# Stereo camera baseline (meters)
STEREO_BASELINE_M = 0.54
STEREO_FUSE_NEAR_WEIGHT = 0.65
STEREO_FUSE_FAR_WEIGHT = 0.35
STEREO_FUSE_DISAGREE_M = 8.0
DEPTH_ROI_SHRINK_DEFAULT = 0.4
STEREO_ROI_SHRINK_DEFAULT = 0.35

# Lane gating (ignore other-lane vehicles)
LANE_HALF_WIDTH = 1.8
LATERAL_MARGIN = 0.6
LATERAL_MAX = LANE_HALF_WIDTH + LATERAL_MARGIN
H_MIN_PX = 12                     # minimum detection height to consider (pixels)

# Cruise / braking
A_MAX = 8.0                      # hard cap
B_COMFORT = 3.5                  # map a_des -> [0..1] brake
V_TARGET = 10.0                  # m/s default target
KP_THROTTLE = 0.15
EPS = 0.5
ALPHA_VBLEND = 0.7               # blend velocity estimates
S_ENGAGE = 80.0                  # generic engage for vehicles/unknown
S_ENGAGE_TL = 80.0               # traffic‑light engage distance (red)
S_ENGAGE_PED = 45.0              # pedestrian engage distance
V_STOP = 0.10
V_AEB_MIN_DEFAULT = 2.5          # minimum speed for obstacle-triggered AEB
GATE_CONFIRM_FRAMES_DEFAULT = 1  # frames of gate_hit before AEB entry
TTC_CONFIRM_S_DEFAULT = 2.5      # TTC threshold for obstacle confirmation
TTC_STAGE_STRONG_DEFAULT = 1.8   # TTC where controller escalates to strong braking
TTC_STAGE_FULL_DEFAULT = 1.0     # TTC where controller escalates to full AEB

# Multi-stage AEB shaping (fraction of μg cap)
BRAKE_STAGE_COMFORT_FACTOR = 0.45
BRAKE_STAGE_STRONG_FACTOR = 0.75
BRAKE_STAGE_FULL_FACTOR = 1.00
AEB_RAMP_UP_DEFAULT = 12.0       # max increase in a_des (m/s^2) per second
AEB_RAMP_DOWN_DEFAULT = 18.0     # max decrease in a_des (m/s^2) per second

# Clear timers (per reason)
CLEAR_DELAY_OBS = 0.9            # obstacle clear debounce
CLEAR_DELAY_RED = 0.5            # GREEN debounce to clear red‑light hold
CLEAR_DELAY_S = 3.0              # legacy/general; still used for HUD counters
STOP_WAIT_S = 5.0                # stop‑sign wait
KICK_SEC = 0.6                   # start "kick"
KICK_THR = 0.25

# Dynamic safety tuning knobs (match original)
TAU_MIN, TAU_MAX = 0.15, 1.50
K_LAT_TAU = 1.2    # sec/sec of pipeline latency
K_MU_TAU = 0.25    # extra tau per (0.9 - mu)
K_UNC_TAU = 0.35   # extra tau per (1 - conf)

D_MIN, D_MAX = 0.3, 60.0   # meters
K_LAT_D = 1.0         # meters per v*latency (reaction dist)
K_UNC_D = 4.0         # meters per 1 m of depth sigma
K_MU_D = 4.0          # meters per (0.9 - mu)

# Brake PI shaping (measured decel tracking) — original-ish gains
KPB = 0.22
KIB = 0.10
I_MAX = 8.0

# Road / tire friction & low‑μ helpers
MU_DEFAULT = 0.90
REV_PULSE_V_MAX = 2.0
REV_THR = 0.18
ABS_V_MAX = 4.0
ABS_B_MIN = 0.20
ABS_PWM_SCALE = 0.5

# False-stop heuristics for telemetry/episode labeling
FALSE_STOP_MARGIN_M = 5.0      # if actual gap exceeds safety distance by this margin while braking → suspicious
FALSE_STOP_TTC_S = 4.0         # TTC above this while brake engaged → likely false stop

# Actuation-latency measurement thresholds
ACTUATION_BRAKE_CMD_MIN = 0.18   # require brake command above this before timing
ACTUATION_DECEL_THRESH = 0.8     # m/s^2 decel magnitude that counts as "brake is biting"
ACTUATION_TIMEOUT_S = 1.5        # give up if no response within this horizon

# Default YOLO weights
YOLO_MODEL_PATH = "yolo12n.pt"
