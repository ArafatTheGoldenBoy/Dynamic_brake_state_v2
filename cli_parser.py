import argparse
import torch

from config import *  # noqa: F401,F403


def parse_args():
    parser = argparse.ArgumentParser(description='Nearestfirst + TL/StopSign (YOLO + depth + stereo) — OOP wired (+TL, ABS, timers, stop-release fix, YOLO opts, CUDA stereo, per-class conf, presets)')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--town', type=str, default='Town10HD_Opt', help='Town name, e.g., Town03 or Town05_Opt')
    parser.add_argument('--mu', type=float, default=MU_DEFAULT, help='Road friction estimate (dry~0.9, wet~0.6, ice~0.2)')
    parser.add_argument('--abs-mode', type=str, default='adaptive', choices=['off','fixed','adaptive'],
                        help='Slip controller: off (direct brake), fixed PI ABS, or μ-adaptive PI ABS')
    parser.add_argument('--bus-latency-perception', type=float, default=0.0,
                        help='Simulated bus latency (seconds) from perception ECU to planning ECU')
    parser.add_argument('--bus-latency-planning', type=float, default=0.0,
                        help='Simulated bus latency (seconds) from planning ECU to actuation ECU')
    parser.add_argument('--bus-drop-perception', type=float, default=0.0,
                        help='Probability of dropping a perception->planning message (0-1)')
    parser.add_argument('--bus-drop-planning', type=float, default=0.0,
                        help='Probability of dropping a planning->actuation message (0-1)')
    parser.add_argument('--bus-jitter-perception', type=float, default=0.0,
                        help='Uniform jitter (seconds) applied to perception messages')
    parser.add_argument('--bus-jitter-planning', type=float, default=0.0,
                        help='Uniform jitter (seconds) applied to planning messages')
    parser.add_argument('--bus-deadline-perception', type=float, default=0.15,
                        help='Deadline (seconds) for consuming perception messages before counting a miss')
    parser.add_argument('--bus-deadline-planning', type=float, default=0.15,
                        help='Deadline (seconds) for consuming planning messages before counting a miss')
    parser.add_argument('--multiprocess-ecus', action='store_true', default=False,
                        help='Run perception/planning/actuation ECUs in separate OS processes connected by queues')
    parser.add_argument('--ecu-process-timeout', type=float, default=0.35,
                        help='Timeout (seconds) for ECU process replies before falling back to safe defaults')
    parser.add_argument('--bus-calibration-file', type=str, default=None,
                        help='Optional JSON file describing bus topic configs (drop, jitter, max_age, deadline, priority)')
    parser.add_argument('--safety-calibration-file', type=str, default=None,
                        help='Optional JSON file describing safety envelopes and freshness thresholds')
    parser.add_argument('--perception-freshness-s', type=float, default=0.35,
                        help='Max allowed age for perception outputs before they are treated as stale')
    parser.add_argument('--planning-freshness-s', type=float, default=0.35,
                        help='Max allowed age for planning outputs before they are treated as stale')
    parser.add_argument('--actuation-freshness-s', type=float, default=0.35,
                        help='Max allowed age for actuation outputs before they are treated as stale')
    parser.add_argument('--safety-ttc-floor-s', type=float, default=0.25,
                        help='TTC floor below which the safety monitor flags implausible readings')
    parser.add_argument('--safety-v-min-plausible', type=float, default=0.5,
                        help='Velocity floor for plausibility checks')
    parser.add_argument('--safety-wheel-slip-max', type=float, default=0.45,
                        help='Slip ratio bound before latching a slip fault')
    parser.add_argument('--brake-fail-safe', type=float, default=1.0,
                        help='Brake command applied in fail-safe mode when faults are latched')
    parser.add_argument('--apply-tire-friction', action='store_true',
                        help='Also set wheel.tire_friction≈mu to make the sim physically slick.')
    parser.add_argument('--persist-frames', type=int, default=2,
                        help='Consecutive frames required to confirm a stop‑sign before arming the stop latch')
    parser.add_argument('--min-aeb-speed', type=float, default=V_AEB_MIN_DEFAULT,
                        help='Minimum ego speed (m/s) before obstacle-triggered AEB can engage')
    parser.add_argument('--gate-confirm-frames', type=int, default=GATE_CONFIRM_FRAMES_DEFAULT,
                        help='Consecutive gate-hit frames required before obstacle braking is allowed')
    parser.add_argument('--ttc-confirm-s', type=float, default=TTC_CONFIRM_S_DEFAULT,
                        help='TTC threshold (seconds) that must be met before obstacle braking is allowed')
    parser.add_argument('--ttc-stage-strong', type=float, default=TTC_STAGE_STRONG_DEFAULT,
                        help='TTC threshold (seconds) to escalate from comfort to strong braking once confirmed')
    parser.add_argument('--ttc-stage-full', type=float, default=TTC_STAGE_FULL_DEFAULT,
                        help='TTC threshold (seconds) to escalate from strong to full AEB braking')
    parser.add_argument('--aeb-stage-comfort', type=float, default=BRAKE_STAGE_COMFORT_FACTOR,
                        help='Fraction of μg to request during the comfort braking stage (0..1)')
    parser.add_argument('--aeb-stage-strong', type=float, default=BRAKE_STAGE_STRONG_FACTOR,
                        help='Fraction of μg to request during the strong braking stage (0..1)')
    parser.add_argument('--aeb-ramp-up', type=float, default=AEB_RAMP_UP_DEFAULT,
                        help='Max increase rate for a_des (m/s^2 per second) when escalating braking')
    parser.add_argument('--aeb-ramp-down', type=float, default=AEB_RAMP_DOWN_DEFAULT,
                        help='Max decrease rate for a_des (m/s^2 per second) when relaxing braking')
    parser.add_argument('--calibration-file', type=str, default=None,
                        help='Optional JSON file containing validated AEB planning calibration values')
    parser.add_argument('--range-est', type=str, default='pinhole',
                        choices=['pinhole', 'depth', 'stereo', 'both'],
                        help='Distance source: monocular pinhole, CARLA depth, stereo vision, or log both (depth vs pinhole)')
    parser.add_argument('--compare-csv', type=str, default=None,
                        help='If set (range-est both/stereo), write pinhole/depth/stereo comparisons to this CSV path')
    parser.add_argument('--stereo-compare-csv', type=str, default=None,
                        help='If set and --range-est=stereo, write stereo vs depth comparisons to this CSV path')
    parser.add_argument('--depth-roi-shrink', type=float, default=DEPTH_ROI_SHRINK_DEFAULT,
                        help='ROI shrink factor (0..0.9) when sampling CARLA depth inside detection boxes')
    parser.add_argument('--stereo-roi-shrink', type=float, default=STEREO_ROI_SHRINK_DEFAULT,
                        help='ROI shrink factor (0..0.9) when sampling stereo disparity depth')
    parser.add_argument('--no-depth-viz', action='store_true',
                        help='(Deprecated) Legacy alias for --no-opencv; OpenCV windows are no longer used')
    parser.add_argument('--no-opencv', action='store_true',
                        help='(Deprecated) No effect now that depth/HUD HUD_DIST windows render inside pygame')
    parser.add_argument('--no-top-cam', action='store_true',
                        help='Disable spawning the top view camera and hide it from the HUD')
    parser.add_argument('--no-depth-cam', action='store_true',
                        help='Disable spawning the depth camera (range_est depth will be auto-fallback)')
    parser.add_argument('--no-telephoto', action='store_true',
                        help='Disable the telephoto traffic-light helper camera and detector scheduling')
    parser.add_argument('--telephoto-stride', type=int, default=TELEPHOTO_STRIDE_DEFAULT,
                        help='Run telephoto YOLO inference every N frames (>=2). Default=3')
    parser.add_argument('--telephoto-zoom', type=float, default=TELEPHOTO_DIGITAL_ZOOM_DEFAULT,
                        help='Digital zoom factor (>=1.0) for the telephoto feed (crop upper-center, resize, reuse boxes).'
                             ' Default=1.5; set to 1.0 to disable')
    parser.add_argument('--telephoto-compute-log', type=str, default=None,
                        help='Optional CSV path to log total compute time with/without telephoto assists (plus cache/skip stats)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI windows (also disables OpenCV windows)')
    parser.add_argument('--video-out', type=str, default=None,
                        help='Optional path to write an MP4 video of the front RGB view (qualitative results)')
    parser.add_argument('--yolo-img', type=int, default=640,
                        help='YOLO inference size (square). Example: 480 -> 480x480')
    parser.add_argument('--yolo-device', type=str, default='cuda',
                        help="Inference device: 'auto'|'cpu'|'cuda'|'cuda:0'")
    parser.add_argument('--yolo-half', action='store_true', help='Use FP16 if CUDA is available')
    parser.add_argument('--yolo-agnostic', action='store_true', help='Class-agnostic NMS')
    parser.add_argument('--yolo-classes', type=str, default=None,
                        help='Comma-separated class names or indices (e.g., "person,car,traffic light" or "0,2,7"). None=all')
    parser.add_argument('--yolo-class-thr', type=str, default=None,
                        help='Per-class confidence thresholds: "traffic light:0.55, stop sign:0.45"')
    parser.add_argument('--yolo-class-iou', type=str, default=None,
                        help='Per-class NMS IoU thresholds: "traffic light:0.40, person:0.55"')
    parser.add_argument('--yolo-max-det', type=int, default=200, help='Max detections per image')
    parser.add_argument('--yolo-dnn', action='store_true', help='Use OpenCV DNN backend (if supported)')
    parser.add_argument('--yolo-augment', action='store_true', help='Enable TTA/augment for detection')
    parser.add_argument('--stereo-cuda', action='store_true', help='Use OpenCV CUDA StereoBM/SGM if available')
    parser.add_argument('--stereo-method', type=str, default='bm', choices=['bm','sgm'], help='Stereo method for CUDA path')
    parser.add_argument('--min-h-override', type=str, default=None,
                        help='Per-class min box height (px): "person:18, traffic light:14"')
    parser.add_argument('--gate-frac-override', type=str, default=None,
                        help='Per-class center-band fraction (0..1): "car:0.35, person:0.45"')
    parser.add_argument('--gate-lateral-override', type=str, default=None,
                        help='Per-class lateral max in meters: "car:2.2, person:3.0"')
    parser.add_argument('--engage-override', type=str, default=None,
                        help='Per-class engage distances (m): "person:45, traffic light:55, car:80, stopsign:80"')
    parser.add_argument('--tl-unknown-conservative', action='store_true',
                        help='If a TL is detected but color is UNKNOWN and within engage distance, pre-brake conservatively')
    parser.add_argument('--preset', type=str, default=None, choices=['fast','quality','gpu480','cpu480'],
                        help='Quick config: fast|quality|gpu480|cpu480')
    parser.add_argument('--telemetry-csv', type=str, default=None,
                        help='Write telemetry CSV with control/safety signals to this path')
    parser.add_argument('--telemetry-hz', type=float, default=10.0,
                        help='Telemetry logging frequency in Hz (default 10)')
    parser.add_argument('--log-interval-frames', type=int, default=5,
                        help='Print a concise state line to the console every N frames (0 to disable)')
    parser.add_argument('--scenario-tag', type=str, default='default',
                        help='Freeform tag to identify this run/scenario in results CSVs')
    parser.add_argument('--scenario-csv', type=str, default=None,
                        help='If set, write high-level braking episode summaries to this CSV path')
    parser.add_argument('--extra-latency-ms', type=float, default=0.0,
                        help='Artificial extra latency (ms) added in safety envelope for sensitivity studies')
    parser.add_argument('--detector', type=str, default='yolo', choices=['yolo'],
                        help='Perception backend: currently only "yolo" is implemented; hook for future detectors')
    parser.add_argument('--npc-vehicles', type=int, default=15, help='Number of NPC vehicles to spawn')
    parser.add_argument('--npc-walkers', type=int, default=5, help='Number of NPC walkers to spawn')
    parser.add_argument('--npc-seed', type=int, default=None, help='Random seed for NPC spawning')
    parser.add_argument('--npc-disable-autopilot', action='store_true', help='Spawn vehicles without autopilot')
    parser.add_argument('--npc-speed-diff-pct', type=int, default=10, help='TrafficManager global percentage speed difference (0..100)')
    parser.add_argument('--use-abs-mu-feedback', action='store_true', help='Enable feedback of ABS-estimated friction coefficient to planner')

    return parser.parse_args()


def apply_preset(args):
    if not args.preset:
        return args
    p = args.preset
    if p == 'fast':
        args.yolo_img = 480
        args.yolo_device = 'cuda' if (torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu'
        args.yolo_half = (args.yolo_device.startswith('cuda'))
        args.yolo_agnostic = True
        args.yolo_max_det = 150
        args.stereo_cuda = False
        args.range_est = 'depth'
    elif p == 'quality':
        args.yolo_img = 640
        args.yolo_device = 'cuda' if (torch is not None and hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu'
        args.yolo_half = False
        args.yolo_max_det = 300
        args.yolo_augment = True
        args.stereo_cuda = True
        args.stereo_method = 'sgm'
    elif p == 'gpu480':
        args.yolo_img = 480
        args.yolo_device = 'cuda'
        args.yolo_half = True
        args.range_est = 'pinhole'
    elif p == 'cpu480':
        args.yolo_img = 480
        args.yolo_device = 'cpu'
        args.yolo_half = False
        args.range_est = 'depth'
    return args


__all__ = ["parse_args", "apply_preset"]
