# Dynamic Brake State v2

Intelligent braking experiments in CARLA using YOLO-based perception and configurable scene parameters.

---

## 🚀 Quick Start

```bash
python dynamic_brake_state.py --no-telephoto --town Town10HD_Opt --preset quality \
  --yolo-class-thr "traffic light:0.25,car:0.25,stop sign:0.25" \
  --min-h-override "stop sign:10,car:10,traffic light:8" \
  --gate-frac-override "car:0.5, stop sign:0.5" \
  --gate-lateral-override "car:4.0, stop sign:4.0"
```

---

## 📦 Requirements

- Python 3.8+  
- CARLA simulator  
- YOLO model weights (`yolo12n.pt`)

> Install Python dependencies (example):
> ```bash
> pip install -r requirements.txt
> ```

---

## 🎛 Key Arguments

- `--no-telephoto` – disable telephoto camera in the sensor rig.
- `--town` – CARLA town to load, e.g. `Town10HD_Opt`.
- `--preset` – rendering / simulation quality (e.g. `quality`).

### Perception & Detection

- `--yolo-class-thr "traffic light:0.25,car:0.25,stop sign:0.25"`
  - Per-class detection thresholds for YOLO.

- `--min-h-override "stop sign:10,car:10,traffic light:8"`
  - Minimum bounding box height (in pixels) for valid detections.

### Gating & Scenario Logic

- `--gate-frac-override "car:0.5, stop sign:0.5"`
  - Fractional visibility threshold to accept detections.

- `--gate-lateral-override "car:4.0, stop sign:4.0"`
  - Maximum lateral distance (m) from ego lane to consider objects.

---

## 📊 Results

Experiment CSVs are stored under `results/`, e.g.:

- `exp1_dry_yolo_depth_scenarios.csv`
- `exp1_dry_yolo_depth_telemetry.csv`
- `exp1b_dry_yolo_depth_*.csv`
- `exp1c_dry_yolo_pinhole_*.csv`

Use `results_analysis_exp1.py` for post-processing and visualization.

---

## 🧠 Project Structure

- `dynamic_brake_state.py` – main experiment runner.
- `perception_pipeline.py` – YOLO detections and range estimation.
- `planning.py` – braking / decision logic.
- `vehicle_control.py` – low-level control interface.
- `sensor_rig.py` – camera and sensor configuration.
- `world_manager.py` – CARLA world and scenario management.
- `cli_parser.py` – command line interface definition.

---

## 📄 License

This project is licensed under the MIT License – see `LICENSE` for details.
Dynamic Brake State – AEB Thesis Codebase

This repository contains the simulation and software stack used in the thesis on dynamic brake-state control and ECU-style AEB architecture in CARLA. The system emulates a realistic automotive setup with separate perception, planning, actuation ECUs, an ABS controller, and a safety manager, plus a full telemetry and scenario-logging pipeline for experiments.

The main focus is:

Comparing different range-estimation pipelines: monocular pinhole, CARLA depth, stereo, and (optionally) learned depth.

Studying dynamic safety envelopes (TTC- and uncertainty-aware) and multi-stage AEB logic.

Evaluating an ECU-style architecture with message bus latency, drop, and jitter, plus a safety manager enforcing fail-safe behaviour.

The current implementation uses YOLO as the baseline detector; the design allows you to plug in MobileNetSSD / MobileNetV4‑SSDLite later.

1. Repository Overview
Core Orchestrator

dynamic_brake_state.py
Top-level application that wires together:

CARLA world and NPC traffic (WorldManager)

Sensor suite (SensorRig)

Range estimation (RangeEstimator)

Perception ECU (YOLO + depth/stereo/pinhole)

Planning ECU (AEB planner + dynamic safety envelope)

ABS / brake ECU

Safety manager, message bus, HUD, telemetry, scenario logging

Configuration & Utilities

config.py
Global constants: image size, FOV, time step DT, baseline for stereo, class heights for pinhole ranging, AEB tuning knobs, default friction MU_DEFAULT, and defaults for YOLO model path (YOLO_MODEL_PATH = "yolo12n.pt").

camera_utils.py
Pinhole camera helpers: FOV→intrinsics, pixel→camera coordinates, and CARLA depth decoding (decode_depth_meters_from_bgra).

World & Sensors

world_manager.py
Handles CARLA world loading (including _Opt maps), synchronous ticking, weather, ego spawn, NPC vehicles/walkers, tire friction application, and collision sensing. Provides a helper lead_distance_ahead to measure true lead distance for analysis.

sensor_rig.py
Spawns and manages:

Front RGB camera

Depth camera (optional)

Stereo pair (optional)

Telephoto RGB + telephoto depth (optional)
Provides a read() method that returns a consistent set of frames for a given frame ID.

Perception

detectors.py
YOLODetector wraps the Ultralytics YOLO model, handling device selection, FP16 fallback, per-class IoU NMS, and mapping model class IDs to normalized labels (person, car, traffic light, stop sign, etc.).

perception_pipeline.py
Implements PerceptionPipeline, which:

Runs YOLO on the front camera (and optional telephoto)

Estimates distances using pinhole, CARLA depth, stereo, or fused depth

Applies class-specific gating (center band, lateral limits, min box height)

Detects traffic-light color via HSV heuristics

Produces a nearest_s_active, nearest_kind, TL state and distance, and per-object 3D points for HUD.

range_estimator.py
Stereo depth pipeline using OpenCV StereoBM or CUDA StereoBM/SGM, converting disparity to metric depth via fx * baseline / disparity.

Planning & Safety

planning.py
Contains:

PlanningContext – persistent planner state (stop latch, hold, PI integrator, timers).

AEBPlannerConfig – wraps calibration and μ, range mode, depth ROI shrink.

AEBPlanner – dynamic safety envelope, TTC-based multi-stage AEB, decel ramping, and hold-logic for stop sign and red light.

calibrations.py
Typed calibration objects (AEBPlanningCalibration, SafetyCalibration, BusTopicCalibration) with range checking and JSON loading helpers.

abs_system.py
Slip-based ABS using a PI controller (PISlipChannel) and friction regime estimator (FrictionEstimator). Provides fixed and μ-adaptive ABS actuators used by the actuation ECU.

ecu.py
ECU-style wrappers and infra:

PerceptionECU, PlanningECU, ActuationECU

MessageBus with configurable drop, jitter, deadlines, and metrics

SafetyManager to aggregate ECU faults and enforce degraded/fail-safe behaviour

Optional DistributedECUPipeline spawning separate OS processes per ECU.

HUD, Control, Logging

hud.py
Pygame HUD showing camera view(s), control state, AEB envelope (tau_dyn, D_safety_dyn, sigma_depth), ABS state, and per-object distances.

vehicle_control.py
Simple lateral controller steer_to_waypoint for lane following, plus wheel_linear_speeds which derives linear wheel speeds from wheel angular velocities and radii.

telemetry.py

TelemetryLogger: fixed-schema CSV logger for every control step (tau, D_safety, TTC, AEB stage, ABS, μ_est, etc.).

ScenarioLogger: episode-level CSV summarizing each braking scenario (initial distance, min gap, TTC margins, collision flag, false-stop flag, etc.).

cli_parser.py
Central place for CLI options: μ, ABS mode, bus latency/drop, range-est mode, YOLO options, stereo options, telemetry/scenario CSV paths, detector (currently yolo), NPC counts, etc. Also defines presets (fast, quality, gpu480, cpu480).

2. Requirements & Setup
2.1 Software Requirements

CARLA simulator (0.9.x series, synchronous mode support).

Python 3.10+ (you are currently using 3.12).

GPU with CUDA is recommended (YOLO, optional CUDA stereo).

2.2 Python Dependencies

Typical dependencies (not exhaustive):

ultralytics (YOLO)

torch

opencv-python (+ opencv-contrib-python and opencv-python-headless if needed)

pygame

numpy

Install (example):

pip install ultralytics torch torchvision torchaudio
pip install opencv-python pygame numpy


You may need additional CARLA-specific packages depending on your installation (e.g., carla egg on PYTHONPATH). carla_utils.import_carla() tries common egg locations automatically.

2.3 YOLO Weights

By default, YOLO_MODEL_PATH is set to:

YOLO_MODEL_PATH = "yolo12n.pt"


in config.py.

Place the corresponding .pt YOLO weights in the repo root or update YOLO_MODEL_PATH to your custom path.

3. Running the Simulator
3.1 Start CARLA

Start CARLA in server mode, e.g.:

On Windows: run CarlaUE4.exe -quality-level=Epic -carla-server

On Linux: ./CarlaUE4.sh -carla-server

Make sure it matches the --host and --port specified on the command line (defaults: 127.0.0.1:2000).

3.2 Basic Run (Interactive HUD)

Example: baseline YOLO + depth experiment (similar to your Exp1-A):

python dynamic_brake_state.py \
    --mu 0.9 \
    --apply-tire-friction \
    --range-est depth \
    --npc-vehicles 5 \
    --npc-walkers 0 \
    --telemetry-csv results/exp1_dry_yolo_depth_telemetry.csv \
    --scenario-csv  results/exp1_dry_yolo_depth_scenarios.csv \
    --scenario-tag  exp1_dry_yolo_depth \
    --log-interval-frames 5


Key options:

--mu 0.9 – dry asphalt friction (also applied to tire friction if --apply-tire-friction is set).

--range-est depth – use CARLA depth camera for range estimation.

--npc-vehicles, --npc-walkers – spawn background traffic.

--telemetry-csv, --scenario-csv, --scenario-tag – log detailed time-series and episodic summaries.

3.3 Other Range-Estimation Modes

You can choose among:

--range-est pinhole – monocular pinhole using canonical object heights from OBJ_HEIGHT_M.

--range-est depth – CARLA depth camera (front).

--range-est stereo – stereo disparity converted to depth; enable --stereo-cuda and choose --stereo-method bm|sgm if you have CUDA.

--range-est both – fuse depth + stereo heuristically for certain objects (in perception), while logging.

Examples:

YOLO + stereo (Exp1-B-style)
python dynamic_brake_state.py \
    --mu 0.9 \
    --apply-tire-friction \
    --range-est stereo \
    --stereo-cuda \
    --stereo-method bm \
    --npc-vehicles 5 \
    --telemetry-csv results/exp1b_dry_yolo_stereo_telemetry.csv \
    --scenario-csv  results/exp1b_dry_yolo_stereo_scenarios.csv \
    --scenario-tag  exp1b_dry_yolo_stereo

YOLO + pinhole (Exp1-C-style)
python dynamic_brake_state.py \
    --mu 0.9 \
    --apply-tire-friction \
    --range-est pinhole \
    --npc-vehicles 5 \
    --telemetry-csv results/exp1c_dry_yolo_pinhole_telemetry.csv \
    --scenario-csv  results/exp1c_dry_yolo_pinhole_scenarios.csv \
    --scenario-tag  exp1c_dry_yolo_pinhole

3.4 Headless Mode

For non-interactive runs (no HUD):

python dynamic_brake_state.py --headless ...


Pygame still initializes but does not render.

4. Command-Line Flags (Short Overview)

The full list of options is defined in cli_parser.parse_args(). Below are the most important families.

4.1 Environment & Simulation

--host, --port – CARLA server address.

--town – CARLA town name (e.g., Town10HD_Opt). WorldManager will auto-switch between _Opt and base maps when possible.

--mu – road friction coefficient (dry ~0.9, wet ~0.6, ice ~0.2).

--apply-tire-friction – actually set wheel tire friction to μ in physics.

4.2 ABS and Friction

--abs-mode off|fixed|adaptive

off – no ABS shaping; brake ECU passes through AEB brake.

fixed – PI slip control with fixed λ* and gains.

adaptive – friction-estimation + configuration switching among high, medium, low.

4.3 ECU & Bus

--multiprocess-ecus – run perception, planning, actuation in separate OS processes via DistributedECUPipeline.

--bus-latency-*, --bus-drop-*, --bus-jitter-* – inject latency, drop, jitter on perception/planning topics.

--bus-calibration-file – JSON description of topic-level configs loaded via load_bus_calibration.

4.4 AEB / Planning Calibration

--min-aeb-speed – minimum ego speed to allow obstacle-triggered AEB.

--gate-confirm-frames – frames of gate-hit before confirming an obstacle.

--ttc-confirm-s – TTC threshold for obstacle confirmation.

--ttc-stage-strong, --ttc-stage-full – TTC thresholds for strong / full AEB stage.

--aeb-stage-comfort, --aeb-stage-strong – fractions of μg for stage caps.

--aeb-ramp-up, --aeb-ramp-down – decel ramp constraints.

--calibration-file – JSON override for AEBPlanningCalibration.

4.5 Perception & YOLO

--range-est pinhole|depth|stereo|both

--stereo-cuda, --stereo-method bm|sgm

--depth-roi-shrink, --stereo-roi-shrink – ROI shrink inside detection boxes for robust depth sampling.

--no-depth-cam, --no-telephoto, --no-top-cam

YOLO-specific:

--yolo-img, --yolo-device, --yolo-half

--yolo-classes, --yolo-class-thr, --yolo-class-iou

--yolo-max-det, --yolo-dnn, --yolo-augment

4.6 Telemetry & Results

--telemetry-csv – frame-level logging (TelemetryLogger).

--telemetry-hz – logging frequency.

--scenario-csv, --scenario-tag – scenario-level summaries (ScenarioLogger).

--log-interval-frames – console logging cadence.

5. Architecture & Data Flow (High Level)

World tick (WorldManager.tick) advances the CARLA sim in synchronous mode.

SensorRig.read returns the latest consistent frame set (front RGB, depth, stereo, telephoto).

App._decode_frames converts raw CARLA images to numpy arrays and decodes depth values in meters.

PerceptionECU calls PerceptionPipeline.run, which:

Runs YOLO detector.

Computes distances per object (pinhole / depth / stereo / fusion).

Applies gating and chooses nearest active obstacle + TL state.

PlanningECU calls AEBPlanner.step, which:

Builds a dynamic safety envelope (tau_dyn, D_safety_dyn, sigma_depth, latency_s).

Computes TTC and decides whether to brake (obstacle vs stop sign vs red light).

Applies multi-stage AEB shaping and PI decel-tracking.

ActuationECU applies ABS shaping (if enabled) and returns an effective brake command with ABS debug metrics.

SafetyManager evaluates ECU health, τ/TTC plausibility, wheel-slip bounds, and may override throttle/brake in degraded or fail-safe modes.

The ego vehicle is commanded via CARLA’s VehicleControl, while the HUD and telemetry log internal state and outcomes.

6. Experiments & Evaluation

Your thesis experiments are structured roughly as:

Exp1-A: YOLO + CARLA depth

Exp1-B: YOLO + stereo

Exp1-C: YOLO + pinhole

Each experiment consists of:

Running dynamic_brake_state.py with an appropriate --range-est and CSV paths.

Post-processing telemetry and scenario CSVs with an offline analysis script (e.g. results_analysis_exp1.py) to produce:

Scenario-level metrics (collisions, false stops, min gap, TTC_min)

Telemetry summaries (TTC distribution during braking, τ_dyn, D_safety_dyn, σ_depth, etc.)

The README intentionally leaves space for additional experiments:

Exp1-D (placeholder): MobileNetSSD + depth

Exp1-E (placeholder): MobileNetV4‑SSDLite + depth (joint detection + depth)

You can add these by:

Integrating the new detector backbone (see Extending detectors below).

Running the same scenario set with identical initial conditions (same town, μ, ego speed).

Adding new result sections in the thesis and in this README (e.g., “6.4 Joint SSDLite + Depth Results”).

7. Telemetry & Scenario CSV Format
7.1 Telemetry CSV

TelemetryLogger writes a fixed schema (one row per control step):

Time, ego speed, τ_dyn, D_safety_dyn, σ_depth

AEB decel request a_des, brake command

ABS slip (lambda_max), ABS factor f_global, μ_est, ABS regime

Loop timing, detection/latency placeholders

Measured decel, estimated range, TTC

Gate hit / confirmation flags, brake stage & factor

Optional tracking diagnostics

Sensor/control/actuation timestamps and timing deltas

This is aligned with your offline_braking_analysis.py and results_analysis_exp1.py scripts.

7.2 Scenario CSV

ScenarioLogger writes one row per braking episode, including: scenario tag, trigger kind (stop sign, TL, obstacle), μ, initial and minimum gaps (sim vs ground truth), whether ego stopped, time to stop, collision flag, range/TTS/TTC margins, reaction time, maximum slip, mean ABS factor, and a false-stop flag.

8. Extending the Perception Stack (MobileNetSSD / MobileNetV4‑SSDLite)

The code is structured so you can swap in new detectors:

Add a new detector class to detectors.py:

Implement a class MobileNetSSDDetector(BaseDetector) (or MobileNetV4SSDLiteDetector) with:

class MobileNetSSDDetector(BaseDetector):
    def __init__(...):
        ...
        self.labels = { ... }  # COCO-style mapping
        self.conf_thr = ...
        self.enabled = True
    def predict_raw(self, bgr: np.ndarray):
        # return classIds, confs, boxes in YOLO-like format
        return classIds, confs, boxes


Keep the same output format as YOLODetector.predict_raw so PerceptionPipeline stays unchanged.

Wire it in dynamic_brake_state.py:

Extend the --detector CLI choices in cli_parser.py from ['yolo'] to ['yolo', 'mobilenetssd', 'mobilenetv4ssdlite'].

In the App.__init__ method, select the appropriate detector based on args.detector.

Joint SSDLite + Depth model (planned)

For your planned joint model (MobileNetV4‑SSDLite backbone with supervised depth head):

Expose it as a detector that outputs both 2D boxes and per-pixel depth (or per-box depth predictions).

You can:

Either feed its depth map into the existing depth pipeline (via depth_m in PerceptionPipeline.run), or

Add a new range-est mode (e.g., --range-est learned) that uses the learned depth exclusively.

Reserve a section in the thesis/README (e.g., 6.5 Learned Monocular Depth Experiment) where you will later insert quantitative results.

9. Reproducibility Checklist (for Thesis)

To ensure your thesis experiments are reproducible:

Record configuration:

CLI command line (copy into thesis appendix).

Calibration JSON versions (calibration-file, safety-calibration-file, bus-calibration-file) including SHA-256 checksum (these are added in calibrations._load_json_like).

Fix random seeds:

Use --npc-seed to fix NPC spawn layouts.

Log telemetry & scenarios:

Keep raw CSVs under results/expX_*.

Use a consistent naming convention: exp1a_dry_yolo_depth, exp1b_dry_yolo_stereo, etc.

Document CARLA version & map:

E.g., CARLA 0.9.x, Town10HD_Opt, ClearNoon weather.

Include screenshots/plots:

TTC over time, τ_dyn & D_safety_dyn, braking profiles, min gaps.

Qualitative frames of perception overlays (HUD capture).

10. Known Limitations & Future Work

Detector choice: currently, only YOLO is fully wired through CLI; MobileNetSSD / MobileNetV4‑SSDLite integration is planned.

Monocular depth: current “pinhole” range relies on fixed canonical object heights; learned depth (joint SSDLite+depth model) will improve this.

Scenario coverage: experiments currently cover stop-sign and lead-vehicle / TL scenarios in one town; expanding to more towns and diverse scenes would increase robustness.

ECU realism: the MessageBus and SafetyManager approximate automotive ECUs but do not model full AUTOSAR or full ISO 26262 fault-handling behaviour.

11. How This Maps to Thesis Chapters

A suggested mapping to your thesis structure:

Chapter 2a (State of the Art) – AEB, ABS, perception literature.

Chapter 2b (State of the Techniques) – YOLO, MobileNetSSD/SSDLite, stereo, monocular depth.

Chapter 3 (Methodology) – High-level architecture (ECUs, bus, safety), range-estimation strategies, experimental design.

Chapter 4 (Implementation) – This codebase: file structure, class design, algorithms (AEB planner, ABS, safety manager).

Chapter 5 (Results) – Quantitative comparison (YOLO+depth vs YOLO+stereo vs YOLO+pinhole vs MobileNet-based pipelines).

Chapter 6 (Discussion) – Safety envelope interpretations, trade-offs between depth modalities, limitations, future work (joint SSDLite+depth).