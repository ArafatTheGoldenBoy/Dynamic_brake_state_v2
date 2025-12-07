#!/usr/bin/env python3
"""
Simple offline analysis for Experiment 1 (and similar experiments).

- Consumes the ScenarioLogger CSV (episode-level summary).
- Optionally consumes the TelemetryLogger CSV (frame-level AEB signals).
- Prints metrics that can be pasted into the thesis tables.

Logs are produced by dynamic_brake_state.App via TelemetryLogger and ScenarioLogger.
"""

import argparse
import csv
import math
from typing import Dict, List, Optional, Tuple


def _parse_float(x: str) -> Optional[float]:
    x = (x or "").strip()
    if not x:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _parse_bool01(x: str) -> Optional[bool]:
    x = (x or "").strip()
    if not x:
        return None
    if x in ("1", "True", "true", "YES", "yes"):
        return True
    if x in ("0", "False", "false", "NO", "no"):
        return False
    return None


def _percentile(values: List[float], q: float) -> float:
    """
    Simple percentile implementation (linear interpolation between points).
    q in [0, 100].
    """
    if not values:
        return math.nan
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    q = max(0.0, min(100.0, float(q)))
    pos = (len(vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


# ---------------------------------------------------------------------------
# Scenario CSV analysis (episode-level)
# ---------------------------------------------------------------------------

def load_scenarios(
    path: str,
    scenario_filter: Optional[str] = None,
) -> List[Dict[str, object]]:
    """
    Load ScenarioLogger CSV into a list of dicts.

    ScenarioLogger schema (columns) is defined in telemetry.py: _ScenarioLogger.HEADER
    and includes: scenario, trigger_kind, mu, v_init_mps, s_init_m, s_min_m,
    t_to_stop_s, collision, false_stop, etc. :contentReference[oaicite:1]{index=1}
    """
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if scenario_filter and raw.get("scenario") != scenario_filter:
                continue
            rec: Dict[str, object] = {}
            rec["scenario"] = raw.get("scenario", "")
            rec["trigger_kind"] = raw.get("trigger_kind", "")
            rec["mu"] = _parse_float(raw.get("mu", ""))
            rec["v_init_mps"] = _parse_float(raw.get("v_init_mps", ""))
            rec["s_init_m"] = _parse_float(raw.get("s_init_m", ""))
            rec["s_min_m"] = _parse_float(raw.get("s_min_m", ""))
            rec["s_init_gt_m"] = _parse_float(raw.get("s_init_gt_m", ""))
            rec["s_min_gt_m"] = _parse_float(raw.get("s_min_gt_m", ""))
            rec["stopped"] = _parse_bool01(raw.get("stopped", ""))
            rec["t_to_stop_s"] = _parse_float(raw.get("t_to_stop_s", ""))
            rec["collision"] = _parse_bool01(raw.get("collision", ""))
            rec["range_margin_m"] = _parse_float(raw.get("range_margin_m", ""))
            rec["tts_margin_s"] = _parse_float(raw.get("tts_margin_s", ""))
            rec["ttc_init_s"] = _parse_float(raw.get("ttc_init_s", ""))
            rec["ttc_min_s"] = _parse_float(raw.get("ttc_min_s", ""))
            rec["reaction_time_s"] = _parse_float(raw.get("reaction_time_s", ""))
            rec["max_lambda"] = _parse_float(raw.get("max_lambda", ""))
            rec["mean_abs_factor"] = _parse_float(raw.get("mean_abs_factor", ""))
            rec["false_stop"] = _parse_bool01(raw.get("false_stop", ""))
            rows.append(rec)
    return rows


def summarize_scenarios(records: List[Dict[str, object]]) -> Dict[str, float]:
    """
    Compute episode-level summary metrics for a given set of rows.
    """
    if not records:
        return {}

    n = len(records)
    collisions = sum(1 for r in records if r.get("collision") is True)
    false_stops = sum(1 for r in records if r.get("false_stop") is True)

    s_min_vals = [
        float(r["s_min_m"]) for r in records
        if r.get("s_min_m") is not None
    ]
    t_stop_vals = [
        float(r["t_to_stop_s"]) for r in records
        if r.get("t_to_stop_s") is not None and r.get("stopped") is True
    ]
    ttc_min_vals = [
        float(r["ttc_min_s"]) for r in records
        if r.get("ttc_min_s") is not None
    ]

    out: Dict[str, float] = {}
    out["episodes"] = float(n)
    out["collisions"] = float(collisions)
    out["collision_rate"] = collisions / float(n)
    out["false_stops"] = float(false_stops)
    out["false_stop_rate"] = false_stops / float(n)

    if s_min_vals:
        out["s_min_mean_m"] = sum(s_min_vals) / len(s_min_vals)
        out["s_min_p95_m"] = _percentile(s_min_vals, 95.0)
    if t_stop_vals:
        out["t_stop_mean_s"] = sum(t_stop_vals) / len(t_stop_vals)
        out["t_stop_p95_s"] = _percentile(t_stop_vals, 95.0)
    if ttc_min_vals:
        out["ttc_min_mean_s"] = sum(ttc_min_vals) / len(ttc_min_vals)
        out["ttc_min_p05_s"] = _percentile(ttc_min_vals, 5.0)

    return out


def print_scenario_summary(
    stats: Dict[str, float],
    label: str,
) -> None:
    """
    Pretty-print stats in a thesis-friendly way.
    """
    if not stats:
        print(f"[{label}] No records.")
        return

    print(f"\n=== Scenario summary: {label} ===")
    print(f"Episodes               : {int(stats['episodes'])}")
    print(f"Collisions             : {int(stats['collisions'])} "
          f"({stats['collision_rate'] * 100.0:.1f} %)")
    print(f"False stops            : {int(stats['false_stops'])} "
          f"({stats['false_stop_rate'] * 100.0:.1f} %)")

    if "s_min_mean_m" in stats:
        print(f"Min gap  mean / p95    : "
              f"{stats['s_min_mean_m']:.2f} m  /  {stats['s_min_p95_m']:.2f} m")
    if "t_stop_mean_s" in stats:
        print(f"t_to_stop mean / p95   : "
              f"{stats['t_stop_mean_s']:.2f} s  /  {stats['t_stop_p95_s']:.2f} s")
    if "ttc_min_mean_s" in stats:
        print(f"TTC_min mean / p05     : "
              f"{stats['ttc_min_mean_s']:.2f} s  /  {stats['ttc_min_p05_s']:.2f} s")


# ---------------------------------------------------------------------------
# Telemetry CSV analysis (frame-level)
# ---------------------------------------------------------------------------

def load_telemetry(path: str) -> List[Dict[str, object]]:
    """
    Load TelemetryLogger CSV into a list of dicts.

    TelemetryLogger schema is defined in telemetry._TelemetryLogger and includes
    columns such as t, v_mps, tau_dyn, D_safety_dyn, sigma_depth, a_des, brake,
    ttc_s, gate_hit, gate_confirmed, false_stop_flag, brake_stage, etc. :contentReference[oaicite:2]{index=2}
    """
    rows: List[Dict[str, object]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rec: Dict[str, object] = {}
            rec["t"] = _parse_float(raw.get("t", ""))
            rec["v_mps"] = _parse_float(raw.get("v_mps", ""))
            rec["tau_dyn"] = _parse_float(raw.get("tau_dyn", ""))
            rec["D_safety_dyn"] = _parse_float(raw.get("D_safety_dyn", ""))
            rec["sigma_depth"] = _parse_float(raw.get("sigma_depth", ""))
            rec["a_des"] = _parse_float(raw.get("a_des", ""))
            rec["brake"] = _parse_float(raw.get("brake", ""))
            rec["ttc_s"] = _parse_float(raw.get("ttc_s", ""))
            rec["gate_hit"] = _parse_bool01(raw.get("gate_hit", ""))
            rec["gate_confirmed"] = _parse_bool01(raw.get("gate_confirmed", ""))
            rec["false_stop_flag"] = _parse_bool01(raw.get("false_stop_flag", ""))
            rec["brake_stage"] = _parse_float(raw.get("brake_stage", ""))
            rows.append(rec)
    return rows


def summarize_telemetry(records: List[Dict[str, object]]) -> Dict[str, float]:
    """
    Compute simple global stats from telemetry (no per-episode segmentation).
    """
    if not records:
        return {}

    # Focus on frames where the brake is actually applied
    brake_frames = [
        r for r in records
        if r.get("brake") is not None and float(r["brake"]) > 0.05
    ]

    stats: Dict[str, float] = {}
    stats["frames_total"] = float(len(records))
    stats["frames_braking"] = float(len(brake_frames))

    if not brake_frames:
        return stats

    def _collect(name: str) -> List[float]:
        vals: List[float] = []
        for r in brake_frames:
            v = r.get(name)
            if v is not None:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
        return vals

    ttc_vals = _collect("ttc_s")
    tau_vals = _collect("tau_dyn")
    dsafe_vals = _collect("D_safety_dyn")
    sigma_vals = _collect("sigma_depth")

    if ttc_vals:
        stats["ttc_s_min"] = min(ttc_vals)
        stats["ttc_s_mean"] = sum(ttc_vals) / len(ttc_vals)
        stats["ttc_s_p05"] = _percentile(ttc_vals, 5.0)
    if tau_vals:
        stats["tau_dyn_mean"] = sum(tau_vals) / len(tau_vals)
    if dsafe_vals:
        stats["D_safety_dyn_mean"] = sum(dsafe_vals) / len(dsafe_vals)
    if sigma_vals:
        stats["sigma_depth_mean"] = sum(sigma_vals) / len(sigma_vals)

    return stats


def print_telemetry_summary(stats: Dict[str, float], label: str) -> None:
    if not stats:
        print(f"[{label}] No telemetry records.")
        return

    print(f"\n=== Telemetry summary: {label} ===")
    print(f"Frames total           : {int(stats['frames_total'])}")
    print(f"Frames with brake>0.05 : {int(stats['frames_braking'])}")
    if "ttc_s_min" in stats:
        print(f"TTC during braking     : "
              f"min={stats['ttc_s_min']:.2f} s, "
              f"mean={stats['ttc_s_mean']:.2f} s, "
              f"p05={stats['ttc_s_p05']:.2f} s")
    if "tau_dyn_mean" in stats:
        print(f"tau_dyn (mean)         : {stats['tau_dyn_mean']:.2f} s")
    if "D_safety_dyn_mean" in stats:
        print(f"D_safety_dyn (mean)    : {stats['D_safety_dyn_mean']:.2f} m")
    if "sigma_depth_mean" in stats:
        print(f"sigma_depth (mean)     : {stats['sigma_depth_mean']:.2f} m")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline analysis for AEB experiments (Scenario + Telemetry CSVs)."
    )
    parser.add_argument(
        "--scenario-csv",
        type=str,
        required=True,
        help="Path to ScenarioLogger CSV (e.g., exp1_dry_yolo_depth_scenarios.csv)",
    )
    parser.add_argument(
        "--telemetry-csv",
        type=str,
        default=None,
        help="Optional path to TelemetryLogger CSV (e.g., exp1_dry_yolo_depth_telemetry.csv)",
    )
    parser.add_argument(
        "--scenario-tag",
        type=str,
        default=None,
        help="Optional scenario tag filter (matches 'scenario' column)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="Experiment",
        help="Label to use in printed summaries (e.g., 'Exp1-A (YOLO+depth)')",
    )

    args = parser.parse_args()

    # Scenario (episode-level) summary
    scenario_rows = load_scenarios(args.scenario_csv, args.scenario_tag)
    if not scenario_rows:
        print("[WARN] No scenario rows loaded (check path or --scenario-tag).")
    else:
        scen_stats = summarize_scenarios(scenario_rows)
        print_scenario_summary(scen_stats, args.label)

    # Optional telemetry (frame-level) summary
    if args.telemetry_csv:
        tele_rows = load_telemetry(args.telemetry_csv)
        tele_stats = summarize_telemetry(tele_rows)
        print_telemetry_summary(tele_stats, args.label)


if __name__ == "__main__":
    main()
