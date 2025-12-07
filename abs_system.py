"""ABS and slip control components for dynamic brake state simulation."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

FRICTION_CONFIGS = {
    'high': {'lambda_star': 0.18, 'kp': 5.0, 'ki': 25.0},
    'medium': {'lambda_star': 0.15, 'kp': 4.0, 'ki': 20.0},
    'low': {'lambda_star': 0.10, 'kp': 3.0, 'ki': 12.0},
}


def compute_slip_per_wheel(v_ego: float, wheel_speeds: List[float]) -> List[float]:
    """Return longitudinal slip λ for each wheel in [0, 1]."""

    v = max(0.1, float(v_ego))
    slips: List[float] = []
    for v_w in wheel_speeds:
        try:
            v_w = max(0.0, float(v_w))
        except Exception:
            v_w = 0.0
        lam = (v - v_w) / v
        lam = max(0.0, min(1.0, lam))
        slips.append(lam)
    return slips


class PISlipChannel:
    def __init__(self, dt: float,
                 lambda_star: float = 0.15,
                 kp: float = 4.0,
                 ki: float = 20.0):
        self.dt = dt
        self.lambda_star = lambda_star
        self.kp = kp
        self.ki = ki
        self.I = 0.0
        self.f = 1.0

    def reset(self):
        self.I = 0.0
        self.f = 1.0

    def step(self, lam: float) -> float:
        e = float(self.lambda_star) - float(max(0.0, min(1.0, lam)))
        u_raw = self.kp * e + self.I
        f_unsat = u_raw
        f_sat = max(0.0, min(1.0, f_unsat))
        if (f_unsat > 1.0 and e > 0.0) or (f_unsat < 0.0 and e < 0.0):
            pass
        else:
            self.I += self.ki * e * self.dt
        self.f = f_sat
        return self.f


class FrictionEstimator:
    def __init__(self, g: float = 9.81, alpha: float = 0.05, mu_init: float = 0.8):
        self.g = g
        self.alpha = alpha
        self.mu_est = mu_init

    def reset(self):
        self.mu_est = 0.8

    def update(self, v_ego: float, lambda_max: float, a_long: float, brake_req: float):
        if v_ego < 5.0:
            return
        if brake_req < 0.3:
            return
        if lambda_max < 0.10 or lambda_max > 0.25:
            return
        if not math.isfinite(a_long):
            return
        mu_inst = abs(a_long) / self.g
        mu_inst = max(0.01, min(1.5, mu_inst))
        self.mu_est = (1.0 - self.alpha) * self.mu_est + self.alpha * mu_inst

    def regime(self) -> str:
        mu = self.mu_est
        if mu > 0.8:
            return 'high'
        if mu > 0.4:
            return 'medium'
        return 'low'


class PISlipABSActuator:
    def __init__(self, dt: float,
                 lambda_star: float = 0.15,
                 kp: float = 4.0,
                 ki: float = 20.0,
                 v_min_abs: float = 3.0,
                 wheel_count: int = 4):
        self.dt = dt
        self.v_min_abs = v_min_abs
        self.channels = [PISlipChannel(dt, lambda_star, kp, ki) for _ in range(wheel_count)]
        self.last_lambda_max = 0.0
        self.last_f_global = 1.0

    def reset(self):
        for ch in self.channels:
            ch.reset()
        self.last_lambda_max = 0.0
        self.last_f_global = 1.0

    def _eligible(self, v_ego: float, wheel_speeds: List[float]) -> bool:
        return v_ego >= self.v_min_abs and len(wheel_speeds) == len(self.channels)

    def step(self, brake_req: float, v_ego: float, wheel_speeds: List[float], a_long: float = 0.0) -> float:
        del a_long  # unused in fixed-mode actuator
        brake_req = max(0.0, min(1.0, float(brake_req)))
        if not self._eligible(v_ego, wheel_speeds):
            self.last_lambda_max = 0.0
            self.last_f_global = 1.0
            for ch in self.channels:
                ch.reset()
            return brake_req
        slips = compute_slip_per_wheel(v_ego, wheel_speeds)
        self.last_lambda_max = max(slips) if slips else 0.0
        f_list = [ch.step(lam) for ch, lam in zip(self.channels, slips)]
        f_global = min(f_list) if f_list else 1.0
        self.last_f_global = f_global
        return brake_req * f_global

    def debug_metrics(self) -> Dict[str, Any]:
        return {
            'lambda_max': self.last_lambda_max,
            'f_global': self.last_f_global,
            'mu_est': None,
            'regime': 'fixed',
        }


class AdaptivePISlipABSActuator(PISlipABSActuator):
    def __init__(self, dt: float, v_min_abs: float = 3.0,
                 wheel_count: int = 4,
                 friction_configs: Optional[Dict[str, Dict[str, float]]] = None):
        super().__init__(dt, v_min_abs=v_min_abs, wheel_count=wheel_count)
        self.friction = FrictionEstimator()
        self.friction_configs = friction_configs if friction_configs is not None else FRICTION_CONFIGS
        self.current_regime = 'medium'
        self._apply_friction_config(self.current_regime)

    def _apply_friction_config(self, regime: str):
        cfg = self.friction_configs.get(regime, self.friction_configs['medium'])
        for ch in self.channels:
            ch.lambda_star = cfg['lambda_star']
            ch.kp = cfg['kp']
            ch.ki = cfg['ki']
            ch.reset()
        self.current_regime = regime

    def reset(self):
        super().reset()
        self.friction.reset()
        self._apply_friction_config('medium')

    def step(self, brake_req: float, v_ego: float, wheel_speeds: List[float], a_long: float = 0.0) -> float:
        brake_req = max(0.0, min(1.0, float(brake_req)))
        if not self._eligible(v_ego, wheel_speeds):
            self.last_lambda_max = 0.0
            self.last_f_global = 1.0
            for ch in self.channels:
                ch.reset()
            return brake_req
        slips = compute_slip_per_wheel(v_ego, wheel_speeds)
        lambda_max = max(slips) if slips else 0.0
        self.last_lambda_max = lambda_max
        self.friction.update(v_ego, lambda_max, a_long, brake_req)
        regime = self.friction.regime()
        if regime != self.current_regime:
            self._apply_friction_config(regime)
        f_list = [ch.step(lam) for ch, lam in zip(self.channels, slips)]
        f_global = min(f_list) if f_list else 1.0
        self.last_f_global = f_global
        return brake_req * f_global

    def debug_metrics(self) -> Dict[str, Any]:
        return {
            'lambda_max': self.last_lambda_max,
            'f_global': self.last_f_global,
            'mu_est': self.friction.mu_est,
            'regime': self.current_regime,
        }

