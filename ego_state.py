from dataclasses import dataclass


@dataclass
class EgoState:
    x: float
    y: float
    z: float
    yaw_deg: float
    v_mps: float
    t: float
