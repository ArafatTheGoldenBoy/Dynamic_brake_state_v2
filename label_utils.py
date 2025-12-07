from typing import Dict, Optional


def _norm_label(s: str) -> str:
    return ''.join(ch for ch in s.lower() if ch.isalpha())


TRIGGER_NAMES_NORM = {
    'trafficlight', 'stopsign', 'person', 'car', 'bus', 'truck', 'motorcycle', 'motorbike', 'bicycle', 'train'
}


def _parse_float_map(spec: Optional[str]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not spec:
        return mapping
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        k = _norm_label(k.strip())
        try:
            mapping[k] = float(v.strip())
        except Exception:
            pass
    return mapping


def _parse_int_map(spec: Optional[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not spec:
        return mapping
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        k = _norm_label(k.strip())
        try:
            mapping[k] = int(float(v.strip()))
        except Exception:
            pass
    return mapping


def parse_per_class_conf_map(spec: Optional[str]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not spec:
        return mapping
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        k = _norm_label(k.strip())
        try:
            mapping[k] = float(v.strip())
        except Exception:
            pass
    return mapping


def parse_per_class_iou_map(spec: Optional[str]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not spec:
        return mapping
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        k = _norm_label(k.strip())
        try:
            mapping[k] = float(v.strip())
        except Exception:
            pass
    return mapping


def parse_engage_override_map(spec: Optional[str]) -> Dict[str, float]:
    mapping: Dict[str, float] = {}
    if not spec:
        return mapping
    for tok in str(spec).split(','):
        tok = tok.strip()
        if not tok or ':' not in tok:
            continue
        k, v = tok.split(':', 1)
        k = _norm_label(k.strip())
        try:
            mapping[k] = float(v.strip())
        except Exception:
            pass
    return mapping


def parse_min_h_override_map(spec: Optional[str]) -> Dict[str, int]:
    return _parse_int_map(spec)


def parse_gate_frac_override_map(spec: Optional[str]) -> Dict[str, float]:
    return _parse_float_map(spec)


def parse_gate_lateral_override_map(spec: Optional[str]) -> Dict[str, float]:
    return _parse_float_map(spec)


__all__ = [
    "_norm_label",
    "TRIGGER_NAMES_NORM",
    "parse_per_class_conf_map",
    "parse_per_class_iou_map",
    "parse_engage_override_map",
    "parse_min_h_override_map",
    "parse_gate_frac_override_map",
    "parse_gate_lateral_override_map",
]
