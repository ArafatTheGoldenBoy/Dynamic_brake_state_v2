import glob
import os
import sys


def import_carla():
    """Import CARLA after probing common egg locations."""
    try:
        import carla  # type: ignore
        return carla
    except Exception:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        patterns = [
            os.path.join(this_dir, '..', 'carla', 'dist', 'carla-*.egg'),
            os.path.join(this_dir, '..', '..', 'PythonAPI', 'carla', 'dist', 'carla-*.egg'),
            os.path.join(this_dir, '..', 'dist', 'carla-*.egg'),
        ]
        for p in patterns:
            candidates.extend(glob.glob(p))
        if candidates:
            sys.path.append(os.path.normpath(candidates[0]))
        import carla  # type: ignore
        return carla
