import numpy as np


def ms_to_pts(t_ms: int, fs: int) -> int:
    """Convert time in ms to time in pts."""
    return int(fs * t_ms / 1000)


def pts_to_ms(t_pts: int, fs: int) -> int:
    """Convert time in pts to time in ms (to nearest whole)"""
    return int(np.round(t_pts * 1000 / fs))