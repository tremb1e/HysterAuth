from __future__ import annotations

from typing import List, Tuple

import numpy as np

AXIS_COLUMNS = (
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "mag_x",
    "mag_y",
    "mag_z",
)


def _resample_time_axis(window: np.ndarray, target_width: int) -> np.ndarray:
    """Linear-resample a (H, T) window to (H, target_width) along the time axis."""
    if window.shape[1] == target_width:
        return window
    x_old = np.linspace(0.0, 1.0, window.shape[1], dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, target_width, dtype=np.float32)
    out = np.empty((window.shape[0], target_width), dtype=np.float32)
    for i in range(window.shape[0]):
        out[i] = np.interp(x_new, x_old, window[i]).astype(np.float32, copy=False)
    return out


def windowize_dataframe(
    df,
    *,
    window_size_sec: float,
    overlap: float,
    sampling_rate_hz: int,
    target_width: int,
) -> Tuple[List[int], np.ndarray]:
    """Convert a normalized dataframe into (window_ids, windows).

    Returns:
        window_ids: List[int]
        windows: np.ndarray shaped (N, 1, 12, target_width)
            9 raw axes + 3 magnitudes (acc/gyr/mag).
    """
    if df.empty:
        return [], np.empty((0, 1, 12, target_width), dtype=np.float32)

    window_points = max(1, int(round(float(window_size_sec) * int(sampling_rate_hz))))
    step_points = max(1, int(round(window_points * (1.0 - float(overlap)))))
    values = df[list(AXIS_COLUMNS)].to_numpy(dtype=np.float32, copy=False)

    window_ids: List[int] = []
    windows: List[np.ndarray] = []
    window_id = 0
    for start in range(0, len(values) - window_points + 1, step_points):
        window_slice = values[start : start + window_points]
        acc = window_slice[:, 0:3]
        gyr = window_slice[:, 3:6]
        mag = window_slice[:, 6:9]
        acc_mag = np.linalg.norm(acc, axis=1)
        gyr_mag = np.linalg.norm(gyr, axis=1)
        mag_mag = np.linalg.norm(mag, axis=1)
        window_raw = np.vstack(
            [
                acc[:, 0],
                acc[:, 1],
                acc[:, 2],
                acc_mag,
                gyr[:, 0],
                gyr[:, 1],
                gyr[:, 2],
                gyr_mag,
                mag[:, 0],
                mag[:, 1],
                mag[:, 2],
                mag_mag,
            ]
        ).astype(np.float32, copy=False)
        if window_raw.shape[1] != int(target_width):
            window_raw = _resample_time_axis(window_raw, int(target_width))
        windows.append(window_raw[np.newaxis, :, :])
        window_ids.append(window_id)
        window_id += 1

    if not windows:
        return [], np.empty((0, 1, 12, target_width), dtype=np.float32)

    return window_ids, np.stack(windows, axis=0).astype(np.float32, copy=False)

