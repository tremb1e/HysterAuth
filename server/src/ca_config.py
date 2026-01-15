from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal, Optional

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib


@dataclass(frozen=True)
class ProcessingConfig:
    min_total_mb: int = 100
    target_total_mb: int = 100
    workers: int = 5
    # HMOG attacker-user selection:
    # - val uses the 1st HMOG user (sorted by directory name)
    # - test uses the 2nd HMOG user (sorted by directory name)
    hmog_val_subject_count: int = 1
    hmog_test_subject_count: int = 1
    # HMOG attacker data volume caps (avoid huge val/test producing too many windows).
    # - max_rows_per_subject: max rows read per HMOG user (sum of train/val/test CSVs).
    # - max_rows_total: max rows per merged split (val or test) across multiple HMOG users.
    hmog_max_rows_per_subject: int = 200_000
    hmog_max_rows_total: int = 1_000_000


@dataclass(frozen=True)
class WindowConfig:
    sizes: List[float] = None  # type: ignore[assignment]
    overlap: float = 0.5
    sampling_rate_hz: int = 100

    def __post_init__(self) -> None:
        if self.sizes is None:
            object.__setattr__(self, "sizes", [0.2])


@dataclass(frozen=True)
class AuthConfig:
    max_decision_time_sec: float = 2.0
    k_rejects_mode: Literal["by_window"] = "by_window"
    # Vote rule: in the most recent N windows, if reject>=M, trigger an interrupt.
    vote_window_size: int = 7
    vote_min_rejects: int = 6
    # Upper bound of tolerated false interrupts at window level (0~1).
    target_window_frr: float = 0.10


@dataclass(frozen=True)
class CAConfig:
    processing: ProcessingConfig = ProcessingConfig()
    windows: WindowConfig = WindowConfig()
    auth: AuthConfig = AuthConfig()

    def k_rejects_for_window(self, window_size_sec: float) -> int:
        """Compute consecutive rejects K for a given window size.

        Policy: convert decision time T into K using stride = window_size * (1 - overlap),
        so that the interrupt is not earlier than T seconds.
        """
        window_size_sec = float(window_size_sec)
        if window_size_sec <= 0.0:
            raise ValueError(f"window_size_sec must be > 0, got {window_size_sec}")
        max_t = float(self.auth.max_decision_time_sec)
        if max_t <= 0.0:
            return 0
        stride_sec = window_size_sec * (1.0 - float(self.windows.overlap))
        if stride_sec <= 0.0:
            raise ValueError(f"Invalid stride_sec={stride_sec} from window_size={window_size_sec}, overlap={self.windows.overlap}")
        return max(1, int(math.ceil(max_t / stride_sec)))


def _default_config_path() -> Path:
    # server/src/ca_config.py -> server/ca_config.toml
    return Path(__file__).resolve().parents[1] / "ca_config.toml"


def load_ca_config(path: Optional[Path] = None) -> CAConfig:
    path = Path(path) if path is not None else _default_config_path()
    if not path.exists():
        return CAConfig()

    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    proc_raw = raw.get("processing", {}) or {}
    win_raw = raw.get("windows", {}) or {}
    auth_raw = raw.get("auth", {}) or {}

    processing = ProcessingConfig(
        min_total_mb=int(proc_raw.get("min_total_mb", ProcessingConfig.min_total_mb)),
        target_total_mb=int(proc_raw.get("target_total_mb", ProcessingConfig.target_total_mb)),
        workers=int(proc_raw.get("workers", ProcessingConfig.workers)),
        hmog_val_subject_count=int(proc_raw.get("hmog_val_subject_count", ProcessingConfig.hmog_val_subject_count)),
        hmog_test_subject_count=int(proc_raw.get("hmog_test_subject_count", ProcessingConfig.hmog_test_subject_count)),
        hmog_max_rows_per_subject=int(proc_raw.get("hmog_max_rows_per_subject", ProcessingConfig.hmog_max_rows_per_subject)),
        hmog_max_rows_total=int(proc_raw.get("hmog_max_rows_total", ProcessingConfig.hmog_max_rows_total)),
    )

    sizes = win_raw.get("sizes", None)
    if sizes is not None:
        sizes = [float(x) for x in sizes]
        sizes = sorted({round(float(x), 3) for x in sizes})
    windows = WindowConfig(
        sizes=sizes,  # type: ignore[arg-type]
        overlap=float(win_raw.get("overlap", WindowConfig.overlap)),
        sampling_rate_hz=int(win_raw.get("sampling_rate_hz", WindowConfig.sampling_rate_hz)),
    )

    target_window_frr_raw = auth_raw.get("target_window_frr", None)
    if target_window_frr_raw is None:
        # Backward-compatible fallback
        target_window_frr_raw = auth_raw.get("target_session_frr", AuthConfig.target_window_frr)

    auth = AuthConfig(
        max_decision_time_sec=float(auth_raw.get("max_decision_time_sec", AuthConfig.max_decision_time_sec)),
        k_rejects_mode=str(auth_raw.get("k_rejects_mode", AuthConfig.k_rejects_mode)),  # type: ignore[arg-type]
        vote_window_size=int(auth_raw.get("vote_window_size", AuthConfig.vote_window_size)),
        vote_min_rejects=int(auth_raw.get("vote_min_rejects", AuthConfig.vote_min_rejects)),
        target_window_frr=float(target_window_frr_raw),
    )

    return CAConfig(processing=processing, windows=windows, auth=auth)


_CACHED: Optional[CAConfig] = None


def get_ca_config(path: Optional[Path] = None) -> CAConfig:
    global _CACHED
    if _CACHED is None:
        _CACHED = load_ca_config(path)
    return _CACHED
