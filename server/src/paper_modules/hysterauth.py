from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class HysterAuthConfig:
    """Hysteresis-based false rejection mitigation (HysterAuth) configuration."""

    alpha: float = 1.0
    beta: float = 0.0
    epsilon: float = 1e-6
    forgetting_factor: float = 0.95  # lambda in the paper
    k_max: float = 20.0
    theta_enter: float = 3.0
    theta_exit: float = 1.0

    def __post_init__(self) -> None:
        if not (0.0 < float(self.forgetting_factor) < 1.0):
            raise ValueError("forgetting_factor must be in (0, 1)")
        if float(self.theta_enter) <= float(self.theta_exit):
            raise ValueError("theta_enter must be > theta_exit")
        if not (0.0 < float(self.epsilon) <= 1e-3):
            raise ValueError("epsilon must be in (0, 1e-3]")


State = Literal["Normal", "Abnormal"]


@dataclass(frozen=True)
class HysterAuthDecision:
    raw_score: float
    anomaly_prob: float
    log_odds: float
    k_evidence: float
    state: State
    switched: bool
    accept: bool
    interrupt: bool


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class HysterAuth:
    """Adaptive decision module with hysteresis, as described in the paper."""

    def __init__(self, cfg: HysterAuthConfig) -> None:
        self.cfg = cfg
        self.reset()

    def reset(self) -> None:
        self._k = 0.0
        self._state: State = "Normal"

    @property
    def state(self) -> State:
        return self._state

    @property
    def k_evidence(self) -> float:
        return float(self._k)

    def update(self, anomaly_score: float) -> HysterAuthDecision:
        # Eq.(platt)
        p_t = _sigmoid(float(self.cfg.alpha) * float(anomaly_score) + float(self.cfg.beta))
        eps = float(self.cfg.epsilon)
        p_t = max(eps, min(1.0 - eps, float(p_t)))

        # Eq.(logodds)
        log_odds = math.log(p_t / (1.0 - p_t))

        # Eq.(accumulate) + Eq.(sat)
        self._k = float(self.cfg.forgetting_factor) * float(self._k) + float(log_odds)
        k_max = float(self.cfg.k_max)
        if k_max > 0:
            self._k = max(-k_max, min(k_max, float(self._k)))

        prev_state = self._state
        if prev_state == "Normal" and self._k >= float(self.cfg.theta_enter):
            self._state = "Abnormal"
        elif prev_state == "Abnormal" and self._k <= float(self.cfg.theta_exit):
            self._state = "Normal"

        switched = self._state != prev_state
        accept = self._state == "Normal"
        interrupt = switched and (self._state == "Abnormal")
        return HysterAuthDecision(
            raw_score=float(anomaly_score),
            anomaly_prob=float(p_t),
            log_odds=float(log_odds),
            k_evidence=float(self._k),
            state=self._state,
            switched=bool(switched),
            accept=bool(accept),
            interrupt=bool(interrupt),
        )


def _demo() -> None:
    cfg = HysterAuthConfig(
        alpha=1.0,
        beta=0.0,
        epsilon=1e-6,
        forgetting_factor=0.9,
        k_max=20.0,
        theta_enter=2.5,
        theta_exit=1.0,
    )
    h = HysterAuth(cfg)

    # Simulated anomaly scores: stable normal, then sustained anomaly, then recovery.
    scores = [-2.0] * 5 + [0.5, 1.0, 1.5, 2.0, 2.0, 2.0] + [0.0, -0.5, -1.0, -1.5, -2.0]

    print("== HysterAuth Demo ==")
    for i, s in enumerate(scores):
        d = h.update(s)
        mark = "SWITCH" if d.switched else ""
        print(
            f"t={i:02d} score={d.raw_score:+.2f} p={d.anomaly_prob:.3f} "
            f"logodds={d.log_odds:+.3f} K={d.k_evidence:+.3f} state={d.state} {mark}"
        )


if __name__ == "__main__":
    _demo()

