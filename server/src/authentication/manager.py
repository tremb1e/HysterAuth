from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from ..config import settings
from ..paper_modules.hysterauth import HysterAuth, HysterAuthConfig
from ..paper_modules.sequence_transformer import TokenTransformerLM, TokenTransformerLMConfig
from ..paper_modules.vector_quantization import (
    VectorQuantizer,
    VectorQuantizerConfig,
    serialize_tokens_to_base64,
)
from ..processing.pipeline import _extract_sensor_records, _resample_records, build_config
from ..processing.scaler import apply_scaler, load_scaler
from ..storage.inference_storage import InferenceStorage
from .runner import AuthRunConfig, load_best_policy
from .windowing import windowize_dataframe


@dataclass
class AuthResultPayload:
    user: str
    session_id: str
    window_id: int
    score: float
    threshold: float
    accept: bool
    interrupt: bool
    normalized_score: float
    k_rejects: int
    window_size: float
    model_version: str
    message: str


@dataclass
class AuthSessionState:
    user_id: str
    session_id: str
    policy: AuthRunConfig
    hysterauth: HysterAuth
    last_activity: float = field(default_factory=time.time)
    tail_records: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {"acc": [], "gyr": [], "mag": []})
    window_index: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


@dataclass(frozen=True)
class _TokenAuthModels:
    vq: VectorQuantizer
    transformer: TokenTransformerLM
    device: torch.device


class AuthSessionManager:
    """Session-oriented authentication manager.

    Server inference order (paper-aligned):
      1) Sensor window -> discrete serialized tokens via Vector Quantization (VQ)
      2) Token sequence -> log-likelihood score via an optimized causal Transformer
      3) Score -> robust decision via HysterAuth (hysteresis accumulation)
    """

    def __init__(
        self,
        *,
        max_cached_models: int = 4,
        session_ttl_sec: int = 600,
        max_concurrent_inference: Optional[int] = None,
    ) -> None:
        self._sessions: Dict[str, AuthSessionState] = {}
        self._processing_cfg = build_config()
        self._inference_storage = InferenceStorage(settings.inference_storage_path)

        self._max_cached_models = max(1, int(max_cached_models))
        self._model_cache: Dict[str, _TokenAuthModels] = {}
        self._model_order: List[str] = []

        self._session_ttl_sec = int(session_ttl_sec)
        if max_concurrent_inference is None:
            max_concurrent_inference = min(8, os.cpu_count() or 1)
        max_concurrent_inference = max(1, int(max_concurrent_inference))
        self._inference_semaphore = asyncio.Semaphore(max_concurrent_inference)

    def _cache_touch(self, key: str) -> None:
        if key in self._model_order:
            self._model_order.remove(key)
        self._model_order.append(key)

    def _cache_put(self, key: str, models: _TokenAuthModels) -> None:
        if key in self._model_cache:
            self._cache_touch(key)
            self._model_cache[key] = models
            return
        if len(self._model_order) >= self._max_cached_models and self._model_order:
            evict = self._model_order.pop(0)
            self._model_cache.pop(evict, None)
        self._model_cache[key] = models
        self._model_order.append(key)

    def _load_models(self, cfg: AuthRunConfig) -> _TokenAuthModels:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        vq_cfg = VectorQuantizerConfig(**cfg.vq_config)
        vq = VectorQuantizer(vq_cfg).to(device).eval()
        if cfg.vq_checkpoint and Path(cfg.vq_checkpoint).exists():
            vq.load_state_dict(torch.load(Path(cfg.vq_checkpoint), map_location=device))

        t_cfg = TokenTransformerLMConfig(**cfg.transformer_config)
        transformer = TokenTransformerLM(t_cfg).to(device).eval()
        if cfg.transformer_checkpoint and Path(cfg.transformer_checkpoint).exists():
            transformer.load_state_dict(torch.load(Path(cfg.transformer_checkpoint), map_location=device))

        return _TokenAuthModels(vq=vq, transformer=transformer, device=device)

    def _get_models(self, cfg: AuthRunConfig) -> _TokenAuthModels:
        key = f"{cfg.user}::{cfg.model_version}::{cfg.vq_checkpoint}::{cfg.transformer_checkpoint}"
        if key in self._model_cache:
            self._cache_touch(key)
            return self._model_cache[key]
        models = self._load_models(cfg)
        self._cache_put(key, models)
        return models

    def _prune_sessions(self) -> None:
        now = time.time()
        expired = [k for k, v in self._sessions.items() if (now - v.last_activity) > self._session_ttl_sec]
        for key in expired:
            self._sessions.pop(key, None)

    def has_trained_model(self, user_id: str) -> bool:
        try:
            _ = load_best_policy(user_id)
        except Exception:
            return False
        return True

    def start_session(self, user_id: str, session_id: str) -> Tuple[bool, str, Optional[AuthRunConfig]]:
        self._prune_sessions()
        try:
            cfg = load_best_policy(user_id)
        except Exception as exc:
            return False, f"model_not_ready: {exc}", None

        hysterauth = HysterAuth(HysterAuthConfig(**cfg.hysterauth_config))
        state = AuthSessionState(user_id=user_id, session_id=session_id, policy=cfg, hysterauth=hysterauth)
        self._sessions[user_id] = state
        return True, "ok", cfg

    def _tokenize_and_score(
        self, models: _TokenAuthModels, windows_ts: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            vq_out = models.vq(windows_ts)
            tokens = vq_out.codes.to(dtype=torch.long)
            scores = models.transformer.score(tokens)
        return tokens, scores

    async def handle_packet(
        self,
        *,
        user_id: str,
        session_id: str,
        parsed_batch: Dict[str, Any],
    ) -> Optional[AuthResultPayload]:
        self._prune_sessions()
        state = self._sessions.get(user_id)
        if state is None or state.session_id != session_id:
            return None

        state.last_activity = time.time()

        async with state.lock:
            await self._inference_storage.append_raw_packet(user_id, session_id, parsed_batch)

            packets = [{"sensor_batch": parsed_batch}]
            records = _extract_sensor_records(packets)
            combined_records = {k: (state.tail_records.get(k, []) + records.get(k, [])) for k in ("acc", "gyr", "mag")}

            df = _resample_records(combined_records, session_label=session_id, user_id=user_id, cfg=self._processing_cfg)
            if df is None or df.empty:
                self._trim_tail(state, combined_records)
                return None

            scaler_path = Path(settings.processed_data_path) / "z-score" / user_id / "scaler.json"
            if scaler_path.exists():
                scaler = load_scaler(scaler_path)
                normalized = apply_scaler(df, scaler)
            else:
                # Runnable fallback when no processed data/scaler exists yet.
                normalized = df

            window_ids, windows = windowize_dataframe(
                normalized,
                window_size_sec=state.policy.window_size,
                overlap=state.policy.overlap,
                sampling_rate_hz=self._processing_cfg.sampling_rate_hz,
                target_width=state.policy.target_width,
            )

            if windows.size == 0:
                self._trim_tail(state, combined_records)
                return None

            models = self._get_models(state.policy)
            device = models.device

            # windows: (B, 1, 12, T) -> (B, T, 12)
            windows_ts = torch.from_numpy(windows[:, 0].transpose(0, 2, 1)).to(
                device=device, dtype=torch.float32, non_blocking=True
            )

            async with self._inference_semaphore:
                tokens, scores = await asyncio.to_thread(self._tokenize_and_score, models, windows_ts)

            scores_cpu = scores.detach().cpu().numpy()

            final_payload: Optional[AuthResultPayload] = None
            for i, (offset, auth_score) in enumerate(zip(window_ids, scores_cpu)):
                window_id = state.window_index + int(offset)
                auth_score_f = float(auth_score)
                anomaly_score = -auth_score_f
                decision = state.hysterauth.update(anomaly_score)

                tokens_np = tokens[i].detach().cpu().numpy()
                tokens_b64 = serialize_tokens_to_base64(tokens_np.astype("int64", copy=False))

                message = (
                    f"state={decision.state} switched={int(decision.switched)} "
                    f"K={decision.k_evidence:+.3f} p_anom={decision.anomaly_prob:.3f}"
                )

                await self._inference_storage.append_result(
                    user_id,
                    session_id,
                    {
                        "window_id": window_id,
                        "auth_score": auth_score_f,
                        "anomaly_score": float(anomaly_score),
                        "anomaly_prob": float(decision.anomaly_prob),
                        "log_odds": float(decision.log_odds),
                        "k_evidence": float(decision.k_evidence),
                        "state": str(decision.state),
                        "switched": bool(decision.switched),
                        "accept": bool(decision.accept),
                        "interrupt": bool(decision.interrupt),
                        "tokens_b64": tokens_b64,
                        "window_size": float(state.policy.window_size),
                        "model_version": state.policy.model_version,
                    },
                )

                final_payload = AuthResultPayload(
                    user=user_id,
                    session_id=session_id,
                    window_id=window_id,
                    score=auth_score_f,
                    threshold=float(state.policy.hysterauth_config.get("theta_enter", 0.0)),
                    accept=bool(decision.accept),
                    interrupt=bool(decision.interrupt),
                    normalized_score=float(1.0 - decision.anomaly_prob),
                    k_rejects=0,
                    window_size=float(state.policy.window_size),
                    model_version=state.policy.model_version,
                    message=message,
                )

            state.window_index += len(window_ids)
            self._trim_tail(state, combined_records)
            return final_payload

    def _trim_tail(self, state: AuthSessionState, records: Dict[str, List[Dict[str, Any]]]) -> None:
        tail_ms = int(round(float(state.policy.window_size) * float(state.policy.overlap) * 1000))
        if tail_ms <= 0:
            state.tail_records = {"acc": [], "gyr": [], "mag": []}
            return
        all_ts = [r["timestamp"] for lst in records.values() for r in lst if "timestamp" in r]
        if not all_ts:
            state.tail_records = {"acc": [], "gyr": [], "mag": []}
            return
        max_ts = max(all_ts)
        cutoff = max_ts - tail_ms
        trimmed = {}
        for sensor in ("acc", "gyr", "mag"):
            trimmed[sensor] = [r for r in records.get(sensor, []) if int(r.get("timestamp", 0)) >= cutoff]
        state.tail_records = trimmed

