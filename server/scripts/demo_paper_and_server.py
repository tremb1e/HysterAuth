"""Runnable examples for the paper modules and the server pipeline.

This script demonstrates:
  1) Each paper module with synthetic inputs (DDPM, VQ, Transformer, HysterAuth)
  2) The end-to-end server inference pipeline on a simulated sensor batch:
       sensor stream -> discrete tokens -> Transformer score -> HysterAuth decision
  3) Basic HTTP app health checks via FastAPI TestClient

Run:
  python scripts/demo_paper_and_server.py
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _set_demo_env(tmp_root: Path) -> None:
    # Prefer CPU for a lightweight, portable demo.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    raw_root = tmp_root / "raw_data"
    processed_root = tmp_root / "processed_data"
    inference_root = tmp_root / "inference"
    logs_root = tmp_root / "logs"

    raw_root.mkdir(parents=True, exist_ok=True)
    processed_root.mkdir(parents=True, exist_ok=True)
    inference_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    os.environ["DATA_STORAGE_PATH"] = str(raw_root)
    os.environ["PROCESSED_DATA_PATH"] = str(processed_root)
    os.environ["INFERENCE_STORAGE_PATH"] = str(inference_root)
    os.environ["LOG_PATH"] = str(logs_root)


def demo_paper_modules() -> None:
    import torch

    from src.paper_modules.hysterauth import HysterAuth, HysterAuthConfig
    from src.paper_modules.modality_aware_diffusion import ModalityAwareDDPM, ModalityAwareDDPMConfig
    from src.paper_modules.sequence_transformer import TokenTransformerLM, TokenTransformerLMConfig
    from src.paper_modules.vector_quantization import VectorQuantizer, VectorQuantizerConfig, serialize_tokens_to_base64

    print("\n== Demo: Paper Modules ==")

    torch.manual_seed(7)
    np.random.seed(7)

    cfg_ddpm = ModalityAwareDDPMConfig(timesteps=12, latent_dim=16, context_dim=16)
    ddpm = ModalityAwareDDPM(cfg=cfg_ddpm, device=torch.device("cpu")).eval()
    inputs = {
        "acc": torch.randn(2, 64, 3),
        "gyr": torch.randn(2, 64, 3),
        "mag": torch.randn(2, 64, 3),
    }
    x0_by_modality, context = ddpm.encode(inputs)
    t = torch.randint(1, cfg_ddpm.timesteps + 1, (2,))
    mu = ddpm.p_mean(ddpm.q_sample(x0_by_modality["acc"], t=t, modality="acc"), t=t, modality="acc", context=context)
    print("DDPM: context", tuple(context.shape), "mu(acc)", tuple(mu.shape))

    cfg_vq = VectorQuantizerConfig(num_codebook_vectors=64, latent_dim=32, input_dim=12)
    vq = VectorQuantizer(cfg_vq).eval()
    x = torch.randn(2, 20, 12)
    vq_out = vq(x)
    tokens = vq_out.codes[0].detach().cpu().numpy().astype(np.int64, copy=False)
    tokens_b64 = serialize_tokens_to_base64(tokens)
    print("VQ: tokens", tuple(vq_out.codes.shape), "perplexity", f"{vq_out.perplexity:.2f}", "b64_len", len(tokens_b64))

    vocab_size = int(cfg_vq.num_codebook_vectors) + 1
    cfg_tf = TokenTransformerLMConfig(
        vocab_size=vocab_size,
        block_size=20,
        sos_token=vocab_size - 1,
        n_layer=2,
        n_head=4,
        n_embd=64,
    )
    tf = TokenTransformerLM(cfg_tf).eval()
    random_tokens = torch.randint(0, vocab_size - 1, (2, cfg_tf.block_size), dtype=torch.long)
    scores = tf.score(random_tokens)
    print("Transformer: scores", [f"{float(s):.4f}" for s in scores])

    hyst = HysterAuth(HysterAuthConfig(forgetting_factor=0.9, theta_enter=2.5, theta_exit=1.0))
    seq = [-2.0] * 5 + [1.5] * 8 + [-2.0] * 5
    states = []
    for s in seq:
        states.append(hyst.update(s).state)
    print("HysterAuth: final_state", hyst.state, "switches", sum(1 for i in range(1, len(states)) if states[i] != states[i - 1]))


def _make_synthetic_batch(*, session_id: str, seconds: float = 1.0, sampling_hz: int = 100) -> Dict:
    """Build a server-compatible sensor batch dict with acc/gyr/mag samples."""
    now_ms = 1_700_000_000_000
    step_ms = int(round(1000.0 / float(sampling_hz)))
    total = max(1, int(round(float(seconds) * float(sampling_hz))))

    def _noise(scale: float) -> float:
        return float(np.random.normal(0.0, scale))

    samples: List[Dict] = []
    for i in range(total):
        ts_ms = now_ms + i * step_ms
        ts_ns = int(ts_ms) * 1_000_000
        t = i / float(sampling_hz)

        samples.append(
            {
                "sensor_name": "accelerometer",
                "timestamp_ns": ts_ns,
                "values": {
                    "x": math.sin(2.0 * math.pi * 1.0 * t) + _noise(0.02),
                    "y": math.cos(2.0 * math.pi * 1.0 * t) + _noise(0.02),
                    "z": 9.81 + _noise(0.05),
                },
                "accuracy": 3,
            }
        )
        samples.append(
            {
                "sensor_name": "gyroscope",
                "timestamp_ns": ts_ns,
                "values": {
                    "x": 0.05 * math.sin(2.0 * math.pi * 0.5 * t) + _noise(0.005),
                    "y": 0.05 * math.cos(2.0 * math.pi * 0.5 * t) + _noise(0.005),
                    "z": _noise(0.005),
                },
                "accuracy": 3,
            }
        )
        samples.append(
            {
                "sensor_name": "magnetometer",
                "timestamp_ns": ts_ns,
                "values": {
                    "x": 30.0 + 0.2 * math.sin(2.0 * math.pi * 0.2 * t) + _noise(0.05),
                    "y": -10.0 + 0.2 * math.cos(2.0 * math.pi * 0.2 * t) + _noise(0.05),
                    "z": 45.0 + _noise(0.05),
                },
                "accuracy": 3,
            }
        )

    return {"session_id": session_id, "samples": samples}


async def demo_server_pipeline(tmp_root: Path) -> None:
    from src.authentication.manager import AuthSessionManager
    from src.authentication.runner import load_best_policy

    print("\n== Demo: Server Pipeline (Synthetic Batch) ==")

    user_id = "demo_user"
    session_id = "demo_session"

    models_root = tmp_root / "models" / user_id
    models_root.mkdir(parents=True, exist_ok=True)
    policy_path = models_root / "best_lock_policy.json"
    if not policy_path.exists():
        policy = {
            user_id: {
                "user": user_id,
                "window": 0.2,
                "overlap": 0.5,
                "target_width": 20,
                "model_version": "demo_vq_transformer_hysterauth",
                # Calibrate the hysteresis mapping so that typical synthetic scores stay in Normal.
                "hysterauth_config": {"alpha": -1.0, "beta": 0.0},
            }
        }
        policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")

    policy = load_best_policy(user_id, policy_path=policy_path)
    print("Loaded policy:", f"window={policy.window_size}", f"target_width={policy.target_width}", f"model={policy.model_version}")

    manager = AuthSessionManager(max_cached_models=2, session_ttl_sec=300, max_concurrent_inference=2)
    accepted, msg, _ = manager.start_session(user_id, session_id)
    if not accepted:
        raise RuntimeError(f"Failed to start session: {msg}")

    parsed_batch = _make_synthetic_batch(session_id=session_id, seconds=1.0, sampling_hz=100)
    payload = await manager.handle_packet(user_id=user_id, session_id=session_id, parsed_batch=parsed_batch)
    if payload is None:
        raise RuntimeError("No auth result produced; try increasing the batch duration.")

    print(
        "AuthResult:",
        f"window_id={payload.window_id}",
        f"score={payload.score:.6f}",
        f"accept={payload.accept}",
        f"interrupt={payload.interrupt}",
        f"message={payload.message}",
    )


def demo_http_app() -> None:
    from fastapi.testclient import TestClient

    from src.main import app

    print("\n== Demo: HTTP Endpoints (TestClient) ==")
    client = TestClient(app)
    r_root = client.get("/")
    r_health = client.get("/health")
    print("GET /:", r_root.status_code, r_root.json().get("status"))
    print("GET /health:", r_health.status_code, r_health.json().get("status"))


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        _set_demo_env(tmp_root)
        demo_paper_modules()
        demo_http_app()
        await demo_server_pipeline(tmp_root)


if __name__ == "__main__":
    asyncio.run(main())
