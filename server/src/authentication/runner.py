from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from ..paper_modules.hysterauth import HysterAuth, HysterAuthConfig
from ..paper_modules.sequence_transformer import TokenTransformerLM, TokenTransformerLMConfig
from ..paper_modules.vector_quantization import VectorQuantizer, VectorQuantizerConfig, serialize_tokens_to_base64
from ..config import settings
from ..utils.ca_train import ensure_ca_train_on_path


@dataclass(frozen=True)
class AuthRunConfig:
    user: str
    window_size: float
    overlap: float
    target_width: int
    vq_config: Dict[str, Any] = field(default_factory=dict)
    transformer_config: Dict[str, Any] = field(default_factory=dict)
    hysterauth_config: Dict[str, Any] = field(default_factory=dict)
    vq_checkpoint: Optional[str] = None
    transformer_checkpoint: Optional[str] = None
    model_version: str = ""


def _server_root() -> Path:
    # server/src/authentication/runner.py -> server/
    return Path(__file__).resolve().parents[2]


def _default_models_root(server_root: Path) -> Path:
    # Keep paths configurable via Settings (env/.env), while preserving the historical default.
    return Path(settings.data_storage_path).parent / "models"


def _resolve_optional_path(raw: Any, *, bases: Sequence[Path]) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw).strip()
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        for base in bases:
            candidate = (Path(base) / path).resolve()
            if candidate.exists():
                return str(candidate)
        path = (Path(bases[0]) / path).resolve()
    return str(path)


def load_best_policy(
    user: str,
    *,
    models_root: Optional[Path] = None,
    policy_path: Optional[Path] = None,
) -> AuthRunConfig:
    server_root = _server_root()
    models_root = Path(models_root) if models_root is not None else _default_models_root(server_root)
    if policy_path is None:
        policy_path = models_root / user / "best_lock_policy.json"
    policy_path = Path(policy_path)
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing policy json for user={user}: {policy_path}")

    payload = json.loads(policy_path.read_text(encoding="utf-8"))
    policy = payload.get(user) or payload.get(str(user))
    if not isinstance(policy, dict):
        raise ValueError(f"Unexpected best_lock_policy.json format: {policy_path}")

    window_size = float(policy.get("window", 0.0))
    overlap = float(policy.get("overlap", 0.5))
    target_width = int(policy.get("target_width", max(1, int(round(window_size * 100)))))

    vq_cfg = policy.get("vq_config") or {}
    if not isinstance(vq_cfg, dict):
        raise ValueError("vq_config must be a dict.")
    if not vq_cfg:
        vq_cfg = asdict(VectorQuantizerConfig(input_dim=12))

    transformer_cfg = policy.get("transformer_config") or {}
    if not isinstance(transformer_cfg, dict):
        raise ValueError("transformer_config must be a dict.")
    if not transformer_cfg:
        num_codebook = int(vq_cfg.get("num_codebook_vectors", VectorQuantizerConfig.num_codebook_vectors))
        vocab_size = int(num_codebook) + 1
        transformer_cfg = asdict(
            TokenTransformerLMConfig(
                vocab_size=vocab_size,
                block_size=int(target_width),
                sos_token=int(vocab_size - 1),
            )
        )

    hyst_cfg = policy.get("hysterauth_config") or {}
    if not isinstance(hyst_cfg, dict):
        raise ValueError("hysterauth_config must be a dict.")
    if not hyst_cfg:
        hyst_cfg = asdict(HysterAuthConfig())

    resolve_bases = (server_root, policy_path.parent, models_root)
    vq_ckpt = _resolve_optional_path(policy.get("vq_checkpoint"), bases=resolve_bases)
    tf_ckpt = _resolve_optional_path(policy.get("transformer_checkpoint"), bases=resolve_bases)

    model_version = str(policy.get("model_version", "") or "")
    if not model_version:
        model_version = "vq_transformer_hysterauth"

    return AuthRunConfig(
        user=str(policy.get("user", user)),
        window_size=float(window_size),
        overlap=float(overlap),
        target_width=int(target_width),
        vq_config=dict(vq_cfg),
        transformer_config=dict(transformer_cfg),
        hysterauth_config=dict(hyst_cfg),
        vq_checkpoint=vq_ckpt,
        transformer_checkpoint=tf_ckpt,
        model_version=model_version,
    )


def _build_models(cfg: AuthRunConfig, *, device: str) -> Tuple[VectorQuantizer, TokenTransformerLM, torch.device]:
    torch_device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")

    vq = VectorQuantizer(VectorQuantizerConfig(**cfg.vq_config)).to(torch_device).eval()
    if cfg.vq_checkpoint and Path(cfg.vq_checkpoint).exists():
        vq.load_state_dict(torch.load(Path(cfg.vq_checkpoint), map_location=torch_device))

    transformer = TokenTransformerLM(TokenTransformerLMConfig(**cfg.transformer_config)).to(torch_device).eval()
    if cfg.transformer_checkpoint and Path(cfg.transformer_checkpoint).exists():
        transformer.load_state_dict(torch.load(Path(cfg.transformer_checkpoint), map_location=torch_device))

    return vq, transformer, torch_device


def run_auth_inference(
    *,
    csv_path: Path,
    policy: AuthRunConfig,
    device: str = "cuda:0",
    output_csv: Optional[Path] = None,
    max_windows: Optional[int] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Offline inference helper for a server-formatted window CSV."""
    ensure_ca_train_on_path()
    from hmog_data import iter_windows_from_csv_unlabeled_with_session  # type: ignore

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input CSV: {csv_path}")

    if output_csv is None:
        server_root = _server_root()
        models_root = _default_models_root(server_root)
        out_dir = models_root / policy.user / "inference"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_csv = out_dir / f"infer_ws_{policy.window_size:.1f}.csv"
    else:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

    vq, transformer, torch_device = _build_models(policy, device=str(device))
    hyst = HysterAuth(HysterAuthConfig(**policy.hysterauth_config))

    current_session_key: Optional[str] = None

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "window_id",
                "subject",
                "session",
                "auth_score",
                "anomaly_prob",
                "k_evidence",
                "state",
                "accept",
                "interrupt",
                "tokens_b64",
            ]
        )

        count = 0
        for idx, (window_id, subject, session, window) in enumerate(
            iter_windows_from_csv_unlabeled_with_session(
                csv_path,
                window_size_sec=float(policy.window_size),
                target_width=int(policy.target_width),
            )
        ):
            if max_windows is not None and idx >= int(max_windows):
                break

            session_key = f"{subject}::{session}"
            if current_session_key is None:
                current_session_key = session_key
            elif current_session_key != session_key:
                hyst.reset()
                current_session_key = session_key

            # window: (1, 12, T) -> (1, T, 12)
            window_ts = torch.from_numpy(window[0].T).unsqueeze(0).to(torch_device, dtype=torch.float32)
            with torch.no_grad():
                codes = vq(window_ts).codes.to(dtype=torch.long)
                auth_score = float(transformer.score(codes)[0].item())
            decision = hyst.update(-auth_score)
            tokens_b64 = serialize_tokens_to_base64(codes[0].detach().cpu().numpy().astype(np.int64, copy=False))

            writer.writerow(
                [
                    window_id,
                    subject,
                    session,
                    f"{auth_score:.6f}",
                    f"{float(decision.anomaly_prob):.6f}",
                    f"{float(decision.k_evidence):.6f}",
                    decision.state,
                    int(decision.accept),
                    int(decision.interrupt),
                    tokens_b64,
                ]
            )
            count += 1

        if count == 0:
            raise ValueError(f"No valid windows produced from {csv_path}")

    meta = {
        "user": policy.user,
        "window": float(policy.window_size),
        "overlap": float(policy.overlap),
        "target_width": int(policy.target_width),
        "vq_config": dict(policy.vq_config),
        "transformer_config": dict(policy.transformer_config),
        "hysterauth_config": dict(policy.hysterauth_config),
        "vq_checkpoint": policy.vq_checkpoint,
        "transformer_checkpoint": policy.transformer_checkpoint,
        "input_csv": str(csv_path),
        "output_csv": str(output_csv),
        "max_windows": None if max_windows is None else int(max_windows),
    }
    return output_csv, meta
