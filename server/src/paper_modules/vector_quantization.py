from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class VectorQuantizerConfig:
    """Vector-quantization configuration (VQ-VAE style)."""

    num_codebook_vectors: int = 256
    latent_dim: int = 32
    input_dim: int = 12
    commitment_beta: float = 0.25


@dataclass(frozen=True)
class VQOutput:
    z_e: torch.Tensor
    z_q: torch.Tensor
    codes: torch.Tensor
    perplexity: float
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor


class _Encoder(nn.Module):
    def __init__(self, *, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), 128),
            nn.GELU(),
            nn.Linear(128, int(latent_dim)),
        )
        self.norm = nn.LayerNorm(int(latent_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x as (B, T, M), got {tuple(x.shape)}")
        return self.norm(self.net(x))


class VectorQuantizer(nn.Module):
    """Vector Quantization (VQ) module producing discrete behavioral tokens.

    Paper alignment:
      - encoder E(·) maps sensor windows to a latent z_e
      - a learnable codebook C is used for nearest-neighbor quantization
      - discrete indices are used as symbolic tokens for sequence modeling
    """

    def __init__(self, cfg: VectorQuantizerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = _Encoder(input_dim=cfg.input_dim, latent_dim=cfg.latent_dim)
        self.codebook = nn.Embedding(int(cfg.num_codebook_vectors), int(cfg.latent_dim))
        nn.init.uniform_(self.codebook.weight, -1.0 / cfg.num_codebook_vectors, 1.0 / cfg.num_codebook_vectors)

    def quantize(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Quantize `z_e` to the nearest codebook vectors.

        Args:
            z_e: (B, T, D)
        Returns:
            z_q: (B, T, D)
            codes: (B, T) int64
            perplexity: float
        """
        if z_e.ndim != 3:
            raise ValueError(f"Expected z_e as (B, T, D), got {tuple(z_e.shape)}")
        bsz, seq_len, dim = z_e.shape
        flat = z_e.reshape(-1, dim)  # (B*T, D)

        # ||z - c||^2 = ||z||^2 + ||c||^2 - 2 z·c
        z2 = (flat ** 2).sum(dim=1, keepdim=True)  # (B*T, 1)
        c2 = (self.codebook.weight ** 2).sum(dim=1).unsqueeze(0)  # (1, K)
        dist = z2 + c2 - 2.0 * flat @ self.codebook.weight.t()  # (B*T, K)
        codes = torch.argmin(dist, dim=1)  # (B*T,)
        z_q = self.codebook(codes).view(bsz, seq_len, dim)

        # Perplexity
        with torch.no_grad():
            one_hot = F.one_hot(codes, num_classes=int(self.cfg.num_codebook_vectors)).float()
            avg_probs = one_hot.mean(dim=0)
            entropy = -(avg_probs * (avg_probs + 1e-10).log()).sum()
            perplexity = float(torch.exp(entropy).item())

        return z_q, codes.view(bsz, seq_len), perplexity

    def forward(self, x: torch.Tensor) -> VQOutput:
        z_e = self.encoder(x)
        z_q, codes, perplexity = self.quantize(z_e)

        # VQ-VAE losses (for training). These are included for completeness but
        # are not required for inference/tokenization.
        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = float(self.cfg.commitment_beta) * F.mse_loss(z_e, z_q.detach())

        # Straight-through estimator for backprop:
        z_q_st = z_e + (z_q - z_e).detach()
        return VQOutput(
            z_e=z_e,
            z_q=z_q_st,
            codes=codes,
            perplexity=float(perplexity),
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )


def serialize_tokens_to_base64(tokens: np.ndarray) -> str:
    """Serialize integer tokens into a base64 string (portable for JSON logs)."""
    if tokens.ndim != 1:
        raise ValueError(f"Expected tokens as 1D array, got {tokens.shape}")
    if np.any(tokens < 0):
        raise ValueError("Tokens must be non-negative integers.")
    max_token = int(tokens.max(initial=0))
    dtype = np.uint16 if max_token <= np.iinfo(np.uint16).max else np.uint32
    payload = {
        "dtype": "u16" if dtype == np.uint16 else "u32",
        "length": int(tokens.shape[0]),
        "data_b64": base64.b64encode(tokens.astype(dtype, copy=False).tobytes()).decode("ascii"),
    }
    return base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")


def deserialize_tokens_from_base64(payload_b64: str) -> np.ndarray:
    raw = base64.b64decode(payload_b64.encode("ascii"))
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Invalid token payload.")
    dtype = payload.get("dtype")
    length = int(payload.get("length", 0))
    data_b64 = payload.get("data_b64")
    if dtype not in {"u16", "u32"} or not isinstance(data_b64, str) or length <= 0:
        raise ValueError("Invalid token payload fields.")
    buf = base64.b64decode(data_b64.encode("ascii"))
    np_dtype = np.uint16 if dtype == "u16" else np.uint32
    arr = np.frombuffer(buf, dtype=np_dtype)
    if int(arr.size) != int(length):
        raise ValueError(f"Token length mismatch: expected={length} got={arr.size}")
    return arr.astype(np.int64, copy=False)


def _demo() -> None:
    torch.manual_seed(7)
    np.random.seed(7)

    cfg = VectorQuantizerConfig(num_codebook_vectors=128, latent_dim=32, input_dim=12)
    vq = VectorQuantizer(cfg).eval()

    bsz, t, m = 2, 50, 12
    x = torch.randn(bsz, t, m)
    out = vq(x)
    tokens = out.codes[0].detach().cpu().numpy().astype(np.int64, copy=False)

    packed = serialize_tokens_to_base64(tokens)
    roundtrip = deserialize_tokens_from_base64(packed)

    print("== Vector Quantization Demo ==")
    print("x:", tuple(x.shape), "z_e:", tuple(out.z_e.shape), "tokens:", tuple(out.codes.shape))
    print("perplexity:", f"{out.perplexity:.2f}")
    print("tokens[:10]:", tokens[:10].tolist())
    print("serialize/deserialize ok:", bool(np.array_equal(tokens, roundtrip)))


if __name__ == "__main__":
    _demo()

