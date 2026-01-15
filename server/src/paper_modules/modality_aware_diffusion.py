from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModalityAwareDDPMConfig:
    """Configuration for the modality-aware DDPM used for data augmentation."""

    timesteps: int = 50
    latent_dim: int = 64
    context_dim: int = 64
    beta_min: float = 1e-4
    beta_max: float = 0.02
    # Per-modality parameter controlling the (t/T)**phi shaping of the noise schedule.
    phi: Dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.phi is None:
            object.__setattr__(self, "phi", {"acc": 1.0, "gyr": 1.2, "mag": 0.8})


class _PerModalityEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Encode raw segments `u` into a unified latent vector.

        Args:
            u: (B, L, C)
        Returns:
            x0: (B, D)
        """
        if u.ndim != 3:
            raise ValueError(f"Expected u as (B, L, C), got {tuple(u.shape)}")
        z = self.proj(u)  # (B, L, D)
        z = z.mean(dim=1)  # (B, D)
        return self.norm(z)


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim != 1:
            raise ValueError(f"Expected t as (B,), got {tuple(t.shape)}")
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(0, half, device=t.device, dtype=torch.float32) / max(1, half - 1)
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class _EpsilonPredictor(nn.Module):
    def __init__(self, latent_dim: int, context_dim: int) -> None:
        super().__init__()
        self.t_emb = _TimeEmbedding(context_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + context_dim + context_dim, 256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if x_t.ndim != 2:
            raise ValueError(f"Expected x_t as (B, D), got {tuple(x_t.shape)}")
        if context.ndim != 2:
            raise ValueError(f"Expected context as (B, C), got {tuple(context.shape)}")
        if x_t.shape[0] != context.shape[0]:
            raise ValueError("Batch size mismatch between x_t and context.")
        t_ctx = self.t_emb(t)
        inp = torch.cat([x_t, context, t_ctx], dim=1)
        return self.net(inp)


def _make_betas(*, timesteps: int, beta_min: float, beta_max: float, phi: float, device: torch.device) -> torch.Tensor:
    t = torch.arange(1, timesteps + 1, device=device, dtype=torch.float32)
    frac = (t / float(timesteps)).clamp(0.0, 1.0) ** float(phi)
    return (float(beta_min) + (float(beta_max) - float(beta_min)) * frac).clamp(1e-8, 0.999)


class ModalityAwareDDPM(nn.Module):
    """Modality-aware DDPM for generating augmented latent fingerprints.

    This is a lightweight, runnable reference implementation aligned with the paper description:
      - per-modality encoders f_enc^(s)
      - fused cross-modal context c
      - domain-specific (per-modality) noise schedule beta_t^(s)
      - epsilon-prediction reverse process conditioned on c
    """

    def __init__(
        self,
        *,
        modalities: Iterable[str] = ("acc", "gyr", "mag"),
        in_channels: Mapping[str, int] | None = None,
        cfg: ModalityAwareDDPMConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or ModalityAwareDDPMConfig()
        self.modalities = tuple(modalities)
        if in_channels is None:
            in_channels = {m: 3 for m in self.modalities}
        self.in_channels = dict(in_channels)
        self.device_ = device or torch.device("cpu")

        self.encoders = nn.ModuleDict(
            {m: _PerModalityEncoder(int(self.in_channels[m]), int(self.cfg.latent_dim)) for m in self.modalities}
        )
        self.fuse = nn.Sequential(
            nn.Linear(int(self.cfg.latent_dim) * len(self.modalities), int(self.cfg.context_dim)),
            nn.GELU(),
            nn.Linear(int(self.cfg.context_dim), int(self.cfg.context_dim)),
        )
        self.eps_models = nn.ModuleDict(
            {m: _EpsilonPredictor(int(self.cfg.latent_dim), int(self.cfg.context_dim)) for m in self.modalities}
        )

        self.register_buffer("_dummy", torch.empty(0), persistent=False)
        self._register_schedules()

    @property
    def device(self) -> torch.device:
        return self._dummy.device

    def _register_schedules(self) -> None:
        for m in self.modalities:
            betas = _make_betas(
                timesteps=int(self.cfg.timesteps),
                beta_min=float(self.cfg.beta_min),
                beta_max=float(self.cfg.beta_max),
                phi=float(self.cfg.phi.get(m, 1.0)),
                device=self.device_,
            )
            alphas = 1.0 - betas
            alpha_bars = torch.cumprod(alphas, dim=0)
            self.register_buffer(f"betas_{m}", betas, persistent=False)
            self.register_buffer(f"alphas_{m}", alphas, persistent=False)
            self.register_buffer(f"alpha_bars_{m}", alpha_bars, persistent=False)

    def encode(self, inputs: Mapping[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        x0_by_modality: Dict[str, torch.Tensor] = {}
        latents = []
        for m in self.modalities:
            if m not in inputs:
                raise KeyError(f"Missing modality input: {m}")
            x0 = self.encoders[m](inputs[m].to(self.device))
            x0_by_modality[m] = x0
            latents.append(x0)
        context = self.fuse(torch.cat(latents, dim=1))
        return x0_by_modality, context

    def q_sample(self, x0: torch.Tensor, *, t: torch.Tensor, modality: str, noise: torch.Tensor | None = None) -> torch.Tensor:
        """Forward diffusion sampling q(x_t | x_0)."""
        if modality not in self.modalities:
            raise KeyError(f"Unknown modality: {modality}")
        if x0.ndim != 2:
            raise ValueError(f"Expected x0 as (B, D), got {tuple(x0.shape)}")
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bars = getattr(self, f"alpha_bars_{modality}")  # (T,)
        if t.ndim != 1 or t.shape[0] != x0.shape[0]:
            raise ValueError("t must be (B,) aligned with x0 batch size.")
        t_idx = (t.long().clamp(1, int(self.cfg.timesteps)) - 1).to(self.device)
        abar = alpha_bars.gather(0, t_idx).unsqueeze(1)  # (B,1)
        return torch.sqrt(abar) * x0 + torch.sqrt(1.0 - abar) * noise

    def p_mean(self, x_t: torch.Tensor, *, t: torch.Tensor, modality: str, context: torch.Tensor) -> torch.Tensor:
        """Reverse mean μ_θ(x_t, t, c) as described in the paper."""
        if modality not in self.modalities:
            raise KeyError(f"Unknown modality: {modality}")
        betas = getattr(self, f"betas_{modality}")
        alphas = getattr(self, f"alphas_{modality}")
        alpha_bars = getattr(self, f"alpha_bars_{modality}")
        t_idx = (t.long().clamp(1, int(self.cfg.timesteps)) - 1).to(self.device)
        beta_t = betas.gather(0, t_idx).unsqueeze(1)
        alpha_t = alphas.gather(0, t_idx).unsqueeze(1)
        abar_t = alpha_bars.gather(0, t_idx).unsqueeze(1)

        eps = self.eps_models[modality](x_t, t, context)
        coef = (1.0 - alpha_t) / torch.sqrt(1.0 - abar_t)
        return (1.0 / torch.sqrt(alpha_t)) * (x_t - coef * eps)


def _demo() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")

    cfg = ModalityAwareDDPMConfig(timesteps=20, latent_dim=32, context_dim=32)
    ddpm = ModalityAwareDDPM(cfg=cfg, device=device).to(device)

    bsz, length = 4, 64
    inputs = {
        "acc": torch.randn(bsz, length, 3, device=device),
        "gyr": torch.randn(bsz, length, 3, device=device),
        "mag": torch.randn(bsz, length, 3, device=device),
    }

    x0_by_mod, context = ddpm.encode(inputs)
    t = torch.randint(1, cfg.timesteps + 1, (bsz,), device=device)

    print("== Modality-Aware DDPM Demo ==")
    print("context:", tuple(context.shape))
    for m in ddpm.modalities:
        x0 = x0_by_mod[m]
        x_t = ddpm.q_sample(x0, t=t, modality=m)
        mu = ddpm.p_mean(x_t, t=t, modality=m, context=context)
        betas = getattr(ddpm, f"betas_{m}")
        print(
            f"{m}: x0={tuple(x0.shape)} xt={tuple(x_t.shape)} mu={tuple(mu.shape)} "
            f"beta[0]={float(betas[0]):.6f} beta[-1]={float(betas[-1]):.6f}"
        )


if __name__ == "__main__":
    _demo()

