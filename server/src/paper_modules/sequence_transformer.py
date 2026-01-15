from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class TokenTransformerLMConfig:
    """Causal self-attention language model configuration for token sequences."""

    vocab_size: int
    block_size: int
    sos_token: int
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1


class _CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TokenTransformerLMConfig) -> None:
        super().__init__()
        if cfg.n_embd % cfg.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = int(cfg.n_head)
        self.head_dim = int(cfg.n_embd // cfg.n_head)
        self.qkv = nn.Linear(int(cfg.n_embd), int(cfg.n_embd) * 3)
        self.proj = nn.Linear(int(cfg.n_embd), int(cfg.n_embd))
        self.attn_drop = nn.Dropout(float(cfg.dropout))
        self.resid_drop = nn.Dropout(float(cfg.dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, embd = x.shape
        qkv = self.qkv(x)  # (B, T, 3*C)
        q, k, v = qkv.split(embd, dim=2)

        def _shape(t: torch.Tensor) -> torch.Tensor:
            return t.view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, Hd)

        q = _shape(q)
        k = _shape(k)
        v = _shape(v)

        try:
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=float(self.attn_drop.p) if self.training else 0.0,
                is_causal=True,
            )
        except TypeError:
            # Fallback for older PyTorch variants that do not support `is_causal`.
            mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=float(self.attn_drop.p) if self.training else 0.0,
            )

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, embd)  # (B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class _MLP(nn.Module):
    def __init__(self, cfg: TokenTransformerLMConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(cfg.n_embd), int(4 * cfg.n_embd)),
            nn.GELU(),
            nn.Linear(int(4 * cfg.n_embd), int(cfg.n_embd)),
            nn.Dropout(float(cfg.dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Block(nn.Module):
    def __init__(self, cfg: TokenTransformerLMConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(int(cfg.n_embd))
        self.attn = _CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(int(cfg.n_embd))
        self.mlp = _MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TokenTransformerLM(nn.Module):
    """Optimized token LM for real-time authentication scoring.

    Scoring follows the paper's log-likelihood definition:
        S_auth = (1/n) * sum_i log p(s_i | s_<i, u)
    where we model the conditional probability with a causal Transformer.
    """

    def __init__(self, cfg: TokenTransformerLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(int(cfg.vocab_size), int(cfg.n_embd))
        self.pos_emb = nn.Embedding(int(cfg.block_size), int(cfg.n_embd))
        self.drop = nn.Dropout(float(cfg.dropout))
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(int(cfg.n_layer))])
        self.ln_f = nn.LayerNorm(int(cfg.n_embd))
        self.head = nn.Linear(int(cfg.n_embd), int(cfg.vocab_size), bias=False)

    def _build_inputs(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"Expected tokens as (B, L), got {tokens.shape}")
        bsz, seq_len = tokens.shape
        if seq_len != int(self.cfg.block_size):
            raise ValueError(f"Token length {seq_len} != block_size {self.cfg.block_size}")
        sos = torch.full((bsz, 1), int(self.cfg.sos_token), device=tokens.device, dtype=tokens.dtype)
        return torch.cat([sos, tokens[:, :-1]], dim=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self._build_inputs(tokens)
        bsz, seq_len = x.shape
        pos = torch.arange(0, seq_len, device=x.device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
        h = self.drop(self.tok_emb(x) + self.pos_emb(pos))
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h)  # (B, L, vocab)

    @torch.no_grad()
    def score(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return per-sample average log-likelihood; larger is more genuine."""
        logits = self(tokens)
        logp = F.log_softmax(logits, dim=-1)
        target = tokens.to(dtype=torch.long)
        picked = logp.gather(-1, target.unsqueeze(-1)).squeeze(-1)  # (B, L)
        return picked.mean(dim=1)


def save_transformer(path: Path, model: TokenTransformerLM) -> Tuple[Path, Path]:
    path = Path(path)
    cfg_path = path.with_suffix(".json")
    cfg_path.write_text(json.dumps(asdict(model.cfg), indent=2), encoding="utf-8")
    torch.save(model.state_dict(), path)
    return path, cfg_path


def load_transformer(path: Path, *, device: torch.device, cfg_path: Optional[Path] = None) -> TokenTransformerLM:
    path = Path(path)
    cfg_path = cfg_path or path.with_suffix(".json")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing transformer config json: {cfg_path}")
    cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg = TokenTransformerLMConfig(**cfg_raw)
    model = TokenTransformerLM(cfg).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def _demo() -> None:
    torch.manual_seed(7)

    cfg = TokenTransformerLMConfig(vocab_size=129, block_size=50, sos_token=128, n_layer=2, n_head=4, n_embd=128)
    model = TokenTransformerLM(cfg).eval()
    tokens = torch.randint(0, 128, (3, cfg.block_size), dtype=torch.long)
    scores = model.score(tokens)

    print("== Transformer Sequence Modeling Demo ==")
    print("tokens:", tuple(tokens.shape))
    print("scores:", [f"{float(s):.4f}" for s in scores])


if __name__ == "__main__":
    _demo()

