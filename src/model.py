"""LiteMoD: a small decoder-only transformer with token-level Mixture-of-Depths.

The dense baseline and the fast model share every component except the block
type.  A ``MoDBlock`` scores every token, sends only the top ``capacity * T``
tokens through its attention + MLP sublayers, and passes the rest through the
residual stream untouched.  Because the selected token indices are sorted by
position, the usual lower-triangular causal mask is still valid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    arch: str = "dense"                 # "dense" | "mod"
    vocab_size: int = 16384
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 8
    n_kv_head: int = 2
    n_embd: int = 512
    mlp_hidden: int = 1408
    dropout: float = 0.0
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tied_embeddings: bool = True

    # Mixture-of-Depths knobs (ignored when arch == "dense").
    mod_capacity: float = 0.5
    mod_first_dense: bool = True
    mod_last_dense: bool = True
    mod_aux_weight: float = 0.01
    mod_router_z_weight: float = 0.001
    mod_router_noise: float = 0.5

    def __post_init__(self) -> None:
        if self.arch not in {"dense", "mod"}:
            raise ValueError(f"arch must be 'dense' or 'mod', got {self.arch!r}")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head != 0:
            raise ValueError("n_head must be divisible by n_kv_head")
        head_dim = self.n_embd // self.n_head
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        if not (0.0 < self.mod_capacity <= 1.0):
            raise ValueError("mod_capacity must be in (0, 1]")
        if self.n_layer < 1:
            raise ValueError("n_layer must be >= 1")


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    theta: float,
    device: torch.device | str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return cos/sin caches with shape ``[seq_len, head_dim // 2]``."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embeddings.

    ``x`` is ``[B, H, T, D]`` and ``cos``/``sin`` are ``[B, T, D // 2]``.
    """
    dtype = x.dtype
    x = x.float()
    cos = cos.float().unsqueeze(1)   # [B, 1, T, D/2]
    sin = sin.float().unsqueeze(1)
    x1, x2 = x.chunk(2, dim=-1)
    out = torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.to(dtype)


class CausalSelfAttention(nn.Module):
    """GQA + RoPE attention using PyTorch's fused SDPA / FlashAttention kernel."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.n_rep = cfg.n_head // cfg.n_kv_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.dropout = cfg.dropout

        self.wq = nn.Linear(cfg.n_embd, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_head * self.head_dim, cfg.n_embd, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        q = self.wq(x).view(bsz, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seq_len, self.n_kv_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        if self.n_rep > 1:
            # Repeat KV heads to match the query-head layout: [0,0,...,1,1,...].
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # selected tokens are sorted by original position, so is_causal=True is
        # still exactly the right mask after gathering.
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)
        self.w2 = nn.Linear(cfg.mlp_hidden, cfg.n_embd, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, cfg.mlp_hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DenseBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class MoDBlock(nn.Module):
    """A transformer block that processes only the top-k routed tokens.

    The router is trained with a binary cross-entropy auxiliary loss whose
    target is the top-k selection itself (the standard MoD predictor loss).
    Gumbel noise is added to the *selection* logits during training so the
    router does not collapse onto the same positional subset; the auxiliary
    loss always sees clean logits.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.capacity = cfg.mod_capacity
        self.router_noise = cfg.mod_router_noise
        self.router_z_weight = cfg.mod_router_z_weight

        self.norm1 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)
        self.router = nn.Linear(cfg.n_embd, 1, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        collect_aux: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = x.shape
        clean_logits = self.router(x).squeeze(-1)          # [B, T]

        select_logits = clean_logits
        if self.training and self.router_noise > 0:
            u = torch.rand_like(clean_logits).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            select_logits = clean_logits + self.router_noise * gumbel

        k = max(1, min(seq_len, int(round(seq_len * self.capacity))))
        _, idx = torch.topk(select_logits, k=k, dim=1, sorted=False)
        idx, _ = torch.sort(idx, dim=1)                    # ascending positions

        aux = x.new_zeros(())
        if collect_aux:
            target = torch.zeros_like(clean_logits)
            target.scatter_(1, idx, 1.0)
            bce = F.binary_cross_entropy_with_logits(clean_logits, target)
            z_loss = clean_logits.float().square().mean()
            aux = bce + self.router_z_weight * z_loss

        # Gather the routed tokens, run the block, then scatter back.
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, dim)
        x_sel = torch.gather(x, 1, gather_idx)
        cos_sel = torch.gather(cos, 1, idx.unsqueeze(-1).expand(-1, -1, cos.size(-1)))
        sin_sel = torch.gather(sin, 1, idx.unsqueeze(-1).expand(-1, -1, sin.size(-1)))

        h = x_sel + self.attn(self.norm1(x_sel), cos_sel, sin_sel)
        h = h + self.mlp(self.norm2(h))
        out = x.scatter(1, gather_idx, h)
        return out, aux


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class LiteMoD(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)

        blocks = []
        for layer in range(cfg.n_layer):
            is_mod = (
                cfg.arch == "mod"
                and not (cfg.mod_first_dense and layer == 0)
                and not (cfg.mod_last_dense and layer == cfg.n_layer - 1)
            )
            blocks.append(MoDBlock(cfg) if is_mod else DenseBlock(cfg))
        self.blocks = nn.ModuleList(blocks)

        self.norm_f = RMSNorm(cfg.n_embd, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tied_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(
            cfg.block_size,
            cfg.n_embd // cfg.n_head,
            cfg.rope_theta,
            device="cpu",
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init_weights)
        self._apply_residual_scaling()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def _apply_residual_scaling(self) -> None:
        std = 0.02 / math.sqrt(2 * self.cfg.n_layer)
        for name, param in self.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w2.weight"):
                nn.init.normal_(param, mean=0.0, std=std)

    @property
    def n_mod_layers(self) -> int:
        return sum(isinstance(block, MoDBlock) for block in self.blocks)

    @property
    def n_dense_layers(self) -> int:
        return len(self.blocks) - self.n_mod_layers

    @property
    def effective_depth(self) -> float:
        return self.n_dense_layers + self.n_mod_layers * self.cfg.mod_capacity

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        bsz, seq_len = idx.shape
        if seq_len > self.cfg.block_size:
            raise ValueError(
                f"sequence length {seq_len} exceeds block_size {self.cfg.block_size}"
            )

        x = self.drop(self.tok_emb(idx))
        cos = self.cos[:seq_len].unsqueeze(0).expand(bsz, -1, -1)
        sin = self.sin[:seq_len].unsqueeze(0).expand(bsz, -1, -1)

        aux_total = x.new_zeros(())
        n_aux = 0
        for block in self.blocks:
            if isinstance(block, MoDBlock):
                x, aux = block(x, cos, sin, collect_aux=self.training)
                if self.training:
                    aux_total = aux_total + aux
                    n_aux += 1
            else:
                x = block(x, cos, sin)

        x = self.norm_f(x)
        logits = self.lm_head(x)
        aux_loss = aux_total / max(1, n_aux)

        if targets is None:
            return logits, None, aux_loss

        lm_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )
        return logits, lm_loss, aux_loss


# ---------------------------------------------------------------------------
# Helpers shared by training / benchmarking
# ---------------------------------------------------------------------------


def model_config_from_dict(raw: dict) -> ModelConfig:
    """Build a ModelConfig while failing loudly on unknown YAML keys."""
    known = {field.name for field in fields(ModelConfig)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown model config keys: {sorted(unknown)}")
    return ModelConfig(**raw)


def count_parameters(model: nn.Module, non_embedding: bool = False) -> int:
    total = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if non_embedding and hasattr(model, "tok_emb"):
        total -= model.tok_emb.weight.numel()
    return int(total)
