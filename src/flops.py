"""Analytical parameter / active-FLOP estimates for LiteMoD.

The functions below count forward-pass model FLOPs.  A training step is
roughly 3x that (forward + backward); ``estimate`` reports both.  The
dense/fast ratio is the same either way.  These estimates are useful for
comparing architectures before running a wall-clock benchmark; they are not a
substitute for measured tokens/sec.
"""

from __future__ import annotations

import argparse

import yaml

from .model import ModelConfig, model_config_from_dict


def _linear_flops_per_token(cfg: ModelConfig) -> float:
    """Q/K/V/O + SwiGLU matmul FLOPs for one token in one dense block."""
    d = cfg.n_embd
    h = cfg.n_head
    kv = cfg.n_kv_head
    # Q and O: 2 * d * d each. K and V: 2 * d * (d * kv / h) each.
    qkvo = 4.0 * d * d * (1.0 + kv / h)
    mlp = 6.0 * d * cfg.mlp_hidden  # three matrices, 2 FLOPs per multiply-add
    return qkvo + mlp


def _attention_flops(cfg: ModelConfig, n_tokens: int) -> float:
    """QK^T + AV for a self-attention call over ``n_tokens`` tokens."""
    return 4.0 * n_tokens * n_tokens * cfg.n_embd


def dense_layer_flops(cfg: ModelConfig, seq_len: int) -> float:
    return _linear_flops_per_token(cfg) * seq_len + _attention_flops(cfg, seq_len)


def mod_layer_flops(cfg: ModelConfig, seq_len: int) -> float:
    active = max(1, min(seq_len, int(round(seq_len * cfg.mod_capacity))))
    # Router: one d -> 1 projection over all tokens.
    router = 2.0 * seq_len * cfg.n_embd
    return _linear_flops_per_token(cfg) * active + _attention_flops(cfg, active) + router


def estimate(cfg: ModelConfig, seq_len: int) -> dict:
    if seq_len > cfg.block_size:
        raise ValueError(f"seq_len {seq_len} > block_size {cfg.block_size}")

    n_dense = cfg.n_layer
    n_mod = 0
    if cfg.arch == "mod":
        first = 1 if cfg.mod_first_dense else 0
        last = 1 if (cfg.mod_last_dense and cfg.n_layer > first) else 0
        n_dense = first + last
        n_mod = cfg.n_layer - n_dense

    body = n_dense * dense_layer_flops(cfg, seq_len)
    body += n_mod * mod_layer_flops(cfg, seq_len)
    head = 2.0 * seq_len * cfg.n_embd * cfg.vocab_size
    total = body + head

    embed = cfg.vocab_size * cfg.n_embd
    attn_params = 2.0 * cfg.n_embd * cfg.n_embd * (1.0 + cfg.n_kv_head / cfg.n_head)
    mlp_params = 3.0 * cfg.n_embd * cfg.mlp_hidden
    router_params = n_mod * (cfg.n_embd + 1)
    norm_params = 2 * cfg.n_layer * cfg.n_embd
    head_params = 0 if cfg.tied_embeddings else embed
    params = (
        embed
        + cfg.n_layer * (attn_params + mlp_params)
        + router_params
        + norm_params
        + cfg.n_embd
        + head_params
    )

    return {
        "arch": cfg.arch,
        "seq_len": seq_len,
        "dense_layers": n_dense,
        "mod_layers": n_mod,
        "capacity": cfg.mod_capacity if cfg.arch == "mod" else 1.0,
        "effective_depth": n_dense + n_mod * (cfg.mod_capacity if cfg.arch == "mod" else 1.0),
        "params": int(round(params)),
        "flops_per_sequence": int(round(total)),
        "forward_flops_per_token": int(round(total / seq_len)),
        "training_flops_per_token": int(round(3 * total / seq_len)),
        "flops_per_token": int(round(total / seq_len)),
        "body_flops_per_token": int(round(body / seq_len)),
        "head_flops_per_token": int(round(head / seq_len)),
    }


def _load_model_cfg(path: str) -> ModelConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return model_config_from_dict(raw["model"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analytical LiteMoD FLOP estimates")
    parser.add_argument("--dense", default="configs/dense_tiny.yaml")
    parser.add_argument("--fast", default="configs/fast_tiny.yaml")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=None)
    args = parser.parse_args()

    dense = _load_model_cfg(args.dense)
    fast = _load_model_cfg(args.fast)
    if args.vocab_size is not None:
        dense.vocab_size = args.vocab_size
        fast.vocab_size = args.vocab_size

    d = estimate(dense, args.seq_len)
    f = estimate(fast, args.seq_len)

    print(f"seq_len={args.seq_len}  vocab={dense.vocab_size}")
    print(f"{'model':<8} {'params':>12} {'depth':>7} {'train MFLOPs/tok':>17} {'relative':>9}")
    for name, item in (("dense", d), ("fast", f)):
        print(
            f"{name:<8} {item['params']/1e6:>10.1f}M "
            f"{item['effective_depth']:>7.2f} "
            f"{item['training_flops_per_token']/1e6:>17.1f} "
            f"{item['training_flops_per_token']/d['training_flops_per_token']:>8.2f}x"
        )
    print(f"theoretical speedup: {d['training_flops_per_token']/f['training_flops_per_token']:.2f}x")
    print(f"fast model active-FLOP reduction: {(1 - f['training_flops_per_token']/d['training_flops_per_token'])*100:.1f}%")


if __name__ == "__main__":
    main()
