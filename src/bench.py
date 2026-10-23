"""Wall-clock training-step benchmark: dense vs Mixture-of-Depths.

This uses random tokens and reuses the same batch, so it measures the model +
optimizer step rather than the data pipeline.  Run it on the target GPU; the
numbers are hardware-specific.
"""

from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path

import torch

from .flops import estimate
from .model import LiteMoD, model_config_from_dict
from .utils import (
    autocast_context,
    configure_torch,
    git_commit,
    load_json,
    load_yaml,
    save_json,
    set_seed,
    unwrap_model,
)


def parse_csv_ints(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return [int(part) for part in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", default="configs/dense_tiny.yaml")
    parser.add_argument("--fast", default="configs/fast_tiny.yaml")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--batch-sizes", type=parse_csv_ints, default=parse_csv_ints("8,16,32"))
    parser.add_argument("--seq-lens", type=parse_csv_ints, default=parse_csv_ints("256,512"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default="bench_results.json")
    return parser.parse_args()


def resolve_amp(device: torch.device, dtype_name: str) -> tuple[torch.dtype, bool]:
    if device.type != "cuda":
        return torch.float32, False
    if dtype_name == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise SystemExit("this GPU does not support bf16; pass --dtype fp16")
        return torch.bfloat16, True
    if dtype_name == "fp16":
        return torch.float16, True
    return torch.float32, False


def load_model_config(path: str, vocab_size: int, block_size: int):
    raw = load_yaml(path)
    cfg = model_config_from_dict(raw["model"])
    cfg.vocab_size = vocab_size
    cfg.block_size = block_size
    return cfg, raw


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    params = [param for param in unwrap_model(model).parameters() if param.requires_grad]
    try:
        return torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), fused=True)
    except (TypeError, RuntimeError):
        return torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is required for the speed check. Run this on the Vast.ai GPU, "
            "not on a laptop CPU."
        )
    if args.iters < 1 or args.warmup < 0:
        raise SystemExit("--iters must be >= 1 and --warmup >= 0")

    seq_lens = sorted(set(args.seq_lens))
    batch_sizes = sorted(set(args.batch_sizes))
    if not seq_lens or not batch_sizes:
        raise SystemExit("provide at least one seq length and batch size")
    if any(value < 1 for value in seq_lens) or any(value < 1 for value in batch_sizes):
        raise SystemExit("sequence lengths and batch sizes must be >= 1")
    max_seq = max(seq_lens)

    vocab_size = args.vocab_size
    if vocab_size is None:
        meta_path = Path(args.data_dir) / "meta.json"
        if meta_path.exists():
            vocab_size = int(load_json(meta_path)["vocab_size"])
        else:
            vocab_size = model_config_from_dict(load_yaml(args.dense)["model"]).vocab_size
            print(
                f"warning: {meta_path} not found; using vocab_size={vocab_size}. "
                "Run data prep for exact numbers."
            )

    set_seed(args.seed)
    configure_torch()
    device = torch.device("cuda")
    amp_dtype, amp_enabled = resolve_amp(device, args.dtype)
    gpu_name = torch.cuda.get_device_name(0)

    print("=" * 96)
    print(f"LiteMoD speed check | GPU: {gpu_name}")
    print(f"torch: {torch.__version__} | dtype: {amp_dtype} | compile: {args.compile}")
    print(f"vocab: {vocab_size:,} | seq lens: {seq_lens} | batch sizes: {batch_sizes}")
    print(f"warmup: {args.warmup} | timed iters: {args.iters}")
    print("=" * 96)
    print(
        f"{'model':<6} {'seq':>5} {'batch':>5} {'tok/step':>9} "
        f"{'ms/step':>10} {'tok/s':>12} {'peak GB':>8} {'Est. TFLOP/s':>13}"
    )

    results: list[dict] = []
    speedups: list[dict] = []

    for name, config_path in (("dense", args.dense), ("fast", args.fast)):
        cfg, _ = load_model_config(config_path, vocab_size, max_seq)
        flops = estimate(cfg, max_seq)
        print(
            f"\n[{name}] params={flops['params']/1e6:.2f}M "
            f"depth={flops['effective_depth']:.2f}/{cfg.n_layer} "
            f"forward MFLOPs/token@{max_seq}={flops['flops_per_token']/1e6:.1f}M "
            f"({config_path})"
        )

        model = LiteMoD(cfg).to(device)
        if args.compile:
            print(f"compiling {name} model (mode={args.compile_mode}) ...")
            model = torch.compile(model, mode=args.compile_mode)
        optimizer = build_optimizer(model)
        model.train()

        try:
            for seq_len in seq_lens:
                for batch_size in batch_sizes:
                    x = None
                    y = None
                    try:
                        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

                        def train_step() -> None:
                            optimizer.zero_grad(set_to_none=True)
                            with autocast_context(device, amp_dtype, amp_enabled):
                                _, lm_loss, aux_loss = model(x, y)
                                loss = lm_loss + cfg.mod_aux_weight * aux_loss
                            loss.backward()
                            optimizer.step()

                        for _ in range(args.warmup):
                            train_step()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()

                        times = []
                        for _ in range(args.iters):
                            start = time.perf_counter()
                            train_step()
                            torch.cuda.synchronize()
                            times.append(time.perf_counter() - start)

                        median_s = statistics.median(times)
                        mean_s = statistics.fmean(times)
                        tokens_per_step = batch_size * seq_len
                        tok_per_sec = tokens_per_step / median_s
                        peak_gb = torch.cuda.max_memory_allocated() / 1e9
                        item = estimate(cfg, seq_len)
                        effective_tflops = item["training_flops_per_token"] * tok_per_sec / 1e12

                        row = {
                            "model": name,
                            "config": config_path,
                            "seq_len": seq_len,
                            "batch_size": batch_size,
                            "tokens_per_step": tokens_per_step,
                            "ms_median": median_s * 1000.0,
                            "ms_mean": mean_s * 1000.0,
                            "tok_per_sec": tok_per_sec,
                            "peak_mem_gb": peak_gb,
                            "params": item["params"],
                            "effective_depth": item["effective_depth"],
                            "forward_flops_per_token": item["forward_flops_per_token"],
                            "training_flops_per_token": item["training_flops_per_token"],
                            "effective_tflops": effective_tflops,
                            "error": None,
                        }
                        results.append(row)
                        print(
                            f"{name:<6} {seq_len:>5} {batch_size:>5} {tokens_per_step:>9,} "
                            f"{median_s * 1000.0:>10.2f} {tok_per_sec:>12,.0f} "
                            f"{peak_gb:>8.2f} {effective_tflops:>13.1f}"
                        )
                    except torch.cuda.OutOfMemoryError:
                        print(
                            f"{name:<6} {seq_len:>5} {batch_size:>5}  OOM -- skipped "
                            "(lower batch size or sequence length)"
                        )
                        results.append(
                            {
                                "model": name,
                                "config": config_path,
                                "seq_len": seq_len,
                                "batch_size": batch_size,
                                "error": "OOM",
                            }
                        )
                    finally:
                        del x, y
                        gc.collect()
                        torch.cuda.empty_cache()
        finally:
            del model, optimizer
            gc.collect()
            torch.cuda.empty_cache()

    by_key = {(row["model"], row["seq_len"], row["batch_size"]): row for row in results}
    print("\nSpeedup (fast / dense, same seq+batch):")
    print(f"{'seq':>5} {'batch':>5} {'dense tok/s':>13} {'fast tok/s':>13} {'speedup':>9}")
    for seq_len in seq_lens:
        for batch_size in batch_sizes:
            dense = by_key.get(("dense", seq_len, batch_size))
            fast = by_key.get(("fast", seq_len, batch_size))
            if not dense or not fast or dense.get("error") or fast.get("error"):
                continue
            speedup = fast["tok_per_sec"] / dense["tok_per_sec"]
            speedups.append(
                {
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "dense_tok_per_sec": dense["tok_per_sec"],
                    "fast_tok_per_sec": fast["tok_per_sec"],
                    "speedup": speedup,
                }
            )
            print(
                f"{seq_len:>5} {batch_size:>5} {dense['tok_per_sec']:>13,.0f} "
                f"{fast['tok_per_sec']:>13,.0f} {speedup:>8.2f}x"
            )

    payload = {
        "meta": {
            "gpu": gpu_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "git_commit": git_commit(),
            "dtype": str(amp_dtype),
            "compile": args.compile,
            "compile_mode": args.compile_mode,
            "vocab_size": vocab_size,
            "warmup": args.warmup,
            "iters": args.iters,
            "dense_config": args.dense,
            "fast_config": args.fast,
        },
        "results": results,
        "speedups": speedups,
    }
    save_json(args.out, payload)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
