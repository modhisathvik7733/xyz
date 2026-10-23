"""Train one LiteMoD / dense model on packed TinyStories tokens."""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch

from .data import PackedTokens
from .flops import estimate
from .model import LiteMoD, count_parameters, model_config_from_dict
from .utils import (
    apply_overrides,
    autocast_context,
    configure_torch,
    format_int,
    get_lr,
    git_commit,
    load_json,
    load_yaml,
    make_grad_scaler,
    save_json,
    set_seed,
    unwrap_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fast_tiny.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument(
        "--compile-mode",
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--resume", default=None, help="checkpoint path, or 'auto'")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    cfg: dict,
    meta: dict,
) -> None:
    payload = {
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": cfg,
        "meta": meta,
    }
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: PackedTokens,
    batch_size: int,
    iters: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    amp_enabled: bool,
    seed: int,
) -> float:
    model.eval()
    losses = []
    for index in range(max(1, iters)):
        x, y = data.get_batch(batch_size, index, device, seed=seed)
        with autocast_context(device, amp_dtype, amp_enabled):
            _, lm_loss, _ = model(x, y)
        if lm_loss is None:
            raise RuntimeError("model did not return a language-modeling loss")
        losses.append(float(lm_loss.detach()))
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    apply_overrides(cfg, args.set)
    if args.compile:
        cfg.setdefault("train", {})["compile"] = True

    model_cfg = model_config_from_dict(cfg.get("model", {}))
    train_cfg = cfg.get("train", {})

    data_dir = Path(train_cfg.get("data_dir", "data"))
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"{meta_path} not found. Run `python -m src.prepare_data` first."
        )
    meta = load_json(meta_path)
    model_cfg.vocab_size = int(meta["vocab_size"])
    cfg.setdefault("model", {})["vocab_size"] = model_cfg.vocab_size

    out_dir = Path(train_cfg.get("out_dir", f"runs/{model_cfg.arch}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 1337))
    batch_size = int(train_cfg.get("batch_size", 32))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    if batch_size < 1 or grad_accum < 1:
        raise SystemExit("batch_size and grad_accum must both be >= 1")
    block_size = int(model_cfg.block_size)
    tokens_per_step = batch_size * grad_accum * block_size
    max_steps = int(args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", 4000))
    if max_steps < 1:
        raise SystemExit("max_steps must be >= 1")

    max_tokens = train_cfg.get("max_tokens")
    if max_tokens:
        max_steps = min(max_steps, math.ceil(int(max_tokens) / tokens_per_step))
    if max_steps < 1:
        raise SystemExit("max_steps/max_tokens must leave at least one training step")
    max_minutes = args.max_minutes if args.max_minutes is not None else train_cfg.get("max_minutes")
    max_minutes = float(max_minutes) if max_minutes else None

    lr = float(train_cfg.get("lr", 5.0e-4))
    min_lr_ratio = float(train_cfg.get("min_lr_ratio", 0.1))
    min_lr = lr * min_lr_ratio
    warmup_steps = int(train_cfg.get("warmup_steps", 200))
    weight_decay = float(train_cfg.get("weight_decay", 0.1))
    betas = tuple(float(value) for value in train_cfg.get("betas", [0.9, 0.95]))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    log_interval = int(train_cfg.get("log_interval", 10))
    eval_interval = int(train_cfg.get("eval_interval", 250))
    eval_iters = int(train_cfg.get("eval_iters", 50))
    save_interval = int(train_cfg.get("save_interval", 500))
    val_seed = int(train_cfg.get("val_seed", 12345))
    compile_model = bool(train_cfg.get("compile", False))
    dtype_name = str(train_cfg.get("dtype", "bf16")).lower()

    set_seed(seed)
    configure_torch()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        if dtype_name == "bf16":
            if not torch.cuda.is_bf16_supported():
                raise SystemExit("this GPU does not support bf16; use dtype: fp16")
            amp_dtype = torch.bfloat16
        elif dtype_name == "fp16":
            amp_dtype = torch.float16
        elif dtype_name == "fp32":
            amp_dtype = torch.float32
        else:
            raise SystemExit(f"unknown dtype {dtype_name!r}")
        amp_enabled = amp_dtype != torch.float32
        gpu_name = torch.cuda.get_device_name(0)
    else:
        amp_dtype = torch.float32
        amp_enabled = False
        gpu_name = "CPU (no CUDA -- speed numbers will be meaningless)"

    print("=" * 78)
    print(f"LiteMoD training | {model_cfg.arch} | {args.config}")
    print(f"device: {device} | gpu: {gpu_name}")
    print(f"torch: {torch.__version__} | amp: {amp_dtype} | compile: {compile_model}")
    print(f"data: {data_dir} | vocab: {model_cfg.vocab_size:,}")
    print(
        f"batch: {batch_size} x grad_accum {grad_accum} x block {block_size} "
        f"= {format_int(tokens_per_step)} tokens/step"
    )
    print(f"max_steps: {max_steps:,} -> {format_int(max_steps * tokens_per_step)} tokens")
    if max_minutes:
        print(f"wall-clock limit: {max_minutes:.1f} minutes")
    print("=" * 78)

    train_data = PackedTokens(data_dir / "train.bin", block_size)
    val_data = PackedTokens(data_dir / "val.bin", block_size)
    print(f"train tokens: {format_int(len(train_data))} | val tokens: {format_int(len(val_data))}")

    model = LiteMoD(model_cfg).to(device)
    param_groups = [
        {
            "params": [
                param
                for _, param in unwrap_model(model).named_parameters()
                if param.requires_grad and param.dim() >= 2
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                param
                for _, param in unwrap_model(model).named_parameters()
                if param.requires_grad and param.dim() < 2
            ],
            "weight_decay": 0.0,
        },
    ]
    try:
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=betas,
            fused=device.type == "cuda",
        )
    except (TypeError, RuntimeError):
        optimizer = torch.optim.AdamW(param_groups, lr=lr, betas=betas)

    start_step = 0
    resume_path = args.resume or train_cfg.get("resume")
    if resume_path:
        path = out_dir / "latest.pt" if resume_path == "auto" else Path(resume_path)
        if not path.exists():
            raise SystemExit(f"resume checkpoint not found: {path}")
        checkpoint = load_checkpoint(path, device)
        unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"]) + 1
        print(f"resumed from {path} at step {start_step:,}")

    if compile_model:
        if not hasattr(torch, "compile"):
            raise SystemExit("torch.compile is unavailable in this PyTorch build")
        print(f"compiling model (mode={args.compile_mode}) ...")
        model = torch.compile(model, mode=args.compile_mode)

    params = count_parameters(unwrap_model(model))
    flops = estimate(model_cfg, block_size)
    print(
        f"params: {params/1e6:.2f}M | effective depth: {flops['effective_depth']:.2f}/{model_cfg.n_layer} "
        f"| {flops['training_flops_per_token']/1e6:.1f} est. training MFLOPs/token"
    )
    print(
        f"blocks: {flops['dense_layers']} dense + {flops['mod_layers']} MoD "
        f"(capacity={model_cfg.mod_capacity if model_cfg.arch == 'mod' else 1.0})"
    )

    if start_step >= max_steps:
        print(f"checkpoint is already at step {start_step:,} >= max_steps {max_steps:,}; nothing to do")
        return

    run_config = {
        "config": cfg,
        "meta": meta,
        "tokens_per_step": tokens_per_step,
        "max_steps": max_steps,
        "seed": seed,
        "git_commit": git_commit(),
        "torch": torch.__version__,
        "gpu": gpu_name,
    }
    save_json(out_dir / "run_config.json", run_config)

    csv_path = out_dir / "metrics.csv"
    append = csv_path.exists() and start_step > 0
    fieldnames = [
        "step",
        "tokens",
        "lm_loss",
        "aux_loss",
        "total_loss",
        "lr",
        "grad_norm",
        "tok_per_sec",
        "elapsed_s",
        "peak_mem_gb",
        "val_loss",
    ]

    scaler = make_grad_scaler(amp_dtype == torch.float16)
    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    last_completed = start_step - 1
    train_start = time.perf_counter()
    last_log_time = train_start
    last_log_step = start_step
    latest_ckpt = out_dir / "latest.pt"

    def maybe_validate(step: int) -> Optional[float]:
        if eval_interval <= 0 or eval_iters <= 0:
            return None
        value = evaluate(
            model,
            val_data,
            batch_size,
            eval_iters,
            device,
            amp_dtype,
            amp_enabled,
            val_seed,
        )
        print(f"  val @ step {step + 1:>6}: {value:.4f}")
        return value

    with open(csv_path, "a" if append else "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not append:
            writer.writeheader()

        initial_val = maybe_validate(start_step - 1) if start_step == 0 else None
        if initial_val is not None:
            writer.writerow(
                {
                    "step": 0,
                    "tokens": 0,
                    "lm_loss": "",
                    "aux_loss": "",
                    "total_loss": "",
                    "lr": "",
                    "grad_norm": "",
                    "tok_per_sec": "",
                    "elapsed_s": 0.0,
                    "peak_mem_gb": "",
                    "val_loss": f"{initial_val:.6f}",
                }
            )
            csv_file.flush()

        # Start the throughput clock after any initial validation pass.
        train_start = time.perf_counter()
        last_log_time = train_start

        try:
            for step in range(start_step, max_steps):
                if max_minutes is not None:
                    elapsed = time.perf_counter() - train_start
                    if elapsed >= max_minutes * 60.0:
                        print(f"wall-clock limit reached after {step - start_step:,} steps")
                        break

                lr_now = get_lr(step, max_steps, warmup_steps, lr, min_lr)
                for group in optimizer.param_groups:
                    group["lr"] = lr_now

                optimizer.zero_grad(set_to_none=True)
                lm_accum = 0.0
                aux_accum = 0.0
                total_accum = 0.0

                for micro in range(grad_accum):
                    x, y = train_data.get_batch(
                        batch_size,
                        step * grad_accum + micro,
                        device,
                        seed=seed,
                    )
                    with autocast_context(device, amp_dtype, amp_enabled):
                        _, lm_loss, aux_loss = model(x, y)
                        total_loss = lm_loss + model_cfg.mod_aux_weight * aux_loss
                        scaled_loss = total_loss / grad_accum

                    if scaler is not None:
                        scaler.scale(scaled_loss).backward()
                    else:
                        scaled_loss.backward()

                    lm_accum += float(lm_loss.detach())
                    aux_accum += float(aux_loss.detach())
                    total_accum += float(total_loss.detach())

                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    unwrap_model(model).parameters(), grad_clip
                )
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                last_completed = step
                is_final = step == max_steps - 1
                should_log = (step + 1) % max(1, log_interval) == 0 or is_final
                should_eval = eval_interval > 0 and ((step + 1) % eval_interval == 0 or is_final)

                if should_log or should_eval:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    now = time.perf_counter()
                    elapsed = now - train_start
                    steps_since_log = step + 1 - last_log_step
                    tok_per_sec = (
                        steps_since_log * tokens_per_step / (now - last_log_time)
                        if now > last_log_time
                        else 0.0
                    )
                    peak_mem = (
                        torch.cuda.max_memory_allocated() / 1e9
                        if device.type == "cuda"
                        else 0.0
                    )
                    val_loss = maybe_validate(step) if should_eval else None
                    row = {
                        "step": step + 1,
                        "tokens": (step + 1) * tokens_per_step,
                        "lm_loss": f"{lm_accum / grad_accum:.6f}",
                        "aux_loss": f"{aux_accum / grad_accum:.6f}",
                        "total_loss": f"{total_accum / grad_accum:.6f}",
                        "lr": f"{lr_now:.8f}",
                        "grad_norm": f"{float(grad_norm):.6f}",
                        "tok_per_sec": f"{tok_per_sec:.2f}",
                        "elapsed_s": f"{elapsed:.3f}",
                        "peak_mem_gb": f"{peak_mem:.3f}",
                        "val_loss": "" if val_loss is None else f"{val_loss:.6f}",
                    }
                    writer.writerow(row)
                    csv_file.flush()
                    # Do not count validation time as training throughput.
                    last_log_time = time.perf_counter() if should_eval else now
                    last_log_step = step + 1

                    print(
                        f"step {step + 1:>6}/{max_steps} | "
                        f"loss {lm_accum / grad_accum:.4f} | "
                        f"aux {aux_accum / grad_accum:.4f} | "
                        f"lr {lr_now:.2e} | "
                        f"{tok_per_sec:,.0f} tok/s | "
                        f"{peak_mem:.2f} GB"
                    )

                if save_interval > 0 and (step + 1) % save_interval == 0:
                    save_checkpoint(latest_ckpt, model, optimizer, step, cfg, meta)
                    print(f"  saved {latest_ckpt}")

        except KeyboardInterrupt:
            print("\ninterrupted; saving checkpoint ...")
            if last_completed >= start_step:
                save_checkpoint(latest_ckpt, model, optimizer, last_completed, cfg, meta)
            raise

    if last_completed >= start_step:
        save_checkpoint(latest_ckpt, model, optimizer, last_completed, cfg, meta)
        save_checkpoint(out_dir / "final.pt", model, optimizer, last_completed, cfg, meta)
        print(
            f"done: {last_completed + 1:,} steps | "
            f"{format_int((last_completed + 1) * tokens_per_step)} tokens | "
            f"artifacts in {out_dir}"
        )
    else:
        print("no training steps completed")


if __name__ == "__main__":
    main()
