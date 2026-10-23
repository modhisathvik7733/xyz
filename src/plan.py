"""Budget and throughput calculator for a 1B-class long-context model.

This is deliberately architecture-agnostic arithmetic, not a training run.  It
answers: at a given GPU price, MFU, active parameter count, attention pattern,
and token budget, how many GPU-hours and dollars are needed for 200B tokens,
and what aggregate tokens/sec is required to fit a fixed budget?

The default scenarios compare:
  * dense 1B with full attention
  * 100M-active MoE with full attention
  * 100M-active MoE with 4k local attention / hybrid linear attention
  * 120M/150M-active variants
  * a 256k extension on 1% of tokens

Assumptions are printed with the results.  Treat these as planning numbers,
then calibrate with a 1B microbenchmark on the actual GPU.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

# Peak dense BF16/FP16 tensor throughput, not sparse.
# Prices are the "from" Vast.ai list prices observed Oct 2026:
#   https://vast.ai/pricing/gpu/RTX-4090
#   https://vast.ai/pricing/gpu/A100-SXM4
#   https://vast.ai/pricing/gpu/H100-SXM
GPU_PRESETS = {
    "4090": {"peak_tflops": 165.0, "price": 0.13},
    "a100": {"peak_tflops": 156.0, "price": 0.31},
    "h100": {"peak_tflops": 495.0, "price": 1.60},
}


@dataclass(frozen=True)
class Scenario:
    name: str
    active_params: float
    attn_mode: str                  # full | window | hybrid
    train_seq: Optional[int] = None
    long_seq: Optional[int] = None
    long_fraction: float = 0.0


@dataclass
class Result:
    name: str
    active_params: float
    attn_mode: str
    flops_per_token: float
    total_flops: float
    gpu_hours: float
    wall_hours: float
    cost: float
    aggregate_tok_per_sec: float
    per_gpu_tok_per_sec: float
    within_budget: bool


def attention_flops(
    seq_len: int,
    d_model: int,
    layers: int,
    window: int,
    mode: str,
    full_layer_fraction: float,
    linear_cost_fraction: float,
) -> float:
    """Training attention FLOPs per token, averaged over the sequence.

    Full attention: one query/key dot product and one value mix per pair.
    Forward is 4 * T * d per layer; training is ~3x that, hence 12 * T * d.
    """
    if mode == "full":
        return 12.0 * seq_len * d_model * layers
    windowed = 12.0 * min(seq_len, window) * d_model * layers
    if mode == "window":
        return windowed
    if mode == "hybrid":
        return windowed * (
            full_layer_fraction + (1.0 - full_layer_fraction) * linear_cost_fraction
        )
    raise ValueError(f"unknown attention mode {mode!r}")


def evaluate(
    scenario: Scenario,
    args: argparse.Namespace,
) -> Result:
    train_seq = scenario.train_seq or args.train_seq
    long_seq = scenario.long_seq or args.long_seq
    frac = scenario.long_fraction

    model_flops = 6.0 * scenario.active_params
    attn_short = attention_flops(
        train_seq, args.d_model, args.layers, args.window,
        scenario.attn_mode, args.full_layer_fraction, args.linear_cost_fraction,
    )
    attn_long = attention_flops(
        long_seq, args.d_model, args.layers, args.window,
        scenario.attn_mode, args.full_layer_fraction, args.linear_cost_fraction,
    )
    flops_per_token = model_flops + (1.0 - frac) * attn_short + frac * attn_long
    total_flops = flops_per_token * args.tokens

    effective_tflops = args.peak_tflops * args.mfu
    gpu_hours = total_flops / (effective_tflops * 1e12 * 3600.0)
    wall_hours = gpu_hours / args.gpus
    cost = gpu_hours * args.price
    aggregate_tok_s = args.gpus * effective_tflops * 1e12 / flops_per_token
    per_gpu_tok_s = effective_tflops * 1e12 / flops_per_token

    return Result(
        name=scenario.name,
        active_params=scenario.active_params,
        attn_mode=scenario.attn_mode,
        flops_per_token=flops_per_token,
        total_flops=total_flops,
        gpu_hours=gpu_hours,
        wall_hours=wall_hours,
        cost=cost,
        aggregate_tok_per_sec=aggregate_tok_s,
        per_gpu_tok_per_sec=per_gpu_tok_s,
        within_budget=cost <= args.budget,
    )


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    return [
        Scenario("dense-1b / full 8k", args.total_params, "full", args.train_seq, args.long_seq, 0.0),
        Scenario("dense-1b / full 256k", args.total_params, "full", args.long_seq, args.long_seq, 0.0),
        Scenario("moe100 / full 8k", 100e6, "full", args.train_seq, args.long_seq, 0.0),
        Scenario("moe100 / local 4k", 100e6, "window", args.train_seq, args.long_seq, 0.0),
        Scenario("moe100 / hybrid 8k+256k 1%", 100e6, "hybrid", args.train_seq, args.long_seq, args.long_fraction),
        Scenario("moe120 / hybrid 8k+256k 1%", 120e6, "hybrid", args.train_seq, args.long_seq, args.long_fraction),
        Scenario("moe150 / hybrid 8k+256k 1%", 150e6, "hybrid", args.train_seq, args.long_seq, args.long_fraction),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--budget", type=float, default=150.0)
    p.add_argument("--gpu", choices=sorted(GPU_PRESETS), default="4090")
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--price", type=float, default=None, help="override $/GPU-hour")
    p.add_argument("--peak-tflops", type=float, default=None, help="override dense BF16 peak/GPU")
    p.add_argument("--mfu", type=float, default=0.35)
    p.add_argument("--tokens", type=float, default=200e9)
    p.add_argument("--total-params", type=float, default=1.0e9)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--d-model", type=int, default=1536)
    p.add_argument("--train-seq", type=int, default=8192)
    p.add_argument("--long-seq", type=int, default=262144)
    p.add_argument("--long-fraction", type=float, default=0.01)
    p.add_argument("--window", type=int, default=4096)
    p.add_argument("--full-layer-fraction", type=float, default=0.25)
    p.add_argument("--linear-cost-fraction", type=float, default=0.10)
    p.add_argument("--json", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    preset = GPU_PRESETS[args.gpu]
    if args.price is None:
        args.price = preset["price"]
    if args.peak_tflops is None:
        args.peak_tflops = preset["peak_tflops"]
    if args.gpus < 1:
        raise SystemExit("--gpus must be >= 1")
    if not (0.0 < args.mfu <= 1.0):
        raise SystemExit("--mfu must be in (0, 1]")
    if args.tokens <= 0 or args.total_params <= 0:
        raise SystemExit("tokens and total params must be positive")

    budget_hours = args.budget / args.price
    effective_tflops = args.peak_tflops * args.mfu
    required_agg = (
        args.tokens / ((budget_hours / args.gpus) * 3600.0)
        if budget_hours > 0
        else float("inf")
    )

    print("=" * 112)
    print("LiteMoD scaling / budget calculator")
    print(
        f"target: {args.total_params/1e9:.2f}B total | {args.tokens/1e9:.1f}B tokens | "
        f"{args.train_seq:,} main ctx + {args.long_seq:,} on {args.long_fraction*100:.1f}% of tokens"
    )
    print(
        f"hardware: {args.gpus}x {args.gpu.upper()} @ ${args.price:.2f}/GPU-hour | "
        f"{args.mfu*100:.0f}% MFU | {effective_tflops:.1f} effective TFLOP/s/GPU"
    )
    print(
        f"budget: ${args.budget:,.0f} -> {budget_hours:,.1f} GPU-hours = "
        f"{budget_hours/args.gpus:,.1f} wall-hours on {args.gpus} GPUs"
    )
    print(
        f"required aggregate throughput for the budget: {required_agg:,.0f} tok/s "
        f"({required_agg/args.gpus:,.0f}/GPU)"
    )
    print("=" * 112)

    header = (
        f"{'scenario':<34} {'active':>8} {'attn':>7} {'GFLOPs/tok':>11} "
        f"{'EFLOPs':>8} {'GPU-h':>9} {'wall-h':>8} {'cost':>9} "
        f"{'agg tok/s':>11} {'/GPU':>9} {'verdict':>8}"
    )
    print(header)
    print("-" * len(header))

    results: list[dict] = []
    for scenario in build_scenarios(args):
        result = evaluate(scenario, args)
        results.append(asdict(result))
        print(
            f"{result.name:<34} {result.active_params/1e6:>7.0f}M "
            f"{result.attn_mode:>7} {result.flops_per_token/1e9:>11.2f} "
            f"{result.total_flops/1e18:>8.1f} {result.gpu_hours:>9,.0f} "
            f"{result.wall_hours:>8,.1f} ${result.cost:>8,.0f} "
            f"{result.aggregate_tok_per_sec:>11,.0f} "
            f"{result.per_gpu_tok_per_sec:>9,.0f} "
            f"{'FIT' if result.within_budget else 'OVER':>8}"
        )

    print("-" * len(header))
    print(
        f"assumptions: model FLOPs/token = 6 x active params; attention training FLOPs = "
        f"12 x effective-context x d_model x layers; hybrid = "
        f"{args.full_layer_fraction*100:.0f}% windowed layers + "
        f"{(1-args.full_layer_fraction)*100:.0f}% linear layers at "
        f"{args.linear_cost_fraction*100:.0f}% of windowed cost."
    )
    print(
        "The 256k rows are for the stated fraction only. Full 256k attention on every token "
        "is not a $150 workload at 1B/200B."
    )
    print(
        "These are paper estimates. Calibrate with a 1B microbenchmark and actual MFU before renting."
    )

    if args.json:
        payload = {"args": vars(args), "budget_hours": budget_hours,
                   "required_aggregate_tok_per_sec": required_agg, "results": results}
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
