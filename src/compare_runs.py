"""Compare two training runs from their metrics.csv / run_config.json files.

Designed for the 1200-step dense-vs-fast decision run:

    python -m src.compare_runs --dense runs/dense_1200 --fast runs/fast_1200

It reports final quality, throughput, speedup, and validation loss at common
wall-clock times (validation pauses excluded from elapsed_s).  It does not
import torch, so it can also be run on a laptop against downloaded CSVs.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class RunSummary:
    name: str
    path: Path
    steps: int = 0
    tokens: int = 0
    wall_s: float = 0.0
    median_tok_s: Optional[float] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    best_val_loss: Optional[float] = None
    peak_gb: Optional[float] = None
    params: Optional[int] = None
    gpu: str = "?"
    arch: str = "?"
    val_points: list[tuple[float, float]] = field(default_factory=list)


def _as_float(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def load_run(name: str, path: str | Path) -> RunSummary:
    path = Path(path)
    metrics_path = path / "metrics.csv"
    if not metrics_path.exists():
        raise SystemExit(f"missing {metrics_path}")

    config = {}
    config_path = path / "run_config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)

    summary = RunSummary(name=name, path=path)
    summary.params = config.get("params")
    summary.gpu = str(config.get("gpu", "?"))
    summary.arch = str(config.get("config", {}).get("model", {}).get("arch", "?"))

    speeds = []
    val_points = []
    with open(metrics_path, "r", encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle):
            step = _as_int(raw.get("step"))
            if step is None:
                continue
            summary.steps = max(summary.steps, step)
            tokens = _as_int(raw.get("tokens"))
            if tokens is not None:
                summary.tokens = max(summary.tokens, tokens)
            elapsed = _as_float(raw.get("elapsed_s"))
            if elapsed is not None:
                summary.wall_s = max(summary.wall_s, elapsed)
            tok_s = _as_float(raw.get("tok_per_sec"))
            if tok_s is not None and tok_s > 0:
                speeds.append((step, tok_s))
            lm_loss = _as_float(raw.get("lm_loss"))
            if lm_loss is not None:
                summary.final_train_loss = lm_loss
            val_loss = _as_float(raw.get("val_loss"))
            if val_loss is not None and elapsed is not None:
                val_points.append((elapsed, val_loss))
                summary.final_val_loss = val_loss
                summary.best_val_loss = (
                    val_loss if summary.best_val_loss is None
                    else min(summary.best_val_loss, val_loss)
                )
            peak = _as_float(raw.get("peak_mem_gb"))
            if peak is not None:
                summary.peak_gb = peak if summary.peak_gb is None else max(summary.peak_gb, peak)

    # Exclude the first 20% of steps (compile + warmup) from the median.
    cutoff = summary.steps * 0.2
    steady = [value for step, value in speeds if step >= cutoff]
    if not steady:
        steady = [value for _, value in speeds]
    if steady:
        summary.median_tok_s = statistics.median(steady)

    summary.val_points = sorted(val_points)
    return summary


def interp(points: list[tuple[float, float]], x: float) -> Optional[float]:
    if not points:
        return None
    xs = [point[0] for point in points]
    if x <= xs[0]:
        return points[0][1]
    if x >= xs[-1]:
        return points[-1][1]
    index = bisect.bisect_right(xs, x)
    x0, y0 = points[index - 1]
    x1, y1 = points[index]
    if x1 == x0:
        return y1
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def fmt(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{value:,.{digits}f}{suffix}"


def fmt_params(params: Optional[int]) -> str:
    if params is None:
        return "n/a"
    return f"{params / 1e6:,.2f}M"


def build_markdown(dense: RunSummary, fast: RunSummary, common_times: list[float]) -> str:
    speedup = (
        fast.median_tok_s / dense.median_tok_s
        if fast.median_tok_s and dense.median_tok_s
        else None
    )
    loss_delta = (
        fast.final_val_loss - dense.final_val_loss
        if fast.final_val_loss is not None and dense.final_val_loss is not None
        else None
    )

    lines = [
        "# LiteMoD 1200-step decision report",
        "",
        f"- dense: `{dense.path}` ({dense.gpu})",
        f"- fast: `{fast.path}` ({fast.gpu})",
        f"- dense arch: `{dense.arch}` | fast arch: `{fast.arch}`",
        "",
        "## Headline",
        "",
        "| metric | dense | fast | fast / dense |",
        "|---|---:|---:|---:|",
        f"| steps | {dense.steps:,} | {fast.steps:,} | |",
        f"| tokens | {dense.tokens:,} | {fast.tokens:,} | |",
        f"| median tok/s | {fmt(dense.median_tok_s, 0)} | {fmt(fast.median_tok_s, 0)} | "
        f"{fmt(speedup, 2, 'x')} |",
        f"| training wall time | {fmt(dense.wall_s / 60, 1, ' min')} | "
        f"{fmt(fast.wall_s / 60, 1, ' min')} | |",
        f"| params | {fmt_params(dense.params)} | {fmt_params(fast.params)} | |",
        f"| final train loss | {fmt(dense.final_train_loss, 4)} | "
        f"{fmt(fast.final_train_loss, 4)} | |",
        f"| best val loss | {fmt(dense.best_val_loss, 4)} | "
        f"{fmt(fast.best_val_loss, 4)} | |",
        f"| final val loss | {fmt(dense.final_val_loss, 4)} | "
        f"{fmt(fast.final_val_loss, 4)} | "
        f"{fmt(loss_delta, 4) if loss_delta is not None else 'n/a'} |",
        f"| peak VRAM | {fmt(dense.peak_gb, 2, ' GB')} | "
        f"{fmt(fast.peak_gb, 2, ' GB')} | |",
        "",
    ]

    if loss_delta is not None:
        if loss_delta <= 0:
            quality = "fast is better on final validation loss"
        elif loss_delta <= 0.10:
            quality = "fast is within 0.10 nats of dense"
        elif loss_delta <= 0.25:
            quality = "fast has a moderate quality gap"
        else:
            quality = "fast has a large quality gap"
        lines.append(f"**At equal tokens:** {quality} ({loss_delta:+.4f} nats).")
        lines.append("")

    if common_times:
        lines += [
            "## Validation loss at equal training wall-clock",
            "",
            "_Validation pauses are excluded; both runs use the same 32,768 tokens/step._",
            "",
            "| common time | dense val | fast val | fast - dense |",
            "|---:|---:|---:|---:|",
        ]
        for seconds in common_times:
            dense_val = interp(dense.val_points, seconds)
            fast_val = interp(fast.val_points, seconds)
            delta = (
                fast_val - dense_val
                if fast_val is not None and dense_val is not None
                else None
            )
            lines.append(
                f"| {seconds/60:.1f} min | {fmt(dense_val, 4)} | {fmt(fast_val, 4)} | "
                f"{fmt(delta, 4) if delta is not None else 'n/a'} |"
            )
        lines.append("")

    lines += [
        "## Verdict",
        "",
        "The 1200-step run is a **speed/quality triage**, not a final model-quality claim.",
        "Use it to decide whether to scale the fast architecture, not to estimate a 1B model's final loss.",
        "",
    ]
    if speedup is None:
        lines.append("- Missing throughput data; rerun with metrics.csv containing `tok_per_sec`.")
    elif speedup < 1.05:
        lines.append("- Speedup is marginal (<1.05x). Do not scale this configuration yet.")
    elif loss_delta is not None and loss_delta > 0.25:
        lines.append("- Speed is real but the quality gap is large. Raise `mod_capacity` (0.6-0.7) and retest.")
    elif loss_delta is not None and loss_delta > 0.10:
        lines.append("- Speed is useful; quality gap is moderate. Try capacity 0.6 before scaling.")
    else:
        lines.append("- Speed/quality trade-off looks usable for a scale-up experiment.")

    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", required=True, help="directory containing dense metrics.csv")
    parser.add_argument("--fast", required=True, help="directory containing fast metrics.csv")
    parser.add_argument("--out", default="runs/compare_1200.md")
    parser.add_argument("--json", default="runs/compare_1200.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dense = load_run("dense", args.dense)
    fast = load_run("fast", args.fast)

    common_max = min(dense.wall_s, fast.wall_s)
    common_times = []
    if common_max > 0:
        for fraction in (0.25, 0.50, 0.75, 1.00):
            common_times.append(common_max * fraction)

    markdown = build_markdown(dense, fast, common_times)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"wrote {out_path}")

    payload = {
        "dense": {
            "path": str(dense.path),
            "steps": dense.steps,
            "tokens": dense.tokens,
            "wall_s": dense.wall_s,
            "median_tok_s": dense.median_tok_s,
            "final_train_loss": dense.final_train_loss,
            "best_val_loss": dense.best_val_loss,
            "final_val_loss": dense.final_val_loss,
            "params": dense.params,
            "peak_gb": dense.peak_gb,
        },
        "fast": {
            "path": str(fast.path),
            "steps": fast.steps,
            "tokens": fast.tokens,
            "wall_s": fast.wall_s,
            "median_tok_s": fast.median_tok_s,
            "final_train_loss": fast.final_train_loss,
            "best_val_loss": fast.best_val_loss,
            "final_val_loss": fast.final_val_loss,
            "params": fast.params,
            "peak_gb": fast.peak_gb,
        },
        "speedup": (
            fast.median_tok_s / dense.median_tok_s
            if fast.median_tok_s and dense.median_tok_s
            else None
        ),
        "final_val_delta": (
            fast.final_val_loss - dense.final_val_loss
            if fast.final_val_loss is not None and dense.final_val_loss is not None
            else None
        ),
        "common_times_s": common_times,
        "equal_time_val": [
            {
                "seconds": seconds,
                "dense": interp(dense.val_points, seconds),
                "fast": interp(fast.val_points, seconds),
            }
            for seconds in common_times
        ],
    }
    json_path = Path(args.json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
