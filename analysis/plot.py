#!/usr/bin/env python3
"""Plot dense vs fast training curves from metrics.csv files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def read_metrics(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            try:
                step = int(raw["step"])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "step": step,
                    "tokens": _as_int(raw.get("tokens")),
                    "lm_loss": _as_float(raw.get("lm_loss")),
                    "val_loss": _as_float(raw.get("val_loss")),
                    "tok_per_sec": _as_float(raw.get("tok_per_sec")),
                    "elapsed_s": _as_float(raw.get("elapsed_s")),
                }
            )
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _as_float(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _as_int(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def smooth(values: list[float], alpha: float) -> list[float]:
    result = []
    last = None
    for value in values:
        last = value if last is None else alpha * last + (1.0 - alpha) * value
        result.append(last)
    return result


def series(rows: list[dict], key: str):
    points = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        x = row.get("tokens")
        if x is None:
            x = row["step"]
        points.append((x, value))
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", default=None, help="dense run metrics.csv")
    parser.add_argument("--fast", default=None, help="fast run metrics.csv")
    parser.add_argument("--out", default="runs/compare.png")
    parser.add_argument("--smooth", type=float, default=0.9)
    parser.add_argument("--title", default="LiteMoD vs dense on TinyStories")
    args = parser.parse_args()

    runs = []
    if args.dense:
        runs.append(("dense", read_metrics(args.dense), "#4c72b0"))
    if args.fast:
        runs.append(("fast (MoD)", read_metrics(args.fast), "#dd8452"))
    if not runs:
        raise SystemExit("pass --dense and/or --fast")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    for label, rows, color in runs:
        points = series(rows, "lm_loss")
        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            axes[0].plot(xs, ys, color=color, alpha=0.25, linewidth=1.0)
            axes[0].plot(xs, smooth(ys, args.smooth), color=color, linewidth=2.0, label=label)

        val_points = series(rows, "val_loss")
        if val_points:
            axes[1].plot(
                [p[0] for p in val_points],
                [p[1] for p in val_points],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.5,
                label=label,
            )

        speed_points = series(rows, "tok_per_sec")
        if speed_points:
            axes[2].plot(
                [p[0] for p in speed_points],
                [p[1] for p in speed_points],
                color=color,
                linewidth=1.8,
                label=label,
            )

    axes[0].set_title("Train loss (smoothed)")
    axes[0].set_xlabel("tokens seen")
    axes[0].set_ylabel("cross-entropy")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Validation loss")
    axes[1].set_xlabel("tokens seen")
    axes[1].set_ylabel("cross-entropy")
    axes[1].grid(alpha=0.25)

    axes[2].set_title("Training throughput")
    axes[2].set_xlabel("tokens seen")
    axes[2].set_ylabel("tokens / second")
    axes[2].grid(alpha=0.25)

    for axis in axes:
        axis.legend(frameon=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    fig.suptitle(args.title)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
