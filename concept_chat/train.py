"""Train and evaluate ConceptChat on the synthetic baby-English corpus.

Example:
  python -m concept_chat.train --config concept_chat/configs/small.yaml --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from .data import BOS, EOS, PAD, Corpus, Example, build_corpus, decode, encode
from .model import ConceptSeq2Seq, count_parameters

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS requested but unavailable")
        return torch.device("mps")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but unavailable")
        return torch.device("cuda")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_batch(
    examples: list[Example],
    corpus: Corpus,
    concept_to_id: dict[str, int],
    device: torch.device,
):
    src_rows = []
    in_rows = []
    out_rows = []
    concepts = []
    max_dec = corpus.max_tgt_len
    for example in examples:
        src_rows.append(encode(example.user, corpus.vocab, corpus.max_src_len))
        reply = encode(example.reply, corpus.vocab, max_dec - 1)
        in_rows.append(([corpus.vocab[BOS]] + reply)[:max_dec])
        out_rows.append((reply + [corpus.vocab[EOS]])[:max_dec])
        concepts.append(concept_to_id[example.concept])

    def pad(rows):
        return [row + [0] * (corpus.max_src_len - len(row)) for row in rows]

    # dec rows need a common width; use the maximum in this batch.
    width = max(len(row) for row in in_rows)
    def pad_dec(rows):
        return [row + [0] * (width - len(row)) for row in rows]

    return (
        torch.tensor(pad(src_rows), dtype=torch.long, device=device),
        torch.tensor(pad_dec(in_rows), dtype=torch.long, device=device),
        torch.tensor(pad_dec(out_rows), dtype=torch.long, device=device),
        torch.tensor(concepts, dtype=torch.long, device=device),
    )


def token_f1(prediction: str, reference: str) -> float:
    pred = Counter(prediction.split())
    ref = Counter(reference.split())
    if not pred or not ref:
        return float(pred == ref)
    overlap = sum((pred & ref).values())
    precision = overlap / sum(pred.values())
    recall = overlap / sum(ref.values())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


@torch.no_grad()
def evaluate_split(
    model: ConceptSeq2Seq,
    examples: list[Example],
    corpus: Corpus,
    concept_to_id: dict[str, int],
    device: torch.device,
    *,
    batch_size: int = 128,
    oracle_concept: bool = False,
) -> dict:
    if not examples:
        return {"n": 0, "exact": 0.0, "f1": 0.0, "concept_acc": None, "per_concept": {}}
    model.eval()
    exact_hits = 0
    f1_total = 0.0
    concept_hits = 0
    concept_total = 0
    per_concept_hits: Counter = Counter()
    per_concept_total: Counter = Counter()

    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        src, dec_in, dec_out, concepts = make_batch(batch, corpus, concept_to_id, device)
        override = concepts if (oracle_concept and model.use_concept) else None
        generated, predicted = model.generate(
            src, corpus.max_tgt_len, concept_override=override
        )
        for row, example in enumerate(batch):
            pred_text = decode(generated[row], corpus.itos)
            exact_hits += int(pred_text == example.reply)
            f1_total += token_f1(pred_text, example.reply)
            per_concept_total[example.concept] += 1
            per_concept_hits[example.concept] += int(pred_text == example.reply)
            if predicted is not None:
                concept_hits += int(int(predicted[row]) == concept_to_id[example.concept])
                concept_total += 1
    return {
        "n": len(examples),
        "exact": exact_hits / len(examples),
        "f1": f1_total / len(examples),
        "concept_acc": (concept_hits / concept_total) if concept_total else None,
        "per_concept": {
            concept: {
                "n": per_concept_total[concept],
                "exact": per_concept_hits[concept] / per_concept_total[concept],
            }
            for concept in per_concept_total
        },
    }


def train_one(
    corpus: Corpus,
    concept_to_id: dict[str, int],
    device: torch.device,
    cfg: dict,
    variant: str,
    seed: int,
    out_dir: Path,
):
    set_seed(seed)
    use_concept = variant == "concept"
    model = ConceptSeq2Seq(
        vocab_size=len(corpus.vocab),
        num_concepts=len(corpus.concepts),
        emb=cfg["model"]["emb"],
        hidden=cfg["model"]["hidden"],
        dropout=cfg["model"]["dropout"],
        use_concept=use_concept,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )
    best_val = -1.0
    best_state = None
    best_epoch = -1
    patience_left = cfg["train"]["patience"]
    concept_counts = Counter(e.concept for e in corpus.train)
    if cfg["train"].get("balanced_concepts", True):
        total = sum(concept_counts.values())
        concept_weights = torch.tensor(
            [total / (len(corpus.concepts) * max(1, concept_counts.get(name, 0)))
             for name in corpus.concepts],
            dtype=torch.float32,
            device=device,
        )
    else:
        concept_weights = None

    train_examples = corpus.train
    if cfg["train"].get("balanced_sampling", True):
        max_count = max(concept_counts.values())
        train_examples = []
        for example in corpus.train:
            repeat = max(1, int(round((max_count / concept_counts[example.concept]) ** 0.5)))
            train_examples.extend([example] * min(repeat, 20))
    print(f"{variant:>8} seed={seed} train_pool={len(train_examples)}")
    history = []
    batch_size = cfg["train"]["batch_size"]
    start = time.perf_counter()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        order = list(range(len(train_examples)))
        random.shuffle(order)
        total_loss = 0.0
        total_batches = 0
        for offset in range(0, len(order), batch_size):
            batch = [train_examples[index] for index in order[offset : offset + batch_size]]
            src, dec_in, dec_out, concepts = make_batch(batch, corpus, concept_to_id, device)
            logits, concept_logits = model(src, dec_in, concepts if use_concept else None)
            loss = F.nll_loss(
                logits.reshape(-1, logits.shape[-1]), dec_out.reshape(-1), ignore_index=0
            )
            if use_concept and concept_logits is not None:
                loss = loss + cfg["train"]["concept_loss_weight"] * F.cross_entropy(
                    concept_logits, concepts, weight=concept_weights
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip"])
            optimizer.step()
            total_loss += float(loss.detach())
            total_batches += 1

        val = evaluate_split(
            model, corpus.val, corpus, concept_to_id, device,
            batch_size=cfg["train"]["eval_batch_size"],
        )
        train_loss = total_loss / max(1, total_batches)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_exact": val["exact"],
                        "val_f1": val["f1"], "val_concept_acc": val["concept_acc"]})
        print(
            f"{variant:>8} seed={seed} epoch={epoch:02d} loss={train_loss:.4f} "
            f"val_exact={val['exact']*100:5.1f}% val_f1={val['f1']*100:5.1f}% "
            f"concept={val['concept_acc']*100 if val['concept_acc'] is not None else -1:5.1f}%"
        )
        score = val["exact"]
        if score > best_val:
            best_val = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            patience_left = cfg["train"]["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"{variant:>8} seed={seed} early stop at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.perf_counter() - start
    checkpoint = {
        "model": best_state,
        "variant": variant,
        "seed": seed,
        "vocab": corpus.vocab,
        "itos": corpus.itos,
        "concepts": corpus.concepts,
        "max_src_len": corpus.max_src_len,
        "max_tgt_len": corpus.max_tgt_len,
        "model_config": cfg["model"],
        "best_val_exact": best_val,
        "best_epoch": best_epoch,
        "history": history,
        "train_seconds": elapsed,
        "data_seed": cfg["data"]["seed"],
    }
    run_dir = out_dir / f"{variant}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, run_dir / "best.pt")

    tests = {}
    for split_name, examples in corpus.tests.items():
        tests[split_name] = evaluate_split(
            model, examples, corpus, concept_to_id, device,
            batch_size=cfg["train"]["eval_batch_size"],
        )
        if use_concept:
            tests[split_name + "_oracle_concept"] = evaluate_split(
                model, examples, corpus, concept_to_id, device,
                batch_size=cfg["train"]["eval_batch_size"], oracle_concept=True,
            )
    return {
        "variant": variant,
        "seed": seed,
        "params": count_parameters(model),
        "train_seconds": elapsed,
        "best_val_exact": best_val,
        "best_epoch": best_epoch,
        "tests": tests,
        "checkpoint": str(run_dir / "best.pt"),
        "history": history,
    }


def aggregate(results: list[dict], split_names: list[str]) -> dict:
    out = {}
    for variant in sorted(set(r["variant"] for r in results)):
        runs = [r for r in results if r["variant"] == variant]
        out[variant] = {}
        for split in split_names:
            if split.endswith("_oracle_concept") and variant != "concept":
                continue
            runs_with = [r for r in runs if split in r["tests"]]
            if not runs_with:
                continue
            vals = [r["tests"][split]["exact"] for r in runs_with]
            f1s = [r["tests"][split]["f1"] for r in runs_with]
            out[variant][split] = {
                "n": runs_with[0]["tests"][split]["n"],
                "exact_mean": statistics.mean(vals),
                "exact_std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
                "f1_mean": statistics.mean(f1s),
            }
        concept_vals = [r["tests"][split_names[0]]["concept_acc"] for r in runs if r["tests"][split_names[0]]["concept_acc"] is not None]
        out[variant]["concept_acc"] = statistics.mean(concept_vals) if concept_vals else None
    return out


def write_report(path: Path, corpus: Corpus, results: list[dict], agg: dict, cfg: dict) -> None:
    lines = [
        "# ConceptChat — measured report",
        "",
        "Synthetic baby-English dialogue corpus. Test sets hold out templates and/or",
        "slot combinations; see the data generator for the exact split logic.",
        "",
        "## Data",
        "",
        f"- concepts: {', '.join(corpus.concepts)}",
        f"- vocab: {len(corpus.vocab)}",
        f"- train/val: {len(corpus.train)} / {len(corpus.val)}",
    ]
    for name, examples in corpus.tests.items():
        lines.append(f"- {name}: {len(examples)}")
    lines += ["", "## Held-out results (exact match, mean ± std over seeds)", "",
              "| variant | split | n | exact | token F1 |", "|---|---|---:|---:|---:|"]
    for variant, splits in agg.items():
        for split, values in splits.items():
            if split == "concept_acc":
                continue
            lines.append(
                f"| {variant} | {split} | {values['n']} | "
                f"{values['exact_mean']*100:.1f} ± {values['exact_std']*100:.1f}% | "
                f"{values['f1_mean']*100:.1f}% |"
            )
    lines += ["", "## Per-concept exact match (best checkpoint, concept variant)", ""]
    concept_rows = [r for r in results if r["variant"] == "concept"]
    if concept_rows:
        best = max(concept_rows, key=lambda r: r["best_val_exact"])
        lines.append("| concept | n | exact |")
        lines.append("|---|---:|---:|")
        merged: dict[str, list[float]] = {}
        for split in corpus.tests:
            for concept, values in best["tests"][split]["per_concept"].items():
                merged.setdefault(concept, []).append(values["exact"])
        for concept, values in sorted(merged.items()):
            lines.append(f"| {concept} | - | {statistics.mean(values)*100:.1f}% |")
    lines += ["", "## Notes", "",
              "- `concept` variant gets a concept label/classifier; `baseline` does not.",
              "- `oracle_concept` generation uses the true concept; the default uses the",
              "  model's predicted concept.",
              "- This is a closed-world synthetic corpus, not open-domain English.",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="concept_chat/configs/small.yaml")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.smoke:
        cfg["train"]["epochs"] = min(2, cfg["train"]["epochs"])
        cfg["train"]["seeds"] = cfg["train"]["seeds"][:1]
    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(10, os.cpu_count() or 1))
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    out_dir = Path(args.out or cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(cfg["data"]["seed"])
    concept_to_id = {name: index for index, name in enumerate(corpus.concepts)}
    with open(out_dir / "data_stats.json", "w", encoding="utf-8") as handle:
        json.dump(corpus.stats, handle, indent=2)
    print(f"device={device} torch={torch.__version__}")
    print("data:", corpus.stats["splits"])
    print("concepts:", ", ".join(corpus.concepts))

    results = []
    for variant in cfg.get("variants", ["concept", "baseline"]):
        for seed in cfg["train"]["seeds"]:
            results.append(train_one(corpus, concept_to_id, device, cfg, variant, seed, out_dir))

    split_names = list(corpus.tests.keys())
    if any(r["variant"] == "concept" for r in results):
        split_names += [name + "_oracle_concept" for name in list(corpus.tests.keys())]
    agg = aggregate(results, split_names)
    with open(out_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump({"config": cfg, "data": corpus.stats, "results": results, "aggregate": agg}, handle, indent=2)
    report_path = out_dir / "report.md"
    write_report(report_path, corpus, results, agg, cfg)

    concept_runs = [r for r in results if r["variant"] == "concept"]
    if concept_runs:
        best = max(concept_runs, key=lambda r: r["best_val_exact"])
        shutil.copyfile(best["checkpoint"], out_dir / "best.pt")
        print(f"copied {best['checkpoint']} -> {out_dir/'best.pt'}")

    print("\n=== held-out summary ===")
    for variant, splits in agg.items():
        for split, values in splits.items():
            if split == "concept_acc":
                continue
            print(f"{variant:>8} {split:<38} exact={values['exact_mean']*100:5.1f}%  n={values['n']}")
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
