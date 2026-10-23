"""Evaluate a saved ConceptChat checkpoint on all held-out splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .chat import choose_device, load_checkpoint
from .data import build_corpus
from .train import evaluate_split


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="concept_chat/runs/best.pt")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    corpus = build_corpus(checkpoint.get("data_seed", 1234))
    concept_to_id = {name: index for index, name in enumerate(corpus.concepts)}
    rows = {}
    for split, examples in corpus.tests.items():
        rows[split] = evaluate_split(
            model, examples, corpus, concept_to_id, device, oracle_concept=False
        )
        if model.use_concept:
            rows[split + "_oracle_concept"] = evaluate_split(
                model, examples, corpus, concept_to_id, device, oracle_concept=True
            )
    print(f"checkpoint={args.checkpoint} device={device} variant={checkpoint['variant']}")
    for split, values in rows.items():
        print(f"{split:<42} n={values['n']:4d} exact={values['exact']*100:5.1f}% "
              f"f1={values['f1']*100:5.1f}% concept="
              f"{values['concept_acc']*100 if values['concept_acc'] is not None else -1:5.1f}%")
    out = Path(args.out or Path(args.checkpoint).parent / "eval_report.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
