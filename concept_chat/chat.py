"""Chat with a trained ConceptChat checkpoint (single-turn baby English)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from .data import decode, encode, tokenize
from .model import ConceptSeq2Seq

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


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


def load_checkpoint(path: str | Path, device: torch.device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["model_config"]
    model = ConceptSeq2Seq(
        vocab_size=len(checkpoint["vocab"]),
        num_concepts=len(checkpoint["concepts"]),
        emb=cfg["emb"],
        hidden=cfg["hidden"],
        dropout=cfg["dropout"],
        use_concept=checkpoint["variant"] == "concept",
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="concept_chat/runs/best.pt")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--show-concept", action="store_true")
    p.add_argument("--max-len", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    vocab = checkpoint["vocab"]
    itos = checkpoint["itos"]
    concepts = checkpoint["concepts"]
    max_src = checkpoint["max_src_len"]
    max_len = args.max_len or checkpoint["max_tgt_len"]
    concept_override = None

    print(f"ConceptChat | device={device} | variant={checkpoint['variant']} | "
          f"val_exact={checkpoint.get('best_val_exact', 0.0)*100:.1f}%")
    print("This model knows a small closed vocabulary. Type simple English.")
    print("Commands: :quit  :concept <name>  :auto  :concepts  :help")
    print("Concepts: " + ", ".join(concepts))

    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in (":quit", ":exit", "quit", "exit"):
            break
        if text == ":help":
            print("Talk in simple English. :concept name forces a concept; :auto returns to prediction.")
            continue
        if text == ":concepts":
            print(", ".join(concepts))
            continue
        if text == ":auto":
            concept_override = None
            print("concept: auto")
            continue
        if text.startswith(":concept"):
            parts = text.split(maxsplit=1)
            if len(parts) != 2 or parts[1] not in concepts:
                print("usage: :concept <name>")
                continue
            concept_override = concepts.index(parts[1])
            print(f"concept: {parts[1]}")
            continue

        tokens = tokenize(text)
        oov = sorted({token for token in tokens if token not in vocab})
        if oov:
            print(f"[unknown words -> <unk>: {', '.join(oov)}]")
        src = torch.tensor([encode(text, vocab, max_src)], dtype=torch.long, device=device)
        generated, predicted = model.generate(
            src, max_len, concept_override=concept_override, repetition_penalty=1.2
        )
        reply = decode(generated[0], itos)
        if predicted is not None:
            concept_name = concepts[int(predicted[0])]
        else:
            concept_name = "baseline"
        if args.show_concept:
            print(f"[concept: {concept_name}]")
        if not reply:
            reply = "i do not know what to say"
        if not reply.endswith((".", "!", "?")):
            reply += "."
        print(f"Pip> {reply}")


if __name__ == "__main__":
    main()
