"""Download TinyStories, train a small BPE tokenizer, and pack uint16 tokens.

Why a custom 16k tokenizer?  At this model scale the LM head (vocab * d) is a
large fraction of total FLOPs.  A 16k TinyStories BPE keeps the head cheap,
which makes the compute saved by Mixture-of-Depths visible in wall-clock time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm

EOS_TOKEN = "<|endoftext|>"


def iter_texts(dataset, limit: int):
    seen = 0
    for batch in dataset.iter(batch_size=2048):
        for text in batch["text"]:
            yield text
            seen += 1
            if limit > 0 and seen >= limit:
                return


def train_tokenizer(
    dataset,
    vocab_size: int,
    sample: int,
    out_path: Path,
) -> ByteLevelBPETokenizer:
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        iter_texts(dataset, sample),
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=[EOS_TOKEN],
        show_progress=True,
    )
    tokenizer.save(str(out_path))
    return tokenizer


def encode_split(
    dataset,
    tokenizer: ByteLevelBPETokenizer,
    out_path: Path,
    eos_id: int,
    desc: str,
) -> int:
    n_tokens = 0
    with open(out_path, "wb") as handle, tqdm(
        total=len(dataset), desc=desc, unit="story"
    ) as progress:
        for batch in dataset.iter(batch_size=4096):
            encodings = tokenizer.encode_batch(batch["text"])
            flat: list[int] = []
            for encoding in encodings:
                flat.extend(encoding.ids)
                flat.append(eos_id)
            if not flat:
                continue

            array = np.asarray(flat, dtype=np.uint16)
            if int(array.max()) >= 65536:
                raise ValueError("token id does not fit in uint16")
            array.tofile(handle)
            n_tokens += int(array.size)
            progress.update(len(batch["text"]))
    return n_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="roneneldan/TinyStories")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument(
        "--tokenizer-sample",
        type=int,
        default=200_000,
        help="number of training stories used to fit the BPE tokenizer (0 = all)",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not (256 <= args.vocab_size < 65536):
        raise SystemExit("--vocab-size must be in [256, 65535]")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_bin = out_dir / "train.bin"
    val_bin = out_dir / "val.bin"
    tokenizer_path = out_dir / "tokenizer.json"
    meta_path = out_dir / "meta.json"

    existing = [path for path in (train_bin, val_bin, tokenizer_path, meta_path) if path.exists()]
    if existing and not args.force:
        names = ", ".join(str(path) for path in existing)
        raise SystemExit(f"output already exists: {names}\nRe-run with --force to rebuild.")
    if args.force:
        for path in existing:
            path.unlink()

    print(f"Loading {args.dataset} ...")
    dataset = load_dataset(args.dataset)
    split_names = list(dataset.keys())
    if args.train_split not in split_names:
        raise SystemExit(
            f"split {args.train_split!r} not found; available: {split_names}"
        )
    if args.val_split not in split_names:
        raise SystemExit(
            f"split {args.val_split!r} not found; available: {split_names}"
        )
    train_ds = dataset[args.train_split]
    val_ds = dataset[args.val_split]
    print(f"train stories: {len(train_ds):,}  val stories: {len(val_ds):,}")

    print(f"Training {args.vocab_size}-token byte-level BPE ...")
    tokenizer = train_tokenizer(
        train_ds,
        vocab_size=args.vocab_size,
        sample=args.tokenizer_sample,
        out_path=tokenizer_path,
    )
    vocab_size = tokenizer.get_vocab_size()
    eos_id = tokenizer.token_to_id(EOS_TOKEN)
    if eos_id is None:
        raise RuntimeError("EOS token was not added to the tokenizer")
    if vocab_size >= 65536:
        raise RuntimeError(f"vocab size {vocab_size} does not fit in uint16")
    print(f"tokenizer vocab size: {vocab_size:,}  eos id: {eos_id}")

    started = time.time()
    train_tokens = encode_split(train_ds, tokenizer, train_bin, eos_id, "Tokenizing train")
    print(f"train tokens: {train_tokens:,} ({time.time() - started:.1f}s)")
    started = time.time()
    val_tokens = encode_split(val_ds, tokenizer, val_bin, eos_id, "Tokenizing val")
    print(f"val tokens: {val_tokens:,} ({time.time() - started:.1f}s)")

    meta = {
        "dataset": args.dataset,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "tokenizer": "byte-level-bpe",
        "tokenizer_file": tokenizer_path.name,
        "vocab_size": vocab_size,
        "eos_token": EOS_TOKEN,
        "eos_token_id": eos_id,
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "dtype": "uint16",
        "tokenizer_sample": args.tokenizer_sample,
        "created_unix": int(time.time()),
    }
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)
        handle.write("\n")

    print("\nDone.")
    print(f"  {tokenizer_path}")
    print(f"  {train_bin}  ({train_tokens:,} tokens)")
    print(f"  {val_bin}    ({val_tokens:,} tokens)")
    print(f"  {meta_path}  (vocab_size={vocab_size:,})")


if __name__ == "__main__":
    main()
