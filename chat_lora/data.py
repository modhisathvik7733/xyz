"""Data formatting for the general-English LoRA chat track.

This is deliberately not the template model.  It fine-tunes a small pretrained
Transformer on real DailyDialog conversations and evaluates on held-out
dialogues.  The base model supplies general English; LoRA adapts its style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import random

SYSTEM_PROMPT = (
    "You are Pip, a friendly little learner. "
    "Answer in simple English with one or two short sentences. "
    "If you do not know something, say so."
)


@dataclass(frozen=True)
class DialoguePair:
    user: str
    assistant: str


def extract_pairs(dialog: Iterable[str], max_pairs: Optional[int] = None) -> list[DialoguePair]:
    """Turn one alternating DailyDialog dialogue into user/assistant pairs."""
    clean = [str(text).strip() for text in dialog if str(text).strip()]
    pairs: list[DialoguePair] = []
    for index in range(1, len(clean), 2):
        pairs.append(DialoguePair(clean[index - 1], clean[index]))
        if max_pairs is not None and len(pairs) >= max_pairs:
            break
    return pairs


def pairs_from_dataset(dataset, max_examples: int, seed: int = 0) -> list[DialoguePair]:
    pairs: list[DialoguePair] = []
    for dialog in dataset["dialog"]:
        pairs.extend(extract_pairs(dialog))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:max_examples]


def format_prompt(tokenizer, user: str, history: Optional[list[dict]] = None) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user.strip()})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_example(tokenizer, pair: DialoguePair, max_len: int = 512):
    prompt = format_prompt(tokenizer, pair.user)
    answer = pair.assistant.strip()
    if not answer:
        return None
    full = prompt + answer + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(
        full, add_special_tokens=False, truncation=True, max_length=max_len
    )["input_ids"]
    if len(prompt_ids) >= len(full_ids):
        return None
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


class DialogueDataset:
    def __init__(self, pairs: list[DialoguePair], tokenizer, max_len: int = 512):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        item = encode_example(self.tokenizer, self.pairs[index], self.max_len)
        if item is None:
            # A malformed/empty dialogue should not crash a long run.
            return {
                "input_ids": [self.tokenizer.eos_token_id],
                "attention_mask": [1],
                "labels": [self.tokenizer.eos_token_id],
            }
        return item


class PadCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]):
        import torch

        width = max(len(item["input_ids"]) for item in features)
        input_ids, attention, labels = [], [], []
        for item in features:
            pad = width - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad)
            attention.append(item["attention_mask"] + [0] * pad)
            labels.append(item["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_datasets(config: dict, tokenizer, quick: bool = False):
    from datasets import load_dataset

    data_cfg = config["data"]
    raw = load_dataset(data_cfg.get("name", "daily_dialog"))
    train_pairs = pairs_from_dataset(
        raw["train"], data_cfg.get("max_train_examples", 40_000), data_cfg.get("seed", 0)
    )
    val_pairs = pairs_from_dataset(
        raw["validation"], data_cfg.get("max_val_examples", 2_000), data_cfg.get("seed", 1)
    )
    test_pairs = pairs_from_dataset(
        raw["test"], data_cfg.get("max_test_examples", 2_000), data_cfg.get("seed", 2)
    )
    if quick:
        train_pairs = train_pairs[:2_000]
        val_pairs = val_pairs[:200]
        test_pairs = test_pairs[:300]
    max_len = data_cfg.get("max_seq_len", 512)
    train = DialogueDataset(train_pairs, tokenizer, max_len)
    val = DialogueDataset(val_pairs, tokenizer, max_len)
    test = DialogueDataset(test_pairs, tokenizer, max_len)
    return train, val, test, train_pairs, val_pairs, test_pairs
