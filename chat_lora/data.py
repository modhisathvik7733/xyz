"""Data formatting for the general-English LoRA chat track.

This is deliberately not the template model.  It fine-tunes a small pretrained
Transformer on real UltraChat conversations and evaluates on held-out
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
    """Turn one alternating dialogue into user/assistant pairs."""
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


def pairs_from_messages(dataset, max_examples: int = 0, seed: int = 0) -> list[DialoguePair]:
    """Extract user->assistant pairs from a messages-format dataset."""
    pairs: list[DialoguePair] = []
    for row in dataset:
        messages = row.get("messages") or []
        for index in range(len(messages) - 1):
            current, following = messages[index], messages[index + 1]
            if current.get("role") == "user" and following.get("role") == "assistant":
                user = str(current.get("content", "")).strip()
                assistant = str(following.get("content", "")).strip()
                if user and assistant:
                    pairs.append(DialoguePair(user, assistant))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_examples and max_examples > 0:
        pairs = pairs[:max_examples]
    return pairs


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
    name = data_cfg.get("name", "HuggingFaceH4/ultrachat_200k")
    train_raw = load_dataset(name, split=data_cfg.get("split_train", "train_sft"))
    test_raw = load_dataset(name, split=data_cfg.get("split_test", "test_sft"))

    all_train = pairs_from_messages(
        train_raw, data_cfg.get("max_train_examples", 20_000) + data_cfg.get("max_val_examples", 1_500),
        data_cfg.get("seed", 0),
    )
    test_pairs = pairs_from_messages(
        test_raw, data_cfg.get("max_test_examples", 1_500), data_cfg.get("seed", 2)
    )
    if quick:
        all_train = all_train[:2_200]
        test_pairs = test_pairs[:300]

    val_count = min(data_cfg.get("max_val_examples", 1_500), max(1, len(all_train) // 10))
    val_pairs = all_train[-val_count:]
    train_pairs = all_train[:-val_count]
    train_pairs = train_pairs[: data_cfg.get("max_train_examples", 20_000)]
    max_len = data_cfg.get("max_seq_len", 512)
    train = DialogueDataset(train_pairs, tokenizer, max_len)
    val = DialogueDataset(val_pairs, tokenizer, max_len)
    test = DialogueDataset(test_pairs, tokenizer, max_len)
    return train, val, test, train_pairs, val_pairs, test_pairs
