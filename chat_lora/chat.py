"""Interactive chat with the general-English LoRA model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import format_prompt

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="chat_lora/runs/qwen05b/best")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    device = choose_device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype)
    if not args.base_only:
        from peft import PeftModel
        if not Path(args.adapter).exists():
            raise SystemExit(f"adapter not found: {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to(device)
    model.eval()
    model.config.use_cache = True

    history: list[dict] = []
    print(f"Pip | base={args.base} | adapter={'none' if args.base_only else args.adapter}")
    print("Commands: :quit  :reset  :help")
    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in (":quit", ":exit", "quit", "exit"):
            break
        if user == ":reset":
            history = []
            print("history cleared")
            continue
        if user == ":help":
            print("Talk normally. The model is a 0.5B pretrained Transformer with LoRA.")
            continue

        prompt = format_prompt(tokenizer, user, history)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        reply = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
        if not reply:
            reply = "I am not sure what to say."
        print(f"Pip> {reply}")
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": reply})
        history = history[-8:]


if __name__ == "__main__":
    main()
