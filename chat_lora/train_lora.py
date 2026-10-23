"""LoRA fine-tuning of a small pretrained Transformer on DailyDialog.

This is the general-English track, not the from-scratch concept experiment.
The base model supplies English; LoRA adapts it to short, friendly replies.
Evaluation is on held-out DailyDialog pairs (loss/perplexity plus samples).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from .data import DialogueDataset, PadCollator, format_prompt, make_datasets

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


@torch.no_grad()
def evaluate_loss(model, loader, device, autocast_dtype):
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else nullcontext():
            outputs = model(**batch)
        total += float(outputs.loss.detach()) * batch["input_ids"].shape[0]
        count += batch["input_ids"].shape[0]
    return total / max(1, count)


@torch.no_grad()
def generate_samples(model, tokenizer, pairs, device, max_new_tokens=80, limit=3):
    model.eval()
    samples = []
    for pair in pairs[:limit]:
        prompt = format_prompt(tokenizer, pair.user)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.1,
            no_repeat_ngram_size=3,
        )
        reply = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        samples.append({"user": pair.user, "reference": pair.assistant, "reply": reply})
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="chat_lora/configs/qwen05b.yaml")
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.quick:
        cfg["train"]["max_steps"] = cfg["train"].get("quick_max_steps", 200)
        cfg["train"]["eval_steps"] = 100
    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(min(10, os.cpu_count() or 1))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    base_name = cfg["model"]["base"]
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    dtype = torch.float32
    autocast_dtype = None
    if device.type == "cuda" and cfg["model"].get("bf16", True):
        dtype = torch.bfloat16
        autocast_dtype = torch.bfloat16
    elif device.type == "cuda":
        dtype = torch.float16
        autocast_dtype = torch.float16

    print(f"device={device} torch={torch.__version__} base={base_name} dtype={dtype}")
    train_ds, val_ds, test_ds, _, val_pairs, _ = make_datasets(cfg, tokenizer, args.quick)
    collator = PadCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        collate_fn=collator, num_workers=cfg["train"].get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["eval_batch_size"], shuffle=False,
        collate_fn=collator, num_workers=cfg["train"].get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg["train"]["eval_batch_size"], shuffle=False,
        collate_fn=collator, num_workers=cfg["train"].get("num_workers", 2),
        pin_memory=device.type == "cuda",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.to(device)
    model.config.use_cache = False
    print(f"train/val/test pairs: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}")

    base_val = evaluate_loss(model, val_loader, device, autocast_dtype)
    base_test = evaluate_loss(model, test_loader, device, autocast_dtype)
    print(f"base val loss={base_val:.4f} ppl={math.exp(min(base_val, 20)):.2f}")

    from peft import LoraConfig, TaskType, get_peft_model

    lora_cfg = cfg["lora"]
    lora = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    if cfg["train"].get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    grad_accum = cfg["train"]["grad_accum"]
    max_steps = int(cfg["train"].get("max_steps", 0) or 0)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = max_steps or steps_per_epoch * cfg["train"]["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=cfg["train"]["warmup_steps"], num_training_steps=total_steps
    )

    out_dir = Path(args.out or cfg["train"]["out_dir"])
    (out_dir / "adapter").mkdir(parents=True, exist_ok=True)
    (out_dir / "best").mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.synchronize()

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        for micro, batch in enumerate(train_loader):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=autocast_dtype) if autocast_dtype else nullcontext():
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            loss.backward()
            if (micro + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % cfg["train"]["log_steps"] == 0:
                    print(
                        f"epoch={epoch} step={global_step} loss={float(outputs.loss.detach()):.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}"
                    )
                if global_step % cfg["train"]["eval_steps"] == 0:
                    val = evaluate_loss(model, val_loader, device, autocast_dtype)
                    print(f"  val loss={val:.4f} ppl={math.exp(min(val, 20)):.2f}")
                    if val < best_val:
                        best_val = val
                        model.save_pretrained(out_dir / "best")
                        tokenizer.save_pretrained(out_dir / "best")
                if max_steps and global_step >= max_steps:
                    break
        if max_steps and global_step >= max_steps:
            break

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if best_val == float("inf"):
        best_val = evaluate_loss(model, val_loader, device, autocast_dtype)
        model.save_pretrained(out_dir / "best")
        tokenizer.save_pretrained(out_dir / "best")
    model.save_pretrained(out_dir / "adapter")
    tokenizer.save_pretrained(out_dir / "adapter")

    lora_val = evaluate_loss(model, val_loader, device, autocast_dtype)
    lora_test = evaluate_loss(model, test_loader, device, autocast_dtype)
    samples = generate_samples(model, tokenizer, val_pairs, device)
    results = {
        "config": cfg,
        "base": base_name,
        "device": str(device),
        "train_pairs": len(train_ds),
        "val_pairs": len(val_ds),
        "test_pairs": len(test_ds),
        "steps": global_step,
        "seconds": elapsed,
        "base_val_loss": base_val,
        "base_test_loss": base_test,
        "lora_val_loss": lora_val,
        "lora_test_loss": lora_test,
        "base_val_ppl": math.exp(min(base_val, 20)),
        "base_test_ppl": math.exp(min(base_test, 20)),
        "lora_val_ppl": math.exp(min(lora_val, 20)),
        "lora_test_ppl": math.exp(min(lora_test, 20)),
        "samples": samples,
    }
    with open(out_dir / "results.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    report = [
        "# General-English LoRA chat — measured report",
        "",
        f"- base: `{base_name}`",
        f"- train/val/test pairs: {len(train_ds)}/{len(val_ds)}/{len(test_ds)}",
        f"- steps: {global_step} in {elapsed/60:.1f} min on {device}",
        "",
        "| model | val loss | val ppl | test loss | test ppl |",
        "|---|---:|---:|---:|---:|",
        f"| base | {base_val:.4f} | {math.exp(min(base_val,20)):.2f} | {base_test:.4f} | {math.exp(min(base_test,20)):.2f} |",
        f"| + LoRA | {lora_val:.4f} | {math.exp(min(lora_val,20)):.2f} | {lora_test:.4f} | {math.exp(min(lora_test,20)):.2f} |",
        "",
        "## Held-out samples",
        "",
    ]
    for sample in samples:
        report += [
            f"**user:** {sample['user']}",
            "",
            f"**reference:** {sample['reference']}",
            "",
            f"**LoRA:** {sample['reply']}",
            "",
        ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"base test ppl={results['base_test_ppl']:.2f}  LoRA test ppl={results['lora_test_ppl']:.2f}")
    print(f"wrote {out_dir/'report.md'}")
    print(f"adapter: {out_dir/'best'}")


if __name__ == "__main__":
    main()
