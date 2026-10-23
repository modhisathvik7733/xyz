# General-English chat: small pretrained Transformer + LoRA

**This is a deliberate pivot from the closed-world template model.** The
`concept_chat` model is a 0.8M-parameter GRU trained only on synthetic templates;
it cannot generalize like a Transformer. No amount of retraining that tiny
closed-world model will make it a general English speaker.

This track uses a **small pretrained Transformer** — Qwen2.5-0.5B-Instruct — and
LoRA fine-tunes it on **real DailyDialog conversations**. The base model supplies
general English; LoRA adapts the style to short, friendly, simple replies.

This is **not** the from-scratch concept-learning experiment. It is the honest
way to get a usable general-English chat model on a 20GB RTX 3080.

## Run on the 3080

```bash
cd /workspace/xyz
git pull

python -m pip install -r chat_lora/requirements.txt

# quick smoke: ~200 optimizer steps, a few minutes
python -u -m chat_lora.train_lora \
  --config chat_lora/configs/qwen05b.yaml \
  --device cuda \
  --quick
```

For the real run:

```bash
python -u -m chat_lora.train_lora \
  --config chat_lora/configs/qwen05b.yaml \
  --device cuda
```

Or:

```bash
bash chat_lora/scripts/train_vastai.sh
```

Outputs:

```text
chat_lora/runs/qwen05b/report.md
chat_lora/runs/qwen05b/results.json
chat_lora/runs/qwen05b/best/          # LoRA adapter selected by val loss
chat_lora/runs/qwen05b/adapter/       # final adapter
```

## Talk to it

```bash
python -m chat_lora.chat \
  --adapter chat_lora/runs/qwen05b/best \
  --device cuda
```

Compare with the un-finetuned base:

```bash
python -m chat_lora.chat --base-only --device cuda
```

Commands: `:quit`, `:reset`.

## What is measured

- Real DailyDialog train/validation/test splits.
- Held-out validation and test **loss/perplexity** before and after LoRA.
- Generated held-out samples in `report.md`.
- No template test set is used here.

This is a proper language-model evaluation, but perplexity is not the same as
chat quality. The interactive chat is the qualitative check.

## Expected behavior

The model will speak general English because Qwen2.5-0.5B-Instruct already does.
LoRA mainly changes the tone toward short, simple, friendly replies. It is still
a 0.5B model:

- it can be generic, repetitive, or wrong;
- it has no concept-mastery tracker;
- it does not learn from your chat messages;
- it is not ChatGPT.

If you want a from-scratch model that learns concepts with explicit teaching,
that is the separate `concept_chat`/`concept_lab` line of work. This folder does
not claim to replace it.

## Runtime

The 0.5B model in bf16 plus LoRA fits comfortably in 20GB with batch 4,
gradient accumulation 4, and gradient checkpointing. The default run uses 20k
DailyDialog pairs for one epoch; the quick run uses 2k pairs and 200 steps.

If you hit OOM, reduce `train.batch_size` to 2 and increase `train.grad_accum`.
