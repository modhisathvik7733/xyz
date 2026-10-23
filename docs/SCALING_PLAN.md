# From 42M proxy to 1B / 200B tokens / 256k context

**Verdict up front:** a 1B dense model trained on 200B tokens at 256k full
context is not a $150 workload. Even the most aggressive plausible design —
1B total parameters, ~100M active, hybrid linear/local attention, 8k main
context plus 1% at 256k — lands at roughly **$150–250 on 8× RTX 4090 spot
instances**, before restarts and engineering overhead. A realistic budget is
**$200–300**, or you reduce one of: active parameters, tokens, or long-context
fraction.

This document is the research/engineering plan. The measured 42M proxy is in
[MEASURED_5060TI.md](MEASURED_5060TI.md). The calculator is:

```bash
python -m src.plan
python -m src.plan --gpu a100 --gpus 1
python -m src.plan --gpu h100 --gpus 8 --mfu 0.40
```

---

## 1. The budget arithmetic

`src/plan.py` uses:

- model training FLOPs/token = `6 × active_params`
- attention training FLOPs/token = `12 × effective_context × d_model × layers`
- hybrid attention = 25% local-window layers + 75% linear layers at 10% of the
  windowed cost
- 35% MFU, 8× RTX 4090 at **$0.13/GPU-hour** (Vast.ai “from” price,
  interruptible; observed Oct 2026), 1B total parameters, 200B tokens.
  Prices: [RTX 4090](https://vast.ai/pricing/gpu/RTX-4090),
  [A100 SXM4](https://vast.ai/pricing/gpu/A100-SXM4),
  [H100 SXM](https://vast.ai/pricing/gpu/H100-SXM).

| scenario | GFLOPs/token | GPU-hours | wall-hours | cost | verdict |
|---|---:|---:|---:|---:|---|
| dense 1B, full 8k | 9.62 | 9,258 | 1,157 | $1,204 | over |
| dense 1B, full 256k | 121.96 | 117,330 | 14,666 | $15,253 | impossible |
| 100M active MoE, full 8k | 4.22 | 4,063 | 508 | $528 | over |
| 100M active MoE, local 4k | 2.41 | 2,320 | 290 | $302 | over |
| 100M active MoE, hybrid 8k + 256k on 1% | 1.19 | 1,144 | 143 | $149 | only at 35% MFU |
| 120M active MoE, hybrid | 1.31 | 1,259 | 157 | $164 | close |
| 150M active MoE, hybrid | 1.49 | 1,432 | 179 | $186 | over |

At $150 and $0.13/GPU-hour you get **1,154 GPU-hours** (144 wall-hours on 8
GPUs). For 200B tokens that is a required aggregate throughput of
**~385k tokens/sec**, or **~48k tokens/sec per 4090**.

The same budget buys only:

| GPU | price | GPU-hours for $150 | required tok/s for 200B | realistic? |
|---|---:|---:|---:|---|
| RTX 4090 | $0.13/h | 1,154 | 48k/GPU × 8 | only at high MFU |
| A100 80GB | $0.31/h | 484 | 115k total | no at 35% MFU |
| H100 SXM | $1.60/h | 94 | 593k total | no |

The counter-intuitive result: **cheap 4090s beat H100s on $/FLOP for this
workload.** At $0.13/h and 58 effective TFLOP/s, a 4090 costs about
**$0.62 per 10^18 FLOPs**. An A100 at $0.31/h and 55 effective TFLOP/s costs
about $1.57; an H100 at $1.60/h and 173 effective TFLOP/s costs about $2.57.
Reliability/interruption and availability can erase the 4090 advantage, so
budget a 30% buffer.

---

## 2. Why the 42M speedup is not the 1B speedup

The measured 5060 Ti proxy:

- dense 42M: **104,078 tok/s** at 512/32, effective **30.3 TFLOP/s**
- fast 42M: **143,693 tok/s**, effective **26.3 TFLOP/s**
- wall-clock speedup: **1.38×**

The fast model is faster because it performs ~37% fewer FLOPs, not because its
kernels have higher utilization. At 1B the ratio changes in both directions:

- **Good:** the fixed LM head is a much smaller fraction of total FLOPs, and
  larger GEMMs usually reach higher MFU.
- **Bad:** MoE routing, gather/scatter, and sparse-attention kernels add
  overhead; the 42M proxy overstates the relative routing cost but understates
  the difficulty of keeping 1B kernels fed.

For a 1B total / 100M active hybrid model at ~1.19 GFLOPs/token:

- at 26.3 effective TFLOP/s (5060 Ti proxy): **22k tok/s**; 200B tokens =
  ~2,514 GPU-hours, ~105 days on one card
- at 58 effective TFLOP/s (per 4090, 35% MFU): **48.6k tok/s**; 8 cards =
  ~388k tok/s aggregate; 200B tokens = ~143 hours

So the required jump is **~2.2× effective FLOP/s per GPU** versus the 5060 Ti
proxy, plus 8-way model parallelism. That is plausible with a 1B workload and
larger GEMMs, but it is not guaranteed.

---

## 3. What the research says

### Sparse MoE is the main parameter/FLOP lever

DeepSeek-V3: **671B total / 37B active** (5.5%), trained on 14.8T tokens in
2.788M H800 GPU-hours using MLA, DeepSeekMoE, FP8, and multi-token prediction.
It demonstrates that total capacity and active compute can be decoupled at
frontier scale. ([arXiv:2412.19437](https://arxiv.org/abs/2412.19437))

Qwen3-Next is the closest public blueprint for your target: **80B total / 3B
active** (3.75%), a **3:1 hybrid of Gated DeltaNet (linear) and gated
attention**, 512 experts with 10 routed + 1 shared, and **256k context**. Qwen
reports performance comparable to a dense 32B model at **less than 10% of its
training GPU-hours**, and >10× inference throughput above 32k context.
([Alibaba Cloud blog](https://www.alibabacloud.com/blog/602580))

### Long context needs sparse or linear attention, not full 256k

At `d_model=1536`, 24 layers, full 256k attention is **~116 GFLOPs/token** in
training — roughly 190× the model FLOPs of a 100M-active model. No optimizer or
MoE trick fixes that.

Relevant trainable sparse-attention work:

- **NSA** (Native Sparse Attention): hierarchical compression + selection +
  sliding window; up to **9.0× forward / 6.0× backward** speedup at 64k, and
  **11.6×** decoding speedup, with pretraining quality maintained.
  ([arXiv:2502.11089](https://arxiv.org/abs/2502.11089))
- **MoBA** (Mixture of Block Attention): top-k block routing; up to **6.5×**
  prefill speedup at 1M tokens and **16×** at 10M.
  ([arXiv:2502.13189](https://arxiv.org/abs/2502.13189))
- **DSA** (DeepSeek Sparse Attention) in DeepSeek-V3.2: sparse attention aimed
  at long-context reasoning and agentic tasks.
  ([arXiv:2512.02556](https://arxiv.org/abs/2512.02556))
- **MiniMax-01**: lightning linear attention + MoE, 456B total / 45.9B active,
  1M training context and 4M inference extrapolation.
  ([arXiv:2501.08313](https://arxiv.org/abs/2501.08313))

For a coding agent, the practical answer is **staged context**, not 200B tokens
at 256k: train 95–98% of tokens at 8k, 2–4% at 32k, and ~1% at 256k with
repo-level long-context data and sparse attention.

### Muon can buy back compute efficiency

Muon reports **~2× computational efficiency vs AdamW** in compute-optimal
scaling, and Moonlight (3B/16B MoE) was trained on 5.7T tokens with it.
([arXiv:2502.16982](https://arxiv.org/abs/2502.16982)) This does not make each
step faster; it can reach a target loss in fewer tokens/FLOPs. Treat the 2× as
a potential budget buffer, not a guarantee for a 1B coding model.

### You are already overtraining relative to Chinchilla

Chinchilla’s compute-optimal ratio is about **20 tokens/parameter**
([arXiv:2203.15556](https://arxiv.org/abs/2203.15556)). 1B × 200B is **200
tokens/parameter**, 10× over compute-optimal. That is sensible for a model you
will serve heavily (Llama-style overtraining), but if the $150 cap is hard, the
first lever is **tokens**: 1B × 20–40B is 5–10× cheaper and may be the right
prototype before spending on 200B.

---

## 4. Recommended v2 architecture

Do **not** extend the current 42M single-GPU `train.py` directly to 1B. Use a
real distributed stack (Megatron-LM, torchtitan, nanotron, or LitGPT) and keep
this repo for architecture ablations and cost planning.

### Target spec

| component | choice |
|---|---|
| total / active params | 1.0B / 100–120M |
| layers | 20–24 |
| d_model | 1280–1536 |
| attention | GQA, 10–12 query heads, 2–4 KV heads, head_dim 128 |
| FFN | 32–64 fine-grained experts, top-1 or top-2, plus one shared expert |
| attention pattern | 3:1 Gated DeltaNet : gated local attention (4k window) |
| global long context | NSA/MoBA/DSA blocks on the attention layers |
| context schedule | 8k main, 32k midtrain, 256k on ~1% of tokens |
| vocab | 64k–96k coding/agentic BPE |
| optimizer | Muon for 2D weights + AdamW for embeddings/router (or AdamW first) |
| parallel | FSDP/ZeRO-3 + activation checkpointing + sequence/context parallel |
| hardware | 8× RTX 4090 24GB, or 4× A100 80GB |
| inference | MLA/GQA KV cache, 4-bit/8-bit KV for 256k |

At d=1280, 20 layers, 32 experts and expert hidden ≈ 384, a top-1 MoE is about
1.1B total and ~108M active; top-2 is ~138M active. The exact split should be
tuned with a 1B parameter-count script before renting.

### Why not MoD as the main 1B lever?

MoD is attractive at 42M because the LM head and small GEMMs dominate. At 1B,
**MoE sparsity and attention complexity** are the larger levers. MoD can be
layered on top later, but two dynamic routers (depth + experts) make training
less stable and complicate distributed dispatch. For v2, use MoE + hybrid
attention; keep MoD as the 42M research proxy.

### Why not full 256k training?

Full attention at 256k dominates by two orders of magnitude. Use NSA/MoBA/DSA
or hybrid linear attention, and train long context only on a small, curated
fraction. This is the difference between a $150–250 run and a $15,000 run.

---

## 5. Build order

1. **1200-step triage on the 42M proxy** — `bash scripts/run_1200.sh`.
   Confirms the speed/quality trade-off at small scale and produces a reusable
   comparison report.
2. **1B microbenchmark before renting** — instantiate the target MoE on 8×4090
   with FSDP, measure tokens/s and MFU at 8k and 32k. Do not trust the
   analytical 35% MFU.
3. **Replace the trainer** — FSDP/ZeRO-3, activation checkpointing, sequence
   parallel; grouped-GEMM MoE kernels (Megablocks/Tutel); FlashAttention /
   FlexAttention / NSA-style sparse kernels.
4. **Add Muon** after the baseline is stable; compare loss at equal tokens and
   equal wall-clock.
5. **Stage context** — 8k → 32k → 256k, with long-code data only in the last
   stage.
6. **Budget buffer** — reserve 30% for restarts, spot interruption, and
   evaluation.

### What not to do

- 200B tokens at full 256k attention.
- A dense 1B model for a $150 budget.
- Train 1B on a 16GB 5060 Ti.
- Rent H100s for this workload at list prices; the $/FLOP is worse than cheap
  4090s.
- Chase a 42M speedup above 1.5×; the 1B ratio is a different problem.

---

## 6. Bottom line

- **1B dense / 200B / 256k full:** not feasible at $150. Expect ~$1,200 for 8k
  and ~$15,000 for full 256k at 35% MFU on cheap 4090s.
- **1B total / 100M active / hybrid attention / 1% at 256k:** ~$149–250 on
  8×4090 at 35% MFU. This is the only path close to the budget, and it requires
  a real MoE + sparse-attention stack.
- **Hard $150 cap:** use ~100M active, 150–200B tokens, 8k main context, and
  long-context extension on ~1% of tokens; expect $180–250 with real overhead.
- **If quality matters more than the 1B label:** a 300–500M active dense/MoE
  model trained on 20–40B tokens will be cheaper and easier to validate, but it
  is not a 1B model.
- **Quality expectation:** even the 1B total / 100M active target is a narrow
  coding model, not a frontier assistant. 1B total capacity with 100–120M
  active parameters is roughly a well-trained 100M-class compute budget with a
  larger memory/capacity pool; expect useful code completion and repo-level
  patterns, not GPT-4-class reasoning.
- **Budget scope:** all dollar figures are GPU rental only. They exclude data
  acquisition/curation, storage and egress, failed runs, and engineering time.
  Collecting and cleaning 200B coding/agentic tokens is a separate project.

The next useful measurement is not another 42M speed run. It is a **1B
microbenchmark** on the actual rented GPU, followed by a 1200-step 42M
quality/speed triage. The repo now gives you both the triage harness and the
budget calculator.
