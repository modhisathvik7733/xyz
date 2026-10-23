# Reference measurement: RTX 5060 Ti, 42M proxy

This is the first real measurement of the repo, provided by the repo owner.
It is a **42M-parameter proxy**, not a 1B model.

- GPU: NVIDIA GeForce RTX 5060 Ti
- PyTorch: `2.11.0+cu128`
- dtype: bf16, `torch.compile` default mode
- benchmark: random tokens, one reused batch, 5 warmup + 20 timed steps
- date: 2026-10-23

## Results

| model | seq | batch | tok/s | ms/step | peak GB | est. TFLOP/s |
|---|---:|---:|---:|---:|---:|---:|
| dense | 256 | 8 | 77,596 | 26.39 | 1.35 | 21.1 |
| dense | 256 | 16 | 101,064 | 40.53 | 2.06 | 27.5 |
| dense | 256 | 32 | 104,527 | 78.37 | 3.40 | 28.4 |
| dense | 512 | 8 | 98,416 | 41.62 | 2.06 | 28.6 |
| dense | 512 | 16 | 101,866 | 80.42 | 3.40 | 29.6 |
| dense | 512 | 32 | 104,078 | 157.42 | 6.17 | 30.3 |
| fast | 256 | 8 | 94,745 | 21.62 | 1.16 | 16.7 |
| fast | 256 | 16 | 123,067 | 33.28 | 1.68 | 21.6 |
| fast | 256 | 32 | 143,105 | 57.24 | 2.71 | 25.2 |
| fast | 512 | 8 | 121,781 | 33.63 | 1.68 | 22.3 |
| fast | 512 | 16 | 141,281 | 57.98 | 2.71 | 25.8 |
| fast | 512 | 32 | 143,693 | 114.02 | 4.72 | 26.3 |

## Speedups

| seq | batch | dense tok/s | fast tok/s | speedup |
|---:|---:|---:|---:|---:|
| 256 | 8 | 77,596 | 94,745 | 1.22x |
| 256 | 16 | 101,064 | 123,067 | 1.22x |
| 256 | 32 | 104,527 | 143,105 | 1.37x |
| 512 | 8 | 98,416 | 121,781 | 1.24x |
| 512 | 16 | 101,866 | 141,281 | 1.39x |
| 512 | 32 | 104,078 | 143,693 | 1.38x |

## What the numbers say

- The speedup improves with batch size, consistent with routing/gather overhead
  being amortized over more tokens.
- LiteMoD uses about **24% less peak VRAM** at 512/32 (4.72 GB vs 6.17 GB).
- The fast model's **effective TFLOP/s is lower** (26.3 vs 30.3 at 512/32).
  It is faster because it does less work, not because the routing kernels are
  more efficient. This is why the speedup is 1.38x rather than the 1.59x
  analytical ceiling.
- At 42M parameters the fixed LM head and small GEMMs are a large share of
  runtime. A 1B model changes this ratio; do not linearly extrapolate this
  speedup to 1B.
- A 1200-step quality check is the next step:
  `bash scripts/run_1200.sh`.
