# ConceptChat — a small, teachable baby-English dialogue model

A **proper small-scale test**, not a scripted demo:

- synthetic English dialogue corpus with 13 concepts;
- train/validation/held-out splits at both **template** and **slot-combination** level;
- concept-conditioned GRU encoder–decoder with attention and **copy**;
- correct `a`/`an` articles in color, animal, and want prompts;
- parameter-matched baseline without concept conditioning;
- test report with exact match, token F1, concept accuracy, and per-concept results;
- interactive chat for qualitative checking.

The model is trained from scratch on the synthetic corpus. It is **closed-world
baby English**, not an open-domain chatbot. It can answer only the concepts and
vocabulary it was taught.

## What it is taught

Concepts:

`greeting`, `name`, `feelings`, `politeness`, `colors`, `animals`,
`counting`, `prepositions`, `wants`, `likes`, `day`, `help`, `smalltalk`.

Grammar/behavior patterns include:

- short statements and questions;
- articles `a` / `an`;
- singular/plural (`one apple`, `three apples`);
- prepositions `on` / `in` / `under`;
- copying names, objects, colors, places, and numbers from the user context;
- polite replies;
- concept-conditioned reply selection.

**Canonical chat prompts and their slots are always in the training split.**
The test split holds out rarer paraphrases and slot combinations; otherwise
“hi”, “what is your name”, “the ball is red…”, and “i see three apples…” would
be held out and chat would look broken even when the compositional test works.

This was the main bug in the first GPU run: the split held out “hi” and
“what is your name”, so the chat model had never seen them. The report showed
95% on seen-template/unseen-slot but only ~5% on unseen templates. The revised
split keeps the canonical chat examples in train while still holding out
paraphrases and slot combinations for the measured test.

The context is part of the user message for information-bearing questions, e.g.:

```text
the ball is red. what color is the ball?
the cat is under the table. where is the cat?
i see three apples. how many apples?
```

This is intentional: “what color is the ball?” with no color in the message is
not answerable. The model must extract and copy the evidence.

## Model

- GRU encoder + GRU decoder with dot-product attention.
- Concept embedding added to the decoder initial state.
- Concept classifier trained jointly with the reply loss.
- **Pointer-generator copy** over the source message, so slot values can be
  copied rather than hallucinated.
- Greedy decoding with an optional repetition penalty in chat.
- ~0.8M parameters; fits easily on an RTX 3080 20GB.
- Parameter-matched baseline: same encoder/decoder/copy, no concept embedding or
  concept classifier.

## Proper held-out evaluation

Three test sets:

| split | what is held out |
|---|---|
| `test_seen_template_unseen_slots` | new slot combinations, seen sentence templates |
| `test_unseen_templates_seen_slots` | new paraphrases, seen slot values |
| `test_unseen_both` | new paraphrases and new combinations |

The report also has an **oracle concept** condition: generation with the true
concept label. This separates concept-classification errors from language errors.

No training pair appears in a test split (checked in `tests/`).

## Run on the RTX 3080

From the repo root:

```bash
git pull

# PyTorch CUDA is already in the Vast.ai image. This installs only PyYAML.
python -m pip install -r concept_chat/requirements.txt

# Train 3 seeds x {concept, baseline}; writes the held-out report.
python -u -m concept_chat.train \
  --config concept_chat/configs/small.yaml \
  --device cuda
```

Or use the wrapper:

```bash
bash concept_chat/scripts/train_vastai.sh
```

Outputs:

```text
concept_chat/runs/report.md
concept_chat/runs/results.json
concept_chat/runs/best.pt
concept_chat/runs/concept_seed*/best.pt
concept_chat/runs/baseline_seed*/best.pt
```

Expected runtime on a 3080: the model is ~0.8M parameters and the corpus is
~1.3k training examples; all six runs should finish in a few minutes (early
stopping is enabled). It is not an hours-long training job.

## Rerun the held-out test only

```bash
python -m concept_chat.evaluate \
  --checkpoint concept_chat/runs/best.pt \
  --device cuda
```

## Talk to it

```bash
python -m concept_chat.chat \
  --checkpoint concept_chat/runs/best.pt \
  --device cuda \
  --show-concept
```

Example messages:

```text
hi
what is your name
the ball is red. what color is the ball?
the cat is under the table. where is the cat?
i see three apples. how many apples?
what sound does a cow make?
i am sad
i want an apple
good night
```

Commands inside chat:

```text
:concept colors     force a concept
:auto               return to predicted concept
:concepts           list concepts
:quit
```

## How to read the result

The meaningful comparison is the **concept variant vs the baseline** on the
three held-out splits, not one cherry-picked chat reply. Good signs:

- concept variant beats baseline on unseen templates and/or unseen slots;
- oracle-concept exact match is materially higher than predicted-concept exact
  match, which would show the classifier is the bottleneck;
- copying works: replies contain the user's object/color/place/number rather
  than a different slot value;
- per-concept table exposes which concepts failed.

A low exact match with high token F1 usually means the reply is close but
copies the wrong slot value or uses a different paraphrase. The report includes
both numbers. The **concept accuracy** column tells you whether a failure is the
concept classifier or the language decoder.

In the first run, the 95% seen-template/unseen-slot result showed that copy and
slot composition worked. The 5% unseen-template result showed that the model
had not learned paraphrase grammar. The chat errors (“hi” → wants) were the
same split bug plus a weak no-slot concept classifier. This revision keeps the
canonical prompts in train, balances no-slot concepts, and fixes articles; rerun
the report before judging the new model.

The `test_unseen_templates_*` split is intentionally hard: the model must
produce a reply for a paraphrase it never saw. A low score there is a
paraphrase-generalization limitation, not necessarily a broken chat model —
the canonical chat prompts are in the training split. Use the concept and
oracle columns to separate the two issues.

## Limitations

- Closed synthetic world; no open-domain language understanding.
- Single-turn; no conversation history.
- The concept classifier is small and may confuse no-slot concepts.
- The model can only use words in its closed vocabulary; unknown chat words are
  mapped to `<unk>`.
- This is a baby-English prototype, not evidence of general intelligence or
  superiority to LLMs.
- No sleep consolidation or continual learning in this version.

## Files

```text
concept_chat/data.py        corpus, splits, vocabulary
concept_chat/model.py       concept seq2seq + attention + copy
concept_chat/train.py       training, held-out evaluation, report
concept_chat/evaluate.py    re-evaluate a checkpoint
concept_chat/chat.py        interactive single-turn chat
concept_chat/tests/         data/model tests
concept_chat/configs/       training config
```
