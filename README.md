# Memento-GPT

A small, from-scratch GPT-style language model trained on a corpus of Christopher
Nolan screenplays (Memento, Inception, The Prestige, Interstellar). It learns the
*structure and voice* of a screenplay — scene headings, character cues, dialogue,
action lines — and generates new scene-like text in that style.

This is an educational project built up from Andrej Karpathy's nanoGPT-style
tutorial, but with a real BPE tokenizer, a data-cleaning pipeline, mixed-precision
training, a cosine learning-rate schedule, early stopping, and best-checkpointing.

> **Note on output quality:** with a small (~300k-token) corpus and a ~3.4M-parameter
> model, the generated text is *format-correct but not coherent* sentence-to-sentence.
> That's the expected ceiling for this data scale, not a bug. See [Results](#results).

---

## Architecture

A decoder-only Transformer (GPT), implemented in [memento-gpt.py](memento-gpt.py):

- Token + learned positional embeddings
- A stack of Transformer `Block`s, each = pre-norm multi-head self-attention + feed-forward, with residual connections
- Final LayerNorm + linear head projecting to next-token logits

### Default hyperparameters

| Hyperparameter | Value | Notes |
|---|---|---|
| `n_embd` | 192 | embedding dimension |
| `n_head` | 6 | attention heads (head size = 32) |
| `n_layer` | 4 | Transformer blocks |
| `block_size` | 128 | context length |
| `dropout` | 0.15 | tuned (see [Results](#results)) |
| `batch_size` | 64 | |
| `learning_rate` | 3e-4 | peak LR (cosine + warmup) |
| `max_iters` | 5000 | upper bound; early stopping usually halts sooner |
| vocab size | 4096 | SentencePiece BPE |

~3.4M parameters total. Deliberately small to match the fixed dataset size — a larger
model overfits this corpus almost immediately.

### Training features

- **Mixed precision** (bfloat16 autocast) + **TF32** matmuls on GPU
- **Gradient clipping** at 1.0
- **Cosine LR schedule with linear warmup** (the transformer-idiomatic schedule)
- **Early stopping** with patience on validation loss
- **Best-val checkpointing** — saves `model.pt` only when validation improves, then
  reloads the best checkpoint before generating (never the overfit final step)

---

## Tokenizer

A **SentencePiece BPE** tokenizer (4096 tokens) trained on the corpus, chosen over:

- *Character-level* — no production LLM uses it; subword tokens let the model start from word-pieces.
- *tiktoken* — can't train a custom vocab, and its 50k–200k web-scale vocabs are wasteful on a small domain corpus.

Configured with `byte_fallback=True` (lossless round-trip on any input) and a vocab
sized to the corpus. Trained once to `tok.model` / `tok.vocab`, then reused.

---

## Setup

Requires Python 3.x and an NVIDIA GPU is recommended (CPU works but is slow).

```powershell
# 1. create and activate a virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. install PyTorch (CUDA build — adjust the index for your GPU/driver)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 3. install the rest
pip install sentencepiece numpy
```

> The `cu130` build above targets a recent NVIDIA (Blackwell) GPU. Pick the CUDA tag
> that matches your hardware from the [PyTorch index](https://download.pytorch.org/whl/).

---

## Data pipeline

1. **Acquire scripts** → assembled into `input.txt`.
2. **Clean** the raw text with [clean.py](clean.py): strips production/pagination noise
   (revision footers, page numbers, `(CONTINUED)` markers, scene-number gutters,
   revision asterisks, OCR glitches) while keeping the real screenplay signal. It
   overwrites `input.txt` in place and is idempotent.
   ```powershell
   python clean.py
   ```

---

## Training & generation

Everything lives in one script:

```powershell
python memento-gpt.py
```

On the **first run** it trains the tokenizer (if missing), trains the model, saves the
best checkpoint to `model.pt`, and prints a generated sample. On **later runs**, if
`model.pt` exists it loads the trained model and skips straight to generation.

> **Important:** if you change the model architecture *or* a training hyperparameter
> (e.g. `dropout`, `n_embd`), delete the old checkpoint first so it retrains:
> ```powershell
> Remove-Item model.pt
> ```
> Architecture changes cause a load error; hyperparameter-only changes (like dropout)
> would otherwise *silently* load the old weights and skip training.

Sampling uses `temperature` and `top_k` (see the `generate(...)` call at the bottom of
the script) and seeds from `"INT."` so output opens on a scene heading.

---

## Results

The dataset is small and fixed, so the model overfits if it has too much capacity. The
final config was reached empirically:

- **Best validation loss ≈ 5.09** (cross-entropy / nats).
- A **dropout sweep** found a broad, flat minimum: `0.4 → 5.142`, `0.3 → 5.111`,
  `0.2 → 5.092`, `0.15 → 5.086`. Improvements flatten into noise below 0.2; the
  binding constraint is **corpus size**, not regularization.
- Early stopping typically halts around step ~3000–3400, well before `max_iters`.

Generated samples reproduce screenplay formatting (scene slugs, character cues,
`(V.O.)`, `CUT TO:`) and the correct character/setting vocabulary per film, but are not
coherent at the sentence or plot level — the expected ceiling at this scale.

---

## Project structure

```
memento-gpt.py   # model, tokenizer, training loop, generation
clean.py         # corpus cleaning pass (run before training)
input.txt        # training corpus (not committed)
tok.model        # trained SentencePiece tokenizer (generated)
tok.vocab        # human-readable vocab (generated)
model.pt         # best model checkpoint (generated)
```

Generated artifacts (`model.pt`, `tok.model`, `tok.vocab`, `input.txt`, `.venv/`)
should be in `.gitignore` — they're build outputs, not source.

---

## Data & licensing note

**This is a personal project I built purely for my own education** — to learn how small
language models, tokenizers, and the training pipeline work end-to-end. It is not a
product and is not intended for commercial or public use.

The training corpus is assembled from **copyrighted screenplays**, used here only for
that private, educational purpose. The screenplay text is **not included in this
repository** and must not be redistributed. The trained model weights are a **derivative
work** of that copyrighted text, so they are **not redistributed** either — anyone
reproducing this project should supply their own corpus. Rights-cleared alternatives:
public-domain text from Project Gutenberg, openly-licensed datasets, or your own writing.

The **source code** is mine to license freely; it's the **data and the trained weights**
that carry the restrictions.
