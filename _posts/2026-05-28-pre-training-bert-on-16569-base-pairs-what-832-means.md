---
title: "Pre-training BERT on 16,569 Base Pairs: What 8.32 Means"
date: 2026-05-28
tags: [mtDNA, foundation model, bioinformatics, machine learning, BERT]
layout: post
---

Before the model sees a single sequence, you already know what its loss will be. For a k-mer vocabulary of 4,102 tokens, the theoretical loss of a random model on the masked language modeling task is log(4,102) = 8.32. That is the number to beat.

The first smoke test confirmed it: initial MLM loss = 8.3244, against a theoretical random baseline of 8.3192. The difference is less than 0.005, which is what you expect from random weight initialization (not perfectly uniform, but close). Everything downstream follows from this anchor.

This is part of an open-source project to build the first dedicated foundation model for mitochondrial DNA. mtDNA mutations drive over 350 inherited diseases, including MELAS, Leigh syndrome, and Leber hereditary optic neuropathy. No sequence AI model designed specifically for the circular mitochondrial genome currently exists.

## The training problem

Pre-training a 6M-parameter BERT encoder on 152,484 mitochondrial genomes (117k vertebrate + 35k human) on a laptop requires solving two resource problems simultaneously: memory and compute.

The model processes 512-token windows at a time. With a hidden size of 256, 6 transformer layers, and 8 attention heads, a forward pass on a batch of 16 windows keeps peak memory at roughly 1.5GB. That fits. But batch size 16 is too small for stable MLM gradient estimates. BERT was trained with effective batches of 256-512.

Gradient accumulation solves this without adding memory. The update rule is:

```python
for _ in range(grad_accum_steps):  # 8 micro-steps
    loss = model(batch).loss / grad_accum_steps
    loss.backward()

optimizer.step()
optimizer.zero_grad()
```

This is an arithmetic identity, not an approximation. Summing 8 micro-gradients before an optimizer step is exactly equivalent to computing the gradient on a batch 8x larger. Physical batch = 16, gradient accumulation = 8, effective batch = 128. Memory cost of batch-16, gradient quality of batch-128.

## The learning rate schedule

Two distinct regimes govern the schedule:

**Warmup (steps 0-2000):** LR ramps linearly from 0 to 1e-4. Random initialization means early gradients are large and poorly directed. Full LR from step zero causes divergence. Two thousand warmup steps is conservative but safe on CPU where the first few steps take longer.

**Cosine decay (steps 2000-50000):** LR decays from 1e-4 to 1e-5 following a cosine curve. The decay prevents the model from overshooting near convergence. Cosine is preferred over linear decay because it is slow early (when the model is still moving quickly toward the loss minimum) and fast late (when fine precision matters).

```python
def _cosine_lr_with_warmup(step, warmup_steps, max_steps, base_lr, min_lr_fraction=0.1):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_fraction + (1.0 - min_lr_fraction) * cosine_factor
```

At step 2000: multiplier = 1.0 (peak). At step 50000: multiplier = 0.1 (floor). The expected loss at step 50k is 2.5-3.0, down from 8.32 at step 0.

## Two phases, not one

Phase 1 uses all 152k sequences (cross-species vertebrates + human). The heteroplasmy prediction head is disabled (`het_weight=0.0`) because cross-species sequences do not have per-position heteroplasmy data. The goal is to learn general mtDNA sequence structure.

Phase 2 loads Phase 1's encoder and retrains on human sequences only, with the heteroplasmy loss enabled (`het_weight=0.3`). This is where the model specializes to human mtDNA variation patterns.

The non-obvious part: Phase 2 must not resume the Phase 1 optimizer. AdamW maintains per-parameter running averages of the first and second gradient moments. After 50k steps on cross-species data, those moments are calibrated to the Phase 1 gradient landscape. Resuming them for Phase 2 means the optimizer's internal model of the gradient curvature is wrong for the new distribution. The corrective force it applies will oppose the direction the model needs to move.

The fix is a two-step load:

```python
def _load_checkpoint(self, path, encoder_weights_only=False):
    if encoder_weights_only:
        phase1_model = MtDNAForMaskedModeling.from_pretrained(str(path))
        phase1_state = phase1_model.state_dict()
        current_state = self.model.state_dict()
        for key, val in phase1_state.items():
            if key.startswith("mtdna.") and key in current_state:
                current_state[key] = val
        self.model.load_state_dict(current_state, strict=False)
```

Only keys starting with `mtdna.` (the encoder) are copied. The prediction heads (`kmer_prediction_head`, `het_prediction_head`) are not copied; they re-initialize fresh. The optimizer is never touched. Phase 2 starts with Phase 1's encoder knowledge and a clean optimizer state.

## What took longer than expected: vocabulary inference

The trainer builds a KmerVocabulary at setup time. The k-mer size k needs to match the model's vocabulary. The obvious approach is to hardcode `k=6` in the trainer. That breaks immediately in tests, which use a tiny 3-mer vocabulary (64 k-mers + 6 special = 70 tokens).

The solution is to derive k from vocab_size:

```python
@staticmethod
def _infer_k_from_vocab_size(vocab_size: int, n_special: int = 6) -> int:
    n_kmers = vocab_size - n_special
    k = round(math.log(n_kmers, 4))
    if 4**k != n_kmers:
        raise ValueError(f"vocab_size={vocab_size} does not match 4^k + {n_special}")
    return k
```

`vocab_size=4102` gives `log₄(4096) = 6.0`, k=6. `vocab_size=70` gives `log₄(64) = 3.0`, k=3. The trainer is now vocabulary-agnostic and the 21 unit tests that use k=3 all pass without mocking anything.

This pattern generalizes: any foundation model trainer that works on multiple vocabulary sizes should derive k (or the equivalent parameter) from the model config rather than hardcoding it.

## Pre-tokenization: upfront cost, zero per-step cost

`MtDNADataset` tokenizes all sequences at construction time. For 152k sequences at 16,569 bp each with k=6, this takes 2-3 minutes. Every training step thereafter does only window index lookups, not re-tokenization.

The tradeoff is straightforward: 2 minutes of upfront cost vs. 50,000 steps of training. If tokenization cost 1ms per sequence, doing it per-step would add 16ms overhead to every batch of 16 sequences, or about 800 seconds total. Pre-tokenizing is 2 minutes vs. 13 minutes saved. The only cost is memory: 152k sequences × 16,569 tokens × int64 = ~20GB for a naive implementation, which is why the dataset stores token IDs as Python lists (not tensors) and constructs window tensors on demand in `__getitem__`.

## MLflow and the `finally` pattern

Every training run writes to an MLflow experiment. The setup is four lines:

```python
mlflow.set_experiment(self.mlflow_experiment)
mlflow.start_run()
mlflow.log_params({...})
# ... training loop ...
mlflow.end_run()
```

The important detail: `mlflow.end_run()` must be in a `finally` block. If training crashes at step 23,000, the MLflow run needs to be marked complete (or failed) rather than left as `RUNNING` indefinitely. Without `finally`, the next `mlflow ui` session shows the crashed run as active, which makes it invisible in comparison views that filter to completed runs.

## Gradient checkpointing

The model has 6 transformer layers. A standard forward pass caches all intermediate activations for the backward pass. For batch=16, seq=512, hidden=256, that is roughly 16 × 512 × 256 × 6 × 4 bytes = 800MB of activation memory, on top of the 24MB for model weights.

`gradient_checkpointing_enable()` removes the cached activations and recomputes them during backprop. Memory use drops by roughly half. Compute time increases by roughly 30%. On a laptop with 16GB RAM, the tradeoff is not optional.

## Current state and expected convergence

The Phase 1 trainer is ready. 174 tests pass. The verified numbers from the smoke test:

- Initial MLM loss: **8.3244** (random baseline: 8.3192)
- Training speed: **6-7 steps/s on CPU** with batch=4, accum=2
- Model size: **~6M parameters** (production config)

Expected Phase 1 convergence:
- Step 5,000: loss 5.5-6.0
- Step 20,000: loss 3.5-4.0
- Step 50,000: loss 2.5-3.0

A loss of 2.5 at step 50k corresponds to perplexity = e^2.5 = 12.2, meaning the model narrows 4,102 possible k-mers down to roughly 12 plausible candidates in context. Whether that is good enough for downstream tasks (haplogroup classification, variant pathogenicity) is what Phase 2 and fine-tuning will measure.

## Key takeaways

- The random baseline loss for MLM is exactly log(vocab_size). Verifying this at step 0 is a necessary sanity check before any training run; a deviation indicates a bug in masking, label construction, or the loss function.
- Gradient accumulation is an arithmetic identity: summing N micro-step gradients is exactly equivalent to computing the gradient on an N-times-larger batch, with no memory overhead beyond one additional scalar division per micro-step.
- Phase 2 domain-adaptive pre-training must start with a fresh optimizer. Resuming Phase 1 optimizer moments applies cross-species gradient curvature estimates to human-specific data, resisting adaptation at the most critical early steps.
- Deriving k-mer size from vocab_size (`k = log₄(vocab_size - n_special)`) makes the trainer vocabulary-agnostic and enables unit testing with small synthetic vocabularies without any mocking.