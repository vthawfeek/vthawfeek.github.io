---
title: "mtDNA-FM is Live on HuggingFace Hub: What a Research-Grade Model Release Actually Requires"
date: 2026-05-31
tags: [mtDNA, foundation model, bioinformatics, machine learning, BERT]
layout: post
---

The model weights are on HuggingFace Hub. Three repositories: the base encoder, a haplogroup classification adapter, and a pathogenic variant prediction adapter. Anyone can load the base model in three lines of Python. That sentence sounds simple, but the gap between "the model works locally" and "anyone can use it from the Hub" took more engineering than I expected.

This is part of an open-source project to build the first dedicated foundation model for mitochondrial DNA. mtDNA mutations cause over 350 inherited diseases, including MELAS, Leigh syndrome, and Leber hereditary optic neuropathy, and are the basis for maternal ancestry tracing. No sequence AI model designed specifically for the circular mitochondrial genome existed before this project.

## The Storage Economics of LoRA Adapters

The base model is 44.6 MB. The haplogroup classification adapter is 400 KB. The pathogenicity adapter is 203 KB.

That ratio matters. The haplogroup adapter has `r=8`, which means LoRA injects rank-8 low-rank matrices into the query, key, value, and dense layers of each attention block. Total trainable parameters: about 500K out of 6.9M (full model with MLM head). The pathogenicity adapter has `r=4` (half the rank, half the size) because the training dataset is smaller and heavier regularisation is needed.

The practical consequence: if you fine-tune a new downstream task from this base model, you get a 203-400 KB artefact that users download on top of the 44.6 MB base they already have. You can update, retrain, or replace an adapter without redistributing the base. For a research group that wants to share a haplogroup classifier trained on their own in-house cohort data, this is the right deployment pattern.

## A Bug That Was Already in the Codebase

The original `push_to_hub.py` script was wrong in a way that would have been hard to notice. It created fresh LoRA adapters (random weights, untrained) and pushed those to the Hub instead of the actual trained adapters from the fine-tuning runs.

The script loaded the base encoder, instantiated a new `MtDNAForHaplogroupClassification` with that encoder, applied a LoRA config, and called `save_pretrained`. No training, no checkpoint loading. The adapters would have loaded without error — PEFT doesn't check whether weights are meaningful — but predictions would be random.

The fix: load the adapter files directly from the fine-tuning output directories (`finetune_haplogroup_v1`, `finetune_pathogenicity_v1`) and upload those files. The only transformation needed before upload is patching the `base_model_name_or_path` field in `adapter_config.json`.

```python
# What the script was doing (wrong):
peft_model = get_peft_model(task_model, lora_config)  # fresh random weights
peft_model.save_pretrained(output_dir)

# What it should do (right):
shutil.copy(source_dir / "adapter_model.safetensors", tmp_dir / "adapter_model.safetensors")
config = json.loads((source_dir / "adapter_config.json").read_text())
config["base_model_name_or_path"] = "vthawfeek/mtdna-foundation-model"  # patch Hub path
(tmp_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
```

The config patching step is necessary because PEFT saves the `base_model_name_or_path` as a local filesystem path during training. If you upload that file unchanged, a user who loads `PeftModel.from_pretrained(model, "vthawfeek/mtdna-fm-haplogroup")` will get an error when their local system doesn't have `models/phase1_v1` at that exact path. The patched version points to the Hub model ID, which resolves correctly regardless of local layout.

## AutoConfig and Custom Architectures

When the verification step runs `AutoConfig.from_pretrained("vthawfeek/mtdna-foundation-model")`, it fails with a warning:

```
The checkpoint you are trying to load has model type `mtdna_fm` but Transformers
does not recognize this architecture.
```

This is expected and not an error. `model_type: mtdna_fm` is a custom value set in `MtDNAConfig`. HuggingFace's `AutoConfig` only recognises architectures registered in the Transformers library. Custom architectures (which is most domain-specific models) will always produce this warning.

The correct loading path is through the concrete class:

```python
from mtdna_fm.model.model import MtDNAForMaskedModeling

model = MtDNAForMaskedModeling.from_pretrained("vthawfeek/mtdna-foundation-model")
# 6,907,393 parameters (encoder + MLM head) — loads correctly
```

The verification script confirmed this works. The model loads all 113 weight tensors from the Hub and reproduces the correct parameter count (6.9M full model; the encoder alone is ~5.8M). The AutoConfig warning is documented in the model card limitations section rather than silently ignored.

## The Model Card as Documentation

The model card is the first thing a user sees on the Hub page. For most uploaded models it's a template placeholder. For a research-grade release it should function as documentation.

What the mtDNA-FM model card includes:

**Architecture novelties with equations.** The circular positional encoding formula is written out:

```
PE[pos, 2i]   = sin(2π × pos / L × 1/10000^(2i/d))
PE[pos, 2i+1] = cos(2π × pos / L × 1/10000^(2i/d))
```

Anyone reading the card should understand why this differs from standard sinusoidal PE and what biological fact it encodes (the circular topology of mtDNA).

**Performance table with baselines.** Not just the fine-tuned numbers:

| Task | Majority class | k-mer freq PCA+LR | mtDNA-FM (zero-shot) | mtDNA-FM (fine-tuned) |
|------|---------------|-------------------|---------------------|----------------------|
| Haplogroup (26-class) | ~15% | ~65% | 50% | 60.8% |
| Pathogenicity AUROC | 0.50 | ~0.72 | — | 0.877 |

The zero-shot 50% haplogroup accuracy (vs 12.5% random baseline on the sampled 8-class subset) is the most honest indicator of whether pre-training learned something useful. A model that scores 50% on a task it was never explicitly trained for is capturing real biological structure from sequence alone.

**Limitations stated plainly.** HmtDB has a European population bias. Haplogroup H is overrepresented. Performance on underrepresented African L sub-haplogroups is unknown. The heteroplasmy channel was architecturally present in Phase 2 training but real per-base heteroplasmy labels were limited to gnomAD variant-level data, not full-genome measurements. These are in the card because users need to know them to decide whether the model is appropriate for their use case.

**Ancient DNA zero-shot result.** Neanderthal and Denisovan sequences were embedded without fine-tuning. L2 distance from modern humans: 1.48x (Neanderthal) and 1.43x (Denisovan) the modern pairwise baseline. Consistent with what paleoanthropology expects from molecular phylogenetics, but derived purely from learned sequence representations.

## What "Three Lines" Actually Means

The plan for this project said the usage example should be five lines. It ended up being three:

```python
from mtdna_fm.inference.api import MtDNAEmbedder

embedder = MtDNAEmbedder.from_pretrained("vthawfeek/mtdna-foundation-model")
embedding = embedder.embed_genome(my_sequence)   # shape: (256,)
```

Getting to three lines required the `MtDNAEmbedder` API built in Day 15: a class that hides the windowing logic, handles the 512-token context window over a 16,569-token genome, mean-pools the CLS token across windows, and returns a single (256,) numpy array. The user doesn't need to know that the genome is chunked into overlapping 512-token windows or that the CLS token is extracted per window.

That API decision was made weeks before the Hub push, but it's what makes the Hub release usable rather than just available.

## What the Hub Release Actually Represents

Three repositories, two days of engineering, one working push script. But the value isn't in the upload mechanics. It's in having:

1. A base model that anyone can fine-tune for a new mtDNA task without training from scratch
2. Adapter files that represent actual trained weights, not placeholders
3. A model card that tells users the architecture, the performance honest to baselines, and the limitations
4. A verified loading path that works from a fresh environment

The Gradio demo (Day 22) will be the interactive surface on top of this. But the Hub release is the foundational artefact — the thing that makes all the downstream work reusable.

## Key takeaways

- LoRA adapter deployment separates the base model (44.6 MB, shared once) from task-specific weights (203-400 KB each); updating or replacing a downstream task costs less than 1% of the base model's storage.
- PEFT saves `base_model_name_or_path` as a local filesystem path during training; this must be patched to the Hub model ID before upload or the adapter will fail to load outside the training environment.
- Zero-shot 50% haplogroup accuracy (vs 12.5% random) on a task the model was never explicitly trained for is a cleaner signal of learned biological structure than any fine-tuned number.
- A model card that includes architecture equations, honest baselines, and stated limitations is the minimum bar for a research-grade release; anything less is a weight dump.

---

*mtDNA-FM is open source: [github.com/vthawfeek/mtdna-foundation-model](https://github.com/vthawfeek/mtdna-foundation-model) | Model on Hub: [huggingface.co/vthawfeek/mtdna-foundation-model](https://huggingface.co/vthawfeek/mtdna-foundation-model)*
