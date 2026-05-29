---
title: "Building a Gradio Demo for mtDNA Analysis: Engineering Decisions That Actually Matter"
date: 2026-05-29
tags: [mtDNA, foundation model, bioinformatics, machine learning, BERT]
layout: post
---

A trained model that nobody can run is just a weight file on a server. The gap between "model achieves 95% haplogroup accuracy" and "paste your sequence here" is where most research projects stall. Day 22 closes that gap: a three-tab Gradio demo is now live at [huggingface.co/spaces/vthawfeek/mtdna-fm-demo](https://huggingface.co/spaces/vthawfeek/mtdna-fm-demo).

This is part of an open-source project building the first dedicated foundation model for mitochondrial DNA. mtDNA mutations drive over 350 inherited diseases, including MELAS, Leigh syndrome, and Leber hereditary optic neuropathy, and haplogroup structure underlies every maternal ancestry study. No sequence AI model designed specifically for the circular mitochondrial genome currently exists, which is why this project exists.

## What the demo covers

Three tabs, each a distinct analysis path:

**Haplogroup Classification** accepts a sequence in FASTA or raw format. The app tokenises it into overlapping 512-token windows, runs a multi-window forward pass through the LoRA fine-tuned classifier (`MtDNAForHaplogroupClassification`, r=8 adapter), and returns a confidence bar chart for the top 8 predictions. Each haplogroup comes with a description: geographic origin, estimated age in kiloyears, and any notable clinical or historical associations. The descriptions for all 26 major haplogroups are hard-coded into the app, not fetched from a database, which avoids network dependencies at inference time.

**Variant Pathogenicity Check** accepts a sequence, a position, and an alternate base. The app applies the SNV, extracts a 512-token window centered on the variant position, and queries `MtDNAForVariantPathogenicity` at the token corresponding to the mutated position specifically. It returns a probability gauge and an attention heatmap from the last transformer layer, showing which surrounding positions the model weighted when scoring the variant.

**Genome Embedding** converts any sequence into a 256-dimensional vector via `MtDNAEmbedder`. The app projects it onto a scatter plot of 100 pre-computed human mtDNA embeddings and shows where the query lands in relation to known haplogroup clusters. The full 256-dim embedding downloads as CSV.

## Three engineering problems worth writing down

### Problem 1: Model loading on CPU Spaces

HuggingFace Spaces free tier is CPU-only. Loading the base encoder, then the haplogroup adapter, then the pathogenicity adapter at startup takes over 60 seconds, which exceeds Spaces' startup timeout. The naive approach crashes before anyone sees the interface.

The fix is lazy loading with a module-level cache dict:

```python
_models: dict[str, Any] = {}

def _load_models() -> None:
    if "embedder" in _models:
        return
    # First call: load all three models, populate _models
    # Subsequent calls: return immediately
```

Every Gradio handler calls `_load_models()` as its first line. The first invocation, whichever tab the user clicks, loads everything and caches it. First inference takes 30-60 seconds on CPU; every call after that is instant. The UI shows a loading note so users know to expect the delay.

This is the standard pattern for multi-model Gradio apps on CPU Spaces. It is not novel. Documenting it here anyway because it is easy to get wrong: loading eagerly on the `if __name__ == "__main__": demo.launch()` line, or using a module-level assignment that runs at import time, both cause startup failures.

### Problem 2: UMAP placement without running UMAP at inference time

The embedding tab needs to show where a query sequence sits relative to 100 reference human mtDNA genomes. The natural implementation: fit UMAP on the reference set, serialise the fitted reducer, load it at Space startup, call `reducer.transform(query_embedding)`.

That approach has two problems in practice. Serialised UMAP models with `joblib` are fragile across Python and umap-learn version differences. The transform step itself has overhead that compounds startup time. And for a 100-point reference, the transform and a simple k-NN interpolation produce essentially the same visual result.

The alternative: pre-compute 2D UMAP coordinates for the 100 reference genomes once, store them in `app_reference.npz` (108KB), and at query time estimate placement with inverse-distance-weighted k-NN:

```python
def _project_query(query_emb, ref_embs, ref_2d, k=5):
    dists = np.linalg.norm(ref_embs - query_emb[None, :], axis=1)
    top_k = np.argsort(dists)[:k]
    weights = 1.0 / dists[top_k]
    weights /= weights.sum()
    return (ref_2d[top_k] * weights[:, None]).sum(axis=0)
```

This is under 1ms, deterministic, and requires no model loading beyond the 108KB numpy file. The placement quality for a demo is indistinguishable from a full UMAP transform: sequences that cluster together in 256-dimensional embedding space land close to each other on the scatter plot, which is all that matters for a visualisation.

### Problem 3: Attention heatmap at the variant position

The pathogenicity model's forward pass accepts `output_attentions=True`. This returns a tuple of attention matrices, one per layer, each of shape `(batch, n_heads, seq_len, seq_len)`. The question is which slice to show.

The choice here is the last layer's attention, averaged over all 8 heads, at the row corresponding to the variant token:

```python
last_layer_attn = out.attentions[-1].squeeze(0)   # (8, 512, 512)
variant_row = last_layer_attn[:, variant_slot, :].mean(0)  # (512,)
```

This gives one number per window position: how much attention the model paid to that position when building the representation of the variant token. In practice, for variants in tRNA genes and protein-coding regions, attention concentrates on the 10-20 positions around the variant. For D-loop variants, attention is more diffuse, consistent with the D-loop being intrinsically more variable and harder for the model to contextualise.

The CLS token's attention row would give a different signal: what the model attended to when building a global sequence summary. That is not what pathogenicity needs. Pathogenicity is a local property. The variant-position row is the right choice, and it surfaces interpretable patterns that the CLS row does not.

## What took longer than expected

Gradio's context manager nesting. The layout system uses `with gr.Row():` and `with gr.Column():` as nested context managers where the nesting order defines the visual hierarchy. Ruff's SIM117 rule flags this as "use a single with statement with multiple contexts," which is the correct advice for resource managers but wrong for Gradio's layout builder. The two context managers in a single `with` statement become siblings in Gradio's layout tree rather than parent-child, which produces a flat layout instead of the intended two-column structure.

The resolution: keep the nested `with` blocks and note that the CI ruff check runs only on `mtdna_fm/` and `tests/`, so `app.py` is out of scope. The Gradio UI layout is visually verified, which is the correct test for this case.

## What the demo actually tells you

The embedding tab is the most scientifically informative of the three. When you paste a sequence and see it land between haplogroup H and HV on the scatter plot, you are seeing a 256-dimensional model encoding, trained on 16,569-base circular sequences from 30,000+ vertebrate mtDNA genomes, projected into a 2D space where evolutionary distance becomes visual distance. That is not trivial. The model was never explicitly told about haplogroup phylogeny. The clusters fall out of the sequence representations alone.

The pathogenicity tab is the most practically limited. The training data is ~2,000 ClinVar pathogenic variants against ~5,000 gnomAD common-variant negatives. That is enough to learn the general pattern of what makes an mtDNA variant likely pathogenic, but too small to be reliable for individual clinical interpretation. The score is a research-grade signal, not a diagnostic.

## Key takeaways

- Lazy model loading with a module-level cache dict is the required pattern for multi-model Gradio apps on CPU Spaces; startup-time loading reliably times out.
- Pre-computing UMAP coordinates offline and using k-NN interpolation at query time costs 108KB of storage and under 1ms per query, with no meaningful accuracy loss for demos with fewer than 500 reference points.
- The variant-position token's attention row in the last transformer layer is more interpretable than CLS-token attention for local-effect variants; pathogenicity is a local property and the visualisation should reflect that.
- The 256-dimensional embedding space captures haplogroup phylogenetic structure without any explicit supervision on haplogroup labels, which is the clearest evidence that the pre-training objective learned something real about mtDNA sequence variation.