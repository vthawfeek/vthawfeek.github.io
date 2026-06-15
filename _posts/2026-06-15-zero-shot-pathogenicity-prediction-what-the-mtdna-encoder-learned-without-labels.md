---
title: "Zero-Shot Pathogenicity Prediction: What the mtDNA Encoder Learned Without Labels"
date: 2026-06-15
tags: [mtDNA, foundation model, bioinformatics]
layout: post
---

I built a pathogenicity predictor for mitochondrial DNA variants. The LoRA fine-tuning adapter exists — but it was trained on synthetic pathogenicity labels, a placeholder for real ClinVar and gnomAD data that was never assembled during the project sprint. Whether that adapter generalises to real variants is an open question.

But before fine-tuning there is a more interesting question: does the pre-trained encoder already know something about pathogenicity? The encoder was trained on real cross-species vertebrate mtDNA through masked language modeling — no pathogenicity labels, no disease databases, just sequence. The test: embed real ClinVar pathogenic variants and real gnomAD common variants using the pre-trained encoder alone, then ask whether the embeddings separate.

**AUROC = 0.777** (95% CI: 0.731–0.821).

This post is about what that number means, where it comes from, and why a pre-trained encoder trained purely on masked language modeling over vertebrate mitochondrial sequences would produce it without ever seeing a pathogenicity label.

<img src="http://rokpayprsizors.files.wordpress.com/2026/06/t6.png?w=1200" alt="Zero-shot pathogenicity prediction: AUROC 0.777 from a pre-trained encoder with no pathogenicity labels, evaluated on ClinVar pathogenic variants vs gnomAD common variants." style="max-width:100%;height:auto;" />

---

## The variant-token architecture

`MtDNAForVariantPathogenicity` wraps the pre-trained mtDNA-FM encoder with a binary classification head that predicts whether a point mutation is pathogenic or benign.

The central architectural decision is which hidden state to use for classification. The obvious choice is the `[CLS]` token, which accumulates global sequence context across the full 16,569 bp genome. That's how most sequence classifiers work. It's the wrong choice here.

Pathogenicity is a local property. The m.3243A>G mutation in the MT-TL1 tRNA gene causes MELAS encephalomyopathy because of what happens at that specific locus, not because of some aggregate property of the full mitochondrial genome. A `[CLS]`-based classifier must learn to extract local variant effects from a global summary vector — asking the architecture to do work it isn't built for.

The variant-token approach: extract the hidden state at the token position that covers the variant. For a mutation at position 3,243, identify which 6-mer token spans that position and pull its 256-dimensional hidden state directly from the encoder output. Project that through a linear classifier.

```python
# Extract hidden state at variant position
hidden = encoder(input_ids, position_ids, attention_mask).last_hidden_state  # (B, L, D)
variant_hidden = hidden[torch.arange(B), variant_positions]                   # (B, D)
logits = self.classifier(variant_hidden)                                       # (B, 1)
```

The inductive bias: the encoder's attention mechanism, operating over the full genome with circular positional encoding, has already assembled a contextualised representation of the k-mer at the mutation site. By the time a variant position's hidden state reaches the classifier, it has attended to surrounding sequence context — nearby tRNA structural elements, conserved coding regions, OXPHOS gene context. The classifier reads that representation directly rather than trying to recover it from a global summary.

This design is not specific to mtDNA. Any task requiring prediction of a single-position change in a longer sequence context can use this approach: splice-site mutations in nuclear DNA, promoter variants, single amino acid substitutions in protein language models. The variant-position hidden state carries local context enriched by global attention — it's a broadly applicable pattern for variant effect prediction in foundation models.

---

## The zero-shot evaluation

Rather than fine-tuning on labeled pathogenicity data, I tested a simpler question first: do the pre-trained encoder's variant-position embeddings already separate pathogenic from benign variants, without any supervised pathogenicity signal?

**Data sources:**

- **Pathogenic:** 118 mitochondrial SNPs from ClinVar (GRCh38) with clinical significance annotated as Pathogenic, Likely_pathogenic, or Pathogenic/Likely_pathogenic.
- **Benign proxies:** 419 mitochondrial variants from gnomAD v3.1 with population allele frequency ≥ 1% in 56,434 individuals. Variants at AF ≥ 1% face strong purifying selection against severe pathogenicity — the same rationale used by CADD and other tools to define negative controls.

**Embedding method:** For each variant, the alt allele was applied to the rCRS reference (NC_012920.1), then `MtDNAEmbedder.embed_variant()` extracted the 256-dimensional hidden state at the variant position token from the pre-trained Phase 1 encoder. No fine-tuning. No pathogenicity labels during training. Pure zero-shot.

**Evaluation:** 5-fold stratified k-NN (k=5, cosine distance). Out-of-fold probability scores aggregated before computing AUROC — more stable than averaging per-fold AUROCs on small datasets.

**Results:**

| Metric | Value | Random baseline |
|--------|-------|-----------------|
| AUROC | 0.777 (95% CI: 0.731–0.821) | 0.500 |
| AUPRC | 0.440 | 0.220 |
| n_pathogenic | 118 | — |
| n_benign | 419 | — |

The AUPRC of 0.440 versus a random baseline of 0.220 (= 118/537) represents a 2× improvement in precision-recall area. CADD, SIFT, and PolyPhen-2 achieve AUROC 0.7–0.9 on their held-out benchmarks, but those are trained tools with thousands of labeled examples. The comparison is to give a sense of scale: zero-shot at 0.777 is within the range of established supervised methods, without a single pathogenicity label.

---

## Per-variant-type breakdown

| Variant type | AUROC | n_pathogenic | n_benign | Note |
|---|---|---|---|---|
| Missense | 0.727 | 56 | 248 | Most reliable |
| tRNA | 0.718 | 44 | 21 | High AUPRC = 0.773 |
| rRNA | 0.639 | 5 | 36 | Low power (n=5 positives) |
| D-loop | — | 1 | 109 | Insufficient data |
| Other (intergenic) | — | 12 | 5 | Inverted class balance |

**Missense and tRNA are the reliable categories.** Missense is the largest (304 total variants), well-balanced, and shows clear signal. tRNA is particularly interesting: the AUPRC of 0.773 means the model achieves high precision at low recall thresholds — it's most confident when it predicts tRNA variants pathogenic, and it's right. This makes biological sense: tRNA secondary structure constraints are highly conserved across vertebrates, and disruption of the anticodon stem or T-loop is a well-established disease mechanism (MELAS, MERRF, CPEO).

**D-loop and "other" are not interpretable.** ClinVar contains almost no pathogenic D-loop variants — the control region is hypervariable by design and pathogenic mutations there are rare. With 1 positive versus 109 negatives, any AUROC number is statistically undefined. The "other" category (intergenic boundary positions) has an inverted class ratio (12 pathogenic, 5 benign) that reflects the category's definition, not the model's behaviour. These should not be read as negative results.

**rRNA is low-powered.** With 5 pathogenic variants against 36 benign, the AUROC=0.639 is indicative but noisy.

---

## Why an MLM-trained encoder knows anything about pathogenicity

The pre-trained encoder was trained entirely on masked language modeling — predicting masked 6-mer tokens from their surrounding context. No pathogenicity labels. No disease databases. Just sequence.

The mechanism that explains the zero-shot result: **evolutionary constraint**.

Pathogenic mitochondrial variants are pathogenic because they disrupt function at positions that evolution has maintained under purifying selection. The m.3243A>G in MT-TL1, the m.11778G>A in MT-ND4, the m.1555A>G in MT-RNR1 — these mutations are at positions conserved across vertebrate mitochondrial genomes because they're essential for tRNA aminoacylation, NADH dehydrogenase function, or ribosomal RNA structure.

The encoder, trained on 30,000+ vertebrate mitochondrial sequences, learned to represent these positions differently from variable ones. At a conserved position, the local sequence context is highly predictable — the 6-mer there looks the same across bats, tuna, and humans — and the encoder builds a representation that reflects that signal. When a pathogenic alt allele substitutes at a constrained position, the resulting representation lies in a distinct region of embedding space from a variant at a high-frequency (low-constraint) position.

This is the same mechanism validated in protein language models. ESM and ProtT5 zero-shot mutation effect scores correlate with experimental fitness measurements because evolutionary conservation is a proxy for functional importance. We're seeing the same pattern in a sequence-specific domain: mitochondrial genome language modeling captures the constraint signatures that predict clinical pathogenicity.

Two caveats to be honest about:

1. **ClinVar ascertainment bias.** The 118 pathogenic variants in our set are the ones severe enough to reach clinical annotation — MELAS, LHON, MERRF, Leigh syndrome. These are high-effect-size variants at highly conserved positions. A more comprehensive set including mildly pathogenic or VUS-upgraded variants would be harder, and AUROC=0.777 likely represents an upper bound.

2. **Benign proxy limitations.** AF≥1% in gnomAD is a strong signal against severe pathogenicity, but mildly deleterious variants can reach 1% frequency in finite populations. This is standard practice (shared by CADD and similar tools) but worth acknowledging.

---

## What a complete evaluation would add

The zero-shot result establishes the baseline: the pre-trained encoder's representations already carry pathogenicity signal. What a proper supervised evaluation adds:

**Calibration.** Does a zero-shot k-NN score of 0.8 correspond to 80% pathogenic in the neighborhood? Probably not — k-NN scores are discrete and coarse ({0, 0.2, 0.4, 0.6, 0.8, 1.0} from a 5-NN). A fine-tuned classifier with a sigmoid head would have better-calibrated probabilities.

**VUS discrimination.** ClinVar's variants of uncertain significance are the clinically relevant class. A model that separates clearly pathogenic from clearly benign might still fail on VUS — the real test. This requires VUS-labeled data with subsequent reclassification outcomes.

**Allele frequency stratification.** The zero-shot evaluation mixes common benign variants (AF 1–50%) with all pathogenic variants. Stratifying by AF bin would show whether the model performs differently against rare benign variants versus common ones.

**Fine-tuning upper bound.** LoRA r=4 fine-tuning on the ClinVar/gnomAD training split would show how much performance headroom the pre-trained representations support. The architecture (variant-token extraction + projection) is in place; the data and compute are the remaining gap.

---

## Reproducing and extending this work

The zero-shot evaluation script is at `scripts/zeroshot_patho_eval.py`. It downloads ClinVar and gnomAD data (idempotent — skips if already present), embeds all variants, runs the k-NN, and saves results to `reports/zeroshot_pathogenicity_knn.json`.

```bash
uv run python scripts/zeroshot_patho_eval.py
```

The full pipeline components:

- **Data acquisition:** `mtdna_fm/data/variant_downloader.py` — `download_clinvar_chrm()`, `download_gnomad_chrm()`
- **Parsing:** `mtdna_fm/data/variant_processor.py` — `parse_clinvar_chrm_vcf()`, `parse_gnomad_chrm_vcf()`, `add_benign_proxies()`
- **Embedding:** `mtdna_fm/inference/api.py` — `MtDNAEmbedder.embed_variant(sequence, position)`
- **Evaluation:** `mtdna_fm/evaluation/variant_eval.py` — `compute_metrics(y_true, y_score, positions)`

The fine-tuning adapter architecture (`MtDNAForVariantPathogenicity`, LoRA r=4) is implemented in `mtdna_fm/model/model.py`. The training dataset class (`PathogenicityVariantDataset`) is in `mtdna_fm/data/variant_dataset.py`. Once a labeled training split exists, fine-tuning requires no changes to those components.
<!-- published: https://rokpayprsizors.wordpress.com/2026/06/04/i-built-a-pathogenicity-predictor-i-dont-have-the-data-to-evaluate-it/ -->