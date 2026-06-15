---
title: "I Built a Pathogenicity Predictor. I Don't Have the Data to Evaluate It."
date: 2026-06-15
tags: [mtDNA, foundation model, bioinformatics, variant pathogenicity, machine learning]
layout: post
---

The model is done. The training code runs. The architecture is implemented and the adapter weights are pushed to HuggingFace.

The evaluation dataset doesn't exist.

This post is about that gap, the architectural decision that makes this approach worth validating, and what it would take for someone to close it.

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

LoRA r=4 fine-tuning on top of the pre-trained encoder. Binary cross-entropy with `pos_weight=2.5` to handle the expected imbalance between pathogenic variants (rare, well-curated) and benign variants (common, numerous). The training loop is complete and runs without errors.

---

## Why the evaluation dataset doesn't exist

A credible evaluation for mtDNA pathogenicity prediction requires two sources.

**Pathogenic variants:** ClinVar maintains a curated set of human mtDNA variants with clinical significance annotations: pathogenic, likely pathogenic, variant of uncertain significance, benign. Filtering to pathogenic and likely pathogenic categories with at least two stars of review status gives the positive examples.

**Benign controls:** gnomAD's mitochondrial variant dataset contains allele frequencies across 56,434 individuals from diverse populations. Variants with population allele frequency above 0.01 (1%) are strong evidence against severe pathogenicity — a variant causing MELAS or Leber hereditary optic neuropathy would face strong purifying selection and not reach 1% population frequency.

The download infrastructure for both sources exists in this project:

```python
from mtdna_fm.data.variant_downloader import download_clinvar_chrm, download_gnomad_chrm

clinvar_vcf = download_clinvar_chrm("data/raw/clinvar")    # filters to chrM automatically
gnomad_vcf  = download_gnomad_chrm("data/raw/gnomad")      # chrM .bgz + .tbi
```

Building the evaluation dataset from these sources requires:

1. Download ClinVar and gnomAD VCFs (client functions exist; a few hours of download time)
2. Map variant positions to the rCRS reference coordinate system and validate consistency between the two databases
3. Reconstruct the full-length 16,569 bp sequence for each variant: reference + alt allele
4. Deduplicate and balance positive and negative sets to avoid class imbalance distorting AUROC estimates
5. Create stratified train/test splits by variant type: missense, tRNA, rRNA, control region
6. Verify no test variants appear in ClinVar's training annotations (data leakage check)

Steps 1–2 take a few hours. Steps 3–6 take a full day of careful data engineering: understanding the rCRS coordinate conventions in both databases, writing the merging and balancing logic, validating that the splits are clean.

The data never got downloaded.

---

## What "no evaluation" means in practice

The only honest statement about this model is: it runs. The architecture makes sense. The training objective is correct. Whether it learned to distinguish pathogenic from benign mtDNA variants is unknown.

Without the evaluation dataset, there's no way to know whether the output logits are meaningful. This is different from the haplogroup case, where the failure mode is visible (1.83% accuracy, class collapse, clear compute diagnosis). For pathogenicity, there's no number at all. The model exists in a state of undefined performance.

Publishing adapter weights without a performance number is misleading in a specific way: the weights being there implies the model is validated. It isn't. Someone could download these weights, run predictions on a clinical variant set, and have no calibration for whether those predictions are better than random.

---

## What a proper evaluation would look like

**AUROC**: area under the ROC curve. Random classifier = 0.5. Established pathogenicity predictors (CADD, SIFT, PolyPhen-2) achieve 0.7–0.9 on held-out benchmarks.

**Precision-recall**: more informative than ROC when positive examples are rare. ClinVar pathogenic mtDNA variants number in the hundreds to low thousands; gnomAD can supply many more benign controls. The precision-recall curve shows whether the model maintains precision at high recall, which matters clinically.

**Calibration**: does a predicted probability of 0.8 actually correspond to 80% of variants being pathogenic? Poor calibration means scores need threshold tuning before clinical use.

**Stratified performance**: AUROC broken down by variant type. A model that works for missense variants in protein-coding genes might fail entirely on tRNA mutations, which have a different molecular mechanism — anticodon stem disruption vs. catalytic site disruption. Stratification reveals which variant classes the model actually learned.

The evaluation code exists in `mtdna_fm/evaluation/variant_eval.py` and computes all four of these analyses. The missing piece is the input data.

---

## Reproducing and extending this work

The full pipeline is public and documented. If you work on mtDNA variant effect prediction and want to validate this architecture, the starting point is there.

**Data acquisition** — both download clients are idempotent and handle decompression:
- `mtdna_fm/data/variant_downloader.py`: `download_clinvar_chrm()`, `download_gnomad_chrm()`

**Training** — `PathogenicityVariantDataset` expects a DataFrame with `pos`, `ref`, `alt`, `label` columns. Once that DataFrame exists, the training pipeline requires no modification:
- `mtdna_fm/data/variant_dataset.py`: `PathogenicityVariantDataset`

**Evaluation** — computes AUROC, AUPRC, calibration, and per-variant-type breakdown:
- `mtdna_fm/evaluation/variant_eval.py`: `compute_metrics()`

**Adapter weights** — the LoRA r=4 adapter trained on the pre-trained encoder (without real ClinVar/gnomAD data) is at [vthawfeek/mtdna-foundation-model](https://huggingface.co/vthawfeek/mtdna-foundation-model) on HuggingFace.

A one-weekend effort with GPU access could download the data, fine-tune on real ClinVar/gnomAD variants, and produce a complete evaluation. The variant-token architecture design is sound. The gap is a data preparation script that was never written and a GPU that wasn't available.
<!-- published: https://rokpayprsizors.wordpress.com/2026/06/04/i-built-a-pathogenicity-predictor-i-dont-have-the-data-to-evaluate-it/ -->