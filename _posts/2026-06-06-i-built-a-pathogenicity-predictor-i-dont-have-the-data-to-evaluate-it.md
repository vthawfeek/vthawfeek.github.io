---
title: "I Built a Pathogenicity Predictor. I Don't Have the Data to Evaluate It."
date: 2026-06-06
tags: [mtDNA, foundation model, bioinformatics]
layout: post
---

The model is done. The training code runs. The architecture is implemented and the adapter weights are pushed to HuggingFace.

The evaluation dataset doesn't exist.

This post is about that gap, why it's there, and what it would take to close it.

---

## What the model does

`MtDNAForVariantPathogenicity` wraps the pre-trained mtDNA-FM encoder with a binary classification head that predicts whether a point mutation is pathogenic or benign.

The key architectural decision is which hidden state to use for the prediction. The obvious choice is the `[CLS]` token, which accumulates global sequence context across the full 16,569 bp genome. That's how most sequence classifiers work. It's the wrong choice here.

Pathogenicity is a local property. The m.3243A>G mutation in the MT-TL1 tRNA gene causes MELAS encephalomyopathy because of what happens at that specific locus, not because of some property of the full mitochondrial genome. A model using `[CLS]` has to learn to extract local variant effects from a global summary vector. That's asking it to do extra work that the architecture doesn't support well.

The variant-token approach: instead of `[CLS]`, extract the hidden state at the token position that covers the variant. For a mutation at position 3,243, identify which k-mer token spans that position, and pull its 256-dimensional hidden state from the encoder output. Project that through a linear classifier.

```python
# Extract hidden state at variant position
hidden = encoder(input_ids, position_ids, attention_mask).last_hidden_state  # (B, L, D)
variant_hidden = hidden[torch.arange(B), variant_positions]                   # (B, D)
logits = self.classifier(variant_hidden)                                       # (B, 1)
```

The inductive bias here is that the encoder's attention mechanism, operating over the full genome with circular positional encoding, has already assembled a contextualised representation of the k-mer at the mutation site. The classifier just needs to learn to read that representation.

LoRA r=4 fine-tuning on top of the pre-trained encoder. Binary cross-entropy with `pos_weight=2.5` to handle the expected imbalance between pathogenic variants (rare, well-curated) and benign variants (common, numerous). The training loop is complete and runs without errors.

---

## Why the evaluation dataset doesn't exist

A credible evaluation for mtDNA pathogenicity prediction requires two things.

First, pathogenic variants. ClinVar maintains a curated set of human mtDNA variants with clinical significance annotations: pathogenic, likely pathogenic, variant of uncertain significance, benign. The pathogenic and likely pathogenic categories, filtered to variants with at least two stars of review status, are the positive examples.

Second, benign controls. gnomAD's mitochondrial variant dataset contains allele frequencies across 56,434 individuals from diverse populations. Variants with population allele frequency above 0.01 (1%) are strong evidence against severe pathogenicity, since a variant causing MELAS or Leber hereditary optic neuropathy would be subject to strong purifying selection.

Building the evaluation dataset from these two sources requires:

1. Download ClinVar mtDNA variants filtered to pathogenic/likely pathogenic (the download client exists in this project at `mtdna_fm/data/variant_downloader.py`)
2. Download gnomAD common mtDNA variants (AF > 0.01, the gnomAD client also exists)
3. Map variant positions to the rCRS reference coordinate system and validate consistency between the two databases
4. Reconstruct the full-length 16,569 bp sequence for each variant using reference + alt allele
5. Deduplicate and balance the positive and negative sets to avoid class imbalance distorting AUROC estimates
6. Create stratified train/test splits by variant type: missense, nonsense, tRNA, rRNA
7. Verify that no test variants appear in ClinVar's own training annotations (data leakage check)

Steps 1 and 2 take a few hours of download and parsing time. Steps 3 through 7 take a full day of careful data engineering: understanding the rCRS coordinate conventions in both databases, writing the merging and balancing logic, validating that the splits are clean.

---

## What "no evaluation" means in practice

The model architecture is sound. The training code runs. There are no bugs in the forward pass or the loss computation. The LoRA adapter produces output logits that could be probabilities of pathogenicity.

But there's no way to know whether those logits are meaningful.

Without the evaluation dataset, the only honest statement about this model is: it runs. The architecture makes sense. The training objective is correct. Whether it learned to distinguish pathogenic from benign mtDNA variants is unknown.

This is different from the haplogroup case, where the failure mode is visible (1.83% accuracy, class collapse, clear compute diagnosis). For pathogenicity, there's no number at all. The model exists in a state of undefined performance.

Publishing adapter weights on HuggingFace without a performance number is a specific kind of misleading. Someone could download these weights, run predictions on their clinical variant set, and have no calibration for whether those predictions are better than random. The architecture writeup is honest about this limitation, but the weights being there makes the model look more finished than it is.

---

## What a proper evaluation would look like

The evaluation metrics for a binary pathogenicity classifier:

**AUROC**: area under the ROC curve. For a random classifier, AUROC = 0.5. Clinically useful pathogenicity predictors (CADD, SIFT, PolyPhen-2) achieve AUROC in the range 0.7 to 0.9 on held-out benchmarks. This is the primary metric.

**Precision-recall curve**: more informative than ROC when positive examples are rare. ClinVar pathogenic mtDNA variants number in the hundreds to low thousands; gnomAD can supply many more benign controls. The precision-recall curve shows whether the model maintains precision at high recall, which matters clinically.

**Calibration**: does a predicted probability of 0.8 actually correspond to 80% of variants being pathogenic? Poor calibration means the scores need threshold tuning before clinical use. Plot predicted probability quantiles against empirical frequency.

**Stratified performance**: AUROC broken down by variant type (missense in protein-coding genes vs tRNA mutations vs rRNA mutations vs control region variants). Different variant types have different molecular mechanisms; a model that works for missense variants might completely fail on tRNA mutations.

A one-weekend effort could assemble the evaluation dataset and produce all four of these analyses. The infrastructure exists. The data sources are public. The gap is a data preparation script that was never written.

---

## What the scope decision cost

This project built three downstream fine-tuning tasks in parallel: haplogroup classification, pathogenic variant prediction, and heteroplasmy regression. That spread four weeks of part-time development effort across three separate architectures, three training pipelines, and three evaluation problems.

The result: one task with a visible convergence failure (haplogroup), one task with no evaluation at all (pathogenicity), and one task with a mean absolute error metric but no real-world calibration (heteroplasmy regression).

A different scope decision would have been: one task, built completely. Choose haplogroup classification, since it has the largest labeled dataset (HmtDB has 8,921 properly-labeled sequences), the clearest evaluation metric (26-class accuracy, top-1 and top-3), and the most interpretable failure modes. Spend the time saved from not building pathogenicity and heteroplasmy regression on assembling a proper evaluation benchmark, running full convergence experiments on GPU, and writing up the haplogroup results with statistical rigor.

Three half-finished tasks is not better than one fully finished one.

The variant-token architecture for pathogenicity prediction is still worth publishing as a design note. It generalises beyond mtDNA: any case where you need to predict the effect of a single-position change in a longer sequence context could use this approach. Splice-site mutations, promoter variants, single amino acid substitutions in proteins. The design is sound.

The evaluation just isn't there yet.
<!-- published: https://rokpayprsizors.wordpress.com/2026/06/04/i-built-a-pathogenicity-predictor-i-dont-have-the-data-to-evaluate-it/ -->