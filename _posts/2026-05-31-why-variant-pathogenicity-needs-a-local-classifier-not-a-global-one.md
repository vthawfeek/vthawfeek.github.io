---
title: "Why Variant Pathogenicity Needs a Local Classifier, Not a Global One"
date: 2026-05-31
tags: [mtDNA, foundation model, bioinformatics, machine learning, BERT]
layout: post
---

There is a design decision in variant pathogenicity prediction that sounds small but turns out to matter a lot: where in the model do you read the prediction from?

The obvious choice is the CLS token. It aggregates the whole sequence window, and it is what haplogroup classification uses. But pathogenicity is not a global property. A single nucleotide substitution at position 3243 that converts an adenine to guanine causes MELAS syndrome, one of the most severe mitochondrial diseases. Nothing about the rest of the genome tells you that. The pathogenicity lives at that position, in that tRNA gene, where the mutation disrupts the TψC loop of mt-tRNALeu. Reading it from a global aggregate loses exactly the signal you need.

This is part of a four-week open-source project to build the first dedicated foundation model for mitochondrial DNA. mtDNA mutations drive over 350 inherited diseases, including MELAS, Leigh syndrome, and Leber hereditary optic neuropathy. No sequence model designed specifically for the circular mitochondrial genome exists.

## What was built

Day 17 adds the second fine-tuning task: binary variant pathogenicity prediction. The architecture change is in `MtDNAForVariantPathogenicity`, which takes a 512-token window centered on the variant, runs it through the encoder, and then extracts the hidden state at the token covering the mutation site:

```python
variant_hidden = hidden[
    torch.arange(hidden.size(0), device=hidden.device),
    idx,  # the token index covering the variant position
    :
]  # shape: (batch, hidden_size)
logit = self.classifier(self.dropout(variant_hidden))  # (batch, 1)
```

Compare this to haplogroup classification, which mean-pools CLS tokens across 63 overlapping windows before classifying. The pooling is correct for haplogroup because the signal is distributed: H vs L3 is the cumulative pattern of hundreds of variant sites. Pathogenicity is the opposite case.

## Training data construction

The training split uses ClinVar pathogenic mtDNA variants as the positive class (roughly 2,000 variants after filtering for single nucleotide substitutions) and gnomAD common variants with allele frequency above 0.01 as the negative class (roughly 5,000 variants). The class ratio is 1:2.5, which is why the BCE loss uses `pos_weight=2.5`:

```python
loss = F.binary_cross_entropy_with_logits(
    logits,
    labels.float(),
    pos_weight=self.pos_weight,  # 2.5
)
```

This is not just a reweighting trick. For a screening task where the downstream consequence of missing a real pathogenic variant is higher than triggering a false positive review, the loss function should reflect that asymmetry. `pos_weight=2.5` roughly doubles the gradient contribution of each pathogenic example relative to a benign one.

The variants are stratified by type at split time: missense, tRNA, rRNA, D-loop. The four classes have different base rates in ClinVar and different expected model performance. Missense variants in protein-coding genes are the largest category; tRNA variants cause some of the most severe diseases but are a smaller set. Keeping the type distribution consistent between train and test prevents type-specific overfitting from inflating the aggregate AUROC.

## LoRA configuration and why it differs from haplogroup

Haplogroup fine-tuning uses LoRA r=8. Pathogenicity uses r=4. The reason is dataset size. Haplogroup fine-tuning has 34,974 human sequences. Pathogenicity has 7,000 variant windows. With r=4, the LoRA adapter has 4 times fewer free parameters than r=8. Reducing capacity is the right move when the training set is small.

Weight decay is also heavier: 0.1 for pathogenicity vs 0.01 for haplogroup. This is standard practice for small-dataset fine-tuning. The L2 penalty on the LoRA matrices keeps them from growing large enough to memorise training variants.

Expected AUROC: above 0.85. The pre-trained encoder has seen all ~117,500 vertebrate mtDNA genomes during Phase 1. Positions that are pathogenic in humans are often conserved across species precisely because they are functionally important. The cross-species corpus teaches the model which positions tolerate change and which do not, before any disease labels appear.

![Shannon entropy per genomic position, showing the D-loop (positions 576-16024) is 7x more variable than protein-coding regions. Conserved positions in coding genes align with known pathogenic variant hotspots.](http://rokpayprsizors.files.wordpress.com/2026/05/positional_entropy-4.png)

The entropy figure illustrates the signal the pre-trained model has already learned: conserved positions (low entropy) are where pathogenic variants cluster. The fine-tuning head learns to convert that pre-trained structural knowledge into explicit pathogenicity probabilities.

## Synthetic fallback for tests

`PathogenicityVariantDataset` generates 64 random synthetic variants when the parquet file is missing:

```python
if not path.exists():
    rng = np.random.default_rng(42)
    ref = "".join(rng.choice(list("ACGT"), size=16569))
    rows = [{"sequence": ..., "position": pos, "label": i % 2, ...} for i in range(64)]
    df = pd.DataFrame(rows)
```

This keeps the test suite runnable in CI without real variant data. The synthetic path exercises the same tokenisation code as real data: it tokenises a full 16,569-bp sequence, extracts a 512-token centered window, and computes the variant token index. What it does not test is whether the labels make biological sense, but that is not what unit tests are for.

The 10 new tests cover: forward shapes, sigmoid probs in [0,1], BCE loss is a scalar and not NaN, no labels means no loss, gradients reach the classifier weights, `pos_weight` is a registered buffer at the right value, out-of-bounds `variant_token_idx` is clamped without raising, freeze/unfreeze encoder, loss decreases over 5 gradient steps, LoRA wraps the model correctly.

Total test count: 274 (was 264 after Day 16).

## Key takeaways

- For pathogenicity prediction, the variant-position token hidden state is more informative than CLS because pathogenicity is a local property: the effect of a substitution on a specific codon or tRNA stem, not on the genome as a whole.
- `pos_weight=2.5` in `BCEWithLogitsLoss` compensates for a 1:2.5 ClinVar/gnomAD class imbalance and encodes the domain asymmetry that missing a real pathogenic variant is more costly than a false alarm.
- LoRA rank selection should scale with dataset size: r=4 for 7k variant windows vs r=8 for 35k full genomes. Smaller rank reduces adapter capacity to match available training signal.
- Cross-species pre-training implicitly encodes functional constraint: positions conserved across ~117k vertebrate genomes are exactly where pathogenic human variants cluster. The fine-tuning head converts that pre-learned signal into probabilities.

[GitHub repo: https://github.com/vthawfeek/mtdna-foundation-model]