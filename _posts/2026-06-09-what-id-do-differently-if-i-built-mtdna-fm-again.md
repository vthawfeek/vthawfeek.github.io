---
title: "What I'd Do Differently If I Built mtDNA-FM Again"
date: 2026-06-15
tags: [mtDNA, foundation model, bioinformatics]
layout: post
---

I'm at the end of a 25-day sprint building a foundation model for mitochondrial DNA on a laptop CPU. The model is trained, the weights are on HuggingFace, the demo is live. And there are six decisions I'd reverse if I started this tomorrow.

This isn't a clean retrospective on a finished, polished project. It's what I'd actually change before starting the next iteration.

---

## The setup

mtDNA-FM is a 6-layer BERT encoder, 256 hidden dimensions, approximately 6M parameters. Pre-trained on 152,484 genomes (117k cross-species vertebrates plus 35k human sequences from HmtDB). Vocabulary of 4,102 tokens: 4,096 overlapping 6-mers plus 6 special tokens. Novel features: circular positional encoding for the genome's topology, and a heteroplasmy projection channel for continuous variant frequency data.

Zero-shot k-NN haplogroup accuracy: ~50% with no fine-tuning labels, against a 3.85% random baseline. Fine-tuned classifier after 2 CPU epochs: 1.83%, below random.

That gap between 50% and 1.83% is the organizing frame for everything below.

---

## Learning 1: Evaluation first

I built the pathogenicity prediction architecture (MtDNAForVariantPathogenicity, LoRA r=4, pos_weight=2.5) before I had a labeled evaluation dataset.

This is backwards. The right order is: define what evaluation success looks like, assemble the labeled dataset, then build toward it. I had the ClinVar download code, the gnomAD variant files, the model architecture. What I didn't have was a labeled dataset that mapped ClinVar variant calls to specific genomic windows in a format the model could evaluate against. By the time I got to evaluation on Day 19, the honest result was: architecture built, no evaluation prepared.

The lesson is not "do more work." It is "work in a different order." If I started tomorrow, I'd spend Day 1 building the pathogenicity evaluation file, running a random baseline, and understanding what an informative confusion matrix would look like. Then I'd build the model to beat that baseline. Scope everything else around the one thing that can be evaluated end-to-end.

*Update: a zero-shot k-NN evaluation was subsequently run using real ClinVar + gnomAD data — AUROC=0.777 (95% CI 0.731–0.821). The lesson still stands: evaluation-first discipline would have caught the data pipeline bugs (ClinVar chromosome naming, gnomAD AF field name differences) earlier and shaped architecture choices differently.*

---

## Learning 2: The $20 compute question

Fine-tuning convergence failed because of CPU compute. Not model architecture. Not data quality. Not hyperparameters. Pure arithmetic.

Each forward+backward pass through the 6-layer, 256-dim transformer takes approximately 30 seconds on CPU. With 644 batches per epoch, one epoch = 6.5 hours. LoRA fine-tuning typically needs 10-50 epochs. The 50-epoch run that would likely converge: roughly 325 hours on the same laptop.

On an A100, the same run takes about 50 minutes total. On a V100, about 2.5 hours.

A Colab Pro subscription costs $10/month. A single A100 GPU session that covers the full fine-tuning run costs approximately $20. This is not a high barrier. It is a decision that I didn't make at the right point in the project, and the fine-tuning story would have been entirely different if I had.

The practical lesson: identify the wall-clock math for any training run before starting it. If CPU time-to-convergence exceeds what you're willing to wait, either get compute or change scope. Don't discover this at epoch 2.

---

## Learning 3: One task, fully evaluated

I built three downstream task heads: haplogroup classification (26-way), pathogenic variant prediction (binary), and heteroplasmy regression. Each has its own LoRA adapter, its own dataset class, its own evaluation framework.

Evaluated end-to-end: haplogroup classification only, and only in the zero-shot regime.

The right trade-off was to pick one task and run it to completion, including labeled fine-tuning evaluation with proper train/val/test splits, a calibrated model, and an honest comparison to a relevant baseline. One task fully evaluated is a clearer signal than three tasks with partial results.

This doesn't mean the other architectures were wasted work. The heteroplasmy regression head, in particular, is the right design for the clinical use case. But if the portfolio question is "did this model work?", the answer is muddier when you have three partial answers instead of one complete one.

---

## Learning 4: The 60-70% European problem

HmtDB is the primary human mtDNA database used for Phase 2 pre-training and haplogroup fine-tuning. An estimated 60-70% of sequences in HmtDB are from European haplogroups (H, HV, U, J, T, and subclades).

This matters because haplogroup representation in the training set directly affects what the model learns. Rare African haplogroups like L0, L1, and L5 have fewer training examples. The embedding space will cluster more tightly around common European haplogroups and more loosely around rare African ones.

The fix is stratified sampling: rather than taking all 35k HmtDB sequences at face value, sample a roughly equal number of sequences per high-level haplogroup. This would reduce the total training set size but produce more balanced representations across the 26 haplogroup classes. At the fine-tuning stage, it also directly reduces the class imbalance problem that drove partial class collapse.

I knew this bias existed from Day 4 of the EDA. I didn't act on it because the downstream effects weren't yet visible. They became visible at Day 19 evaluation. Stratified sampling is a one-day fix that should have been applied during data preprocessing.

---

## Learning 5: The ablations I didn't run

The circular positional encoding is mathematically correct. It produces position representations where pos 0 and pos 16,568 are adjacent, as they physically are in the molecule. The heteroplasmy projection channel is theoretically motivated: clinical thresholds for conditions like MELAS depend on a continuous float that can't be recovered from the sequence alone.

What I didn't do: run controlled experiments to show that either of these actually improves the representations.

Ablations I'd run if I started over:

1. Circular PE vs standard sinusoidal PE: pre-train both configurations on the same data, compare zero-shot k-NN accuracy and the D-loop attention patterns specifically.

2. Het channel vs no het channel: pre-train with and without the het projection, compare Phase 2 pre-training loss and any downstream heteroplasmy regression performance.

3. Masking rate: I used 15% (standard BERT). mtDNA with its 4,096-token vocabulary may have an easier reconstruction problem than full-genome models, which suggests a higher masking rate (25-30%) could force better representations. This is an 8-line change and would take less than a day to run.

4. Tokenization stride: adjacent 6-mers share 5 positions out of 6 (stride=1 overlapping tokenization). A stride-3 or stride-6 tokenization would produce less redundant tokens. Whether the overlap actually helps or hurts is testable with a pre-training loss comparison.

None of these ablations are expensive. Each is a configuration change and a new training run. The cost of not running them is that the specific claims about what circular PE and the het channel contribute are not supported by ablation evidence. The designs are still the right ones. I just can't prove it quantitatively yet.

---

## Learning 6: Get a baseline before week 2

I built the entire pre-training pipeline before asking: what would DNABERT2 score on this same zero-shot k-NN task if you just ran it on mtDNA sequences?

This matters because without an external comparison point, the 50% zero-shot k-NN result is ambiguous. It's clearly better than 3.85% random. But is it better than a general DNA model that happens to have been pre-trained on vertebrate sequences including mtDNA? Is the circular PE doing something above what a standard architecture would get? I don't know.

The right approach: in Week 1, before writing any model code, run DNABERT2 and one other baseline (nucleotide transformer, or even a simple k-mer frequency cosine similarity classifier) on the haplogroup task. Document those numbers. Then every subsequent result has a reference point.

Setting up a DNABERT2 inference run takes maybe half a day. The comparison would have made the entire evaluation section of this project more rigorous. I'd make it the first thing I did after finishing the tokenizer.

---

## What I'd keep

The circular positional encoding is the right design. The D-loop spans the origin-of-replication junction, and linearisation is a genuine problem for any model trying to represent that region. The fix is correct even without ablation confirmation.

The heteroplasmy projection channel is the right design. Encoding a continuous float as a separate channel, projected into the embedding space with a single nn.Linear layer, is how single-cell foundation models handle expression values. It avoids arbitrary discretization and preserves resolution near clinical thresholds.

The two-phase pre-training strategy (Phase 1 on 117k cross-species genomes, Phase 2 on 35k human-specific sequences) is the right structure. Cross-species pre-training builds broad structural representations of what mitochondrial DNA looks like. Human-specific fine-tuning specializes those representations toward the population-level variation that matters for clinical use.

The execution gaps are all fixable: GPU compute for fine-tuning, stratified sampling for HmtDB, ablations for the novel components, an earlier baseline comparison. None of them require changing the architecture. They require changing the order of work and spending $20 at the right moment.

---

*Code and pre-trained weights: [github.com/vthawfeek/mtdna-foundation-model](https://github.com/vthawfeek/mtdna-foundation-model)*
*Weights on HuggingFace: [vthawfeek/mtdna-foundation-model](https://huggingface.co/vthawfeek/mtdna-foundation-model)*
<!-- published: https://rokpayprsizors.wordpress.com/2026/06/04/what-id-do-differently-if-i-built-mtdna-fm-again/ -->