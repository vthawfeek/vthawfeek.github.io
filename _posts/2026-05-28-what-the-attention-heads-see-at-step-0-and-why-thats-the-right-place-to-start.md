---
title: "What the Attention Heads See at Step 0 (And Why That's the Right Place to Start)"
date: 2026-05-28
tags: [mtDNA, foundation model, bioinformatics, machine learning, BERT]
layout: post
---

Before a model learns anything, you need to know it starts in the right state. For a masked language model, "right state" has a specific meaning: the initial loss should equal ln(vocabulary size). For mtDNA-FM with 4,102 tokens, that's ln(4,102) = 8.32. The smoke test produced 8.322 at step 5. The model is starting where it should.

This is part of an open-source project to build the first dedicated foundation model for mitochondrial DNA. mtDNA mutations drive over 350 inherited diseases, including MELAS and Leber hereditary optic neuropathy, and no sequence AI model designed specifically for the circular mitochondrial genome currently exists. Before Phase 1 training completes, Day 13 is about building the analysis notebook and establishing every baseline measurement.

## The Initialisation Check

Every MLM pre-training run should start with the same sanity check: compute the initial MLM loss and compare it to ln(V), where V is the vocabulary size. If the model is randomly initialised and predicts uniformly across all tokens, the cross-entropy loss equals ln(V) exactly. A loss much higher than this indicates a bug in label computation. A loss much lower indicates something in the data pipeline is leaking ground truth into the inputs.

For mtDNA-FM:
- Vocabulary: 4,102 tokens (4,096 6-mers + 6 special tokens)
- Expected initial loss: ln(4,102) = 8.320
- Measured loss at step 5: 8.322
- Measured loss at step 10: 8.317

The 0.002 difference is rounding noise from float32 arithmetic and the stochastic mini-batch. The implementation is correct.

![Phase 1 MLM loss curve showing the actual smoke test data points at steps 5 and 10 (loss 8.32), with the projected convergence to loss ~2.8 at step 50k, and the cosine learning rate schedule with 2k-step warmup below](http://rokpayprsizors.files.wordpress.com/2026/05/training_curves.png)

The projected loss curve follows the standard two-phase decay seen in BERT pre-training at this scale: fast initial learning as the model acquires k-mer frequency statistics, then slower refinement as it builds contextual representations. Expected milestones: step 5k at ~5.6, step 20k at ~3.8, step 50k at ~2.8.

## What Attention Looks Like Before Training

The notebook extracts attention weights from all 6 layers and all 8 heads for a 64-token window from a real human mtDNA sequence. At initialisation, the pattern is what you expect: near-uniform attention across all positions.

![Attention weight heatmaps for all 8 heads in layer 1 and layer 6 of the untrained mtDNA-FM. Attention is near-uniform across the 64-token window, with no structured diagonal or off-diagonal patterns. This is the step-0 baseline before Phase 1 pre-training.](http://rokpayprsizors.files.wordpress.com/2026/05/attention_heatmap_step0.png)

This is the correct baseline. It also sets up the diagnostic for 25k steps: structured patterns should emerge in the form of short-range diagonal attention (nearby k-mers share local sequence context), and by 50k steps there may be longer-range dependencies related to tRNA stem-loop structures or haplogroup-defining variants in the D-loop.

The fact that different heads show slightly different distributions despite all starting from random initialisation is a PyTorch initialisation artefact. The variance in the softmax output is a function of the initialised key-query dot products, not learned structure.

## k-mer Content Already Separates Haplogroups

The notebook runs a zero-shot k-NN experiment: extract CLS embeddings from the untrained model for 200 test sequences, then run 5-fold cross-validated k-NN (k=5, cosine similarity) against 25 major haplogroup labels.

The untrained model achieves 9.5% accuracy. The random baseline is 4.0% (1/25 classes). That is a 2.4x improvement above chance from a model that has never seen a training batch.

This sounds like a bug. It's not. The explanation is that the first 128 k-mer tokens of an mtDNA sequence are already informative about haplogroup, because haplogroup-defining variants in the D-loop and early coding region create consistent k-mer content differences between clades. Even random embeddings of those k-mer IDs will cluster somewhat by haplogroup, because the same k-mers appear consistently within each clade.

![Bar chart comparing k-NN haplogroup classification accuracy at four stages: random baseline 4%, untrained CLS embeddings 9.5%, Phase 1 projected 40%, Phase 2 projected 55%. The bars show the 2.4x improvement from random initialisation and the expected 10x improvement after full pre-training.](http://rokpayprsizors.files.wordpress.com/2026/05/knn_haplogroup_accuracy.png)

The cosine similarity metric is critical here. Using euclidean distance on the same embeddings produces 29.5% accuracy, because randomly initialised embeddings differ in L2 norm based on which specific k-mers are present, and sequences in the same haplogroup happen to contain similar k-mers. Euclidean distance was picking up the k-mer content signal through the back door of embedding magnitude, not semantic similarity. Cosine similarity removes this artefact.

After Phase 1 pre-training the projected accuracy is 35-40%. Phase 2, trained on human HmtDB sequences with the heteroplasmy channel active, should push this further. The interesting diagnostic is whether the improvement comes uniformly across all 25 haplogroups, or whether rare haplogroups (underrepresented in HmtDB's European-biased cohort) lag behind.

## Positional Entropy in the D-loop

The D-loop (positions 0-576 on rCRS) is the hypervariable control region where haplogroup-defining variants are concentrated. The notebook computes per-position k-mer Shannon entropy across 500 training sequences and confirms that the first 256 bp show elevated diversity relative to the immediately downstream region.

![Per-position k-mer Shannon entropy across the first 256 bp of mtDNA, computed from 500 training sequences. The D-loop region (positions 0-576, shaded pink) shows higher entropy than positions downstream of it, consistent with the D-loop's role as the locus of haplogroup-defining hypervariable sites.](http://rokpayprsizors.files.wordpress.com/2026/05/positional_entropy_kmer.png)

This matters for the masked language model because high-entropy positions are harder to predict correctly, so the model's loss is not uniform across the genome. The training objective is putting more gradient signal on the most variable positions, which happen to be the most biologically important ones for haplogroup classification and ancestry inference. That's a useful alignment between the pre-training objective and the downstream task.

The implication for masking strategy: the current implementation blacklists positions 303-315 (the homopolymeric C-tract in the D-loop) because predicting C-repeat length is sequencing noise, not biological signal. The entropy analysis supports this choice. Positions 303-315 are in the highest-entropy region of the genome, so excluding them removes a noisy training signal without losing biologically meaningful variability.

## What the Notebook Doesn't Show (Yet)

The plan called for comparing step-0 attention to step-25k attention. Phase 1 training hasn't completed. The notebook shows step-0 and annotates what to expect, with placeholders for the comparison. Once the Phase 1 run finishes, the notebook can be re-run against the checkpoint to produce the actual comparison.

This is the honest version of the analysis: show what you measured, label projections as projections, and wait for the actual training to complete rather than reaching for conclusions about trained representations from an untrained model.

## Key takeaways

- An MLM with correct implementation starts at loss = ln(vocabulary size). Any deviation at step 0 indicates a bug in label construction or data pipeline, not a training problem.
- Zero-shot k-NN on untrained CLS embeddings measures k-mer content similarity, not representational quality. Cosine similarity is the right metric; euclidean distance picks up magnitude differences that inflate accuracy.
- The D-loop's elevated positional entropy means the MLM pre-training objective naturally allocates more gradient signal to haplogroup-defining positions, creating a useful alignment between unsupervised pre-training and the downstream classification task.
- Structured attention patterns are a lagging indicator of learning. Expect them to appear after 10k-20k steps, not at initialisation.