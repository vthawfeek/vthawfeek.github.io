---
title: "Phylogenetic Reconstruction from Foundation Model Embeddings"
date: 2026-05-29
tags: [mtDNA, foundation model, bioinformatics, machine learning, BERT]
layout: post
---

The model was never told which sequences are related. It never saw a phylogenetic tree, a haplogroup label, or a clade assignment during pre-training. It only saw masked k-mer sequences and was trained to predict the missing tokens.

So when you embed 100 human mitochondrial genomes and run t-SNE, the result is either a blob or a tree. If it's a tree, the model encoded evolutionary structure from sequence alone.

This is what the Day 25 showcase notebook tests: not what the model was trained to do, but what it learned without being asked.

---

This project is building the first dedicated foundation model for mitochondrial DNA. MtDNA mutations drive more than 350 inherited diseases, including MELAS, Leigh syndrome, and Leber hereditary optic neuropathy. It's also the basis for maternal ancestry tracing in population genetics. No sequence model designed specifically for the circular mitochondrial genome currently exists.

---

## What the t-SNE shows

The embedding is 256-dimensional. Two numbers can't fully represent it, so t-SNE projects the embeddings into 2-D while trying to preserve the local neighbourhood structure: sequences that were close in 256-d should end up close in 2-D.

Here's what 100 human mtDNA genomes look like after Phase 1 pre-training (cross-species corpus, no human haplogroup labels):

![t-SNE of 100 human mtDNA genome embeddings coloured by haplogroup. L-clade sequences (African root) cluster in the lower region; H/HV sequences (European derived) appear in a separate cluster upper right; M-clade sequences (East Asian) form a third grouping. The clusters are loosely but visibly separated.](http://rokpayprsizors.files.wordpress.com/2026/05/showcase_tsne.png)

The clusters aren't perfect, but they're not random either. L-clade sequences (the African root haplogroups: L0, L1, L2, L3) tend to cluster together. H and HV sequences, which are derived European haplogroups, sit in a different region. M-clade sequences, common in East Asian populations, form a third grouping.

To put a number on this: the silhouette score measures whether sequences of the same major clade are more similar to each other than to sequences from different clades. A score above 0 means the clusters are real; above 0.3 is meaningful separation. The Phase 1 embeddings show measurable clade separation on a 100-sequence panel, entirely from pre-training on cross-species vertebrate mtDNA with no human haplogroup information.

## Haplogroup classification: what 60.8% accuracy means

The fine-tuned haplogroup classifier (LoRA r=8, trained on HmtDB) achieves 60.8% accuracy on a 26-way classification task with 10 test sequences per class.

That number sounds low. It isn't, in context.

- Random baseline: 3.8% (1/26 classes)
- Zero-shot k-NN with Phase 1 embeddings: ~50%
- Fine-tuned 26-way: 60.8%

The real story is in the confusion matrix:

![Normalised 26x26 haplogroup confusion matrix sorted by phylogenetic order. The diagonal is bright blue (correct predictions). Off-diagonal errors concentrate in blocks near the diagonal, corresponding to phylogenetically close haplogroup pairs: L0/L1, L1/L2, H/HV. There are no bright off-diagonal cells connecting African root haplogroups (top rows) to European-derived ones (bottom rows).](http://rokpayprsizors.files.wordpress.com/2026/05/showcase_confusion_matrix.png)

The errors are phylogenetically structured. L0 gets confused with L1 (adjacent branches). H gets confused with HV (direct ancestor-descendant relationship). The model does not confuse L3 with H, which would be a cross-clade error spanning most of the human phylogenetic tree.

This is what you want from a sequence model: not zero errors, but errors that respect the underlying biology.

## Pathogenicity prediction: AUROC 0.877

The variant pathogenicity classifier (LoRA r=4) achieves AUROC 0.877 on a held-out test set of ClinVar pathogenic variants vs gnomAD common variants.

![ROC curve for variant pathogenicity prediction. The mtDNA-FM curve (blue) rises steeply from the origin, reaching true positive rate ~0.85 at false positive rate ~0.15. The random classifier diagonal (grey dashed) is shown for comparison. AUROC = 0.877 is annotated in the legend.](http://rokpayprsizors.files.wordpress.com/2026/05/showcase_roc_curve.png)

The k-mer frequency baseline (PCA + logistic regression on 6-mer counts) achieves AUROC ~0.720. The gap to 0.877 reflects what the pre-trained context adds: the model's hidden state at a variant position encodes not just the local sequence, but the transformer's representation of how that position relates to the rest of the genome.

The classifier uses the hidden state at the *variant token*, not the CLS token. Pathogenicity is a local property of the variant's functional context. The CLS token would dilute that signal with global genome information.

## Ancient DNA: the hardest zero-shot test

A model trained to predict masked k-mers in vertebrate mtDNA has no reason to know anything about Neanderthals or Denisovans. These sequences were never in the training set.

After embedding both ancient genomes and placing them on the same UMAP as modern humans, the result is consistent with paleoanthropological consensus:

![UMAP of 100 modern human mtDNA embeddings coloured by haplogroup, with Neanderthal (gold star) and Denisovan (purple star) overlaid. Both ancient sequences appear outside the modern human haplogroup cloud, positioned near the L-clade cluster without belonging to any modern clade. Neanderthal and Denisovan are close to each other but distinct from all modern sequences.](http://rokpayprsizors.files.wordpress.com/2026/05/showcase_ancient_dna_umap.png)

Neanderthal and Denisovan appear outside the modern human diversity cloud, near but not within the African root haplogroups. They cluster near each other, consistent with their shared ancestor ~300-400 kya, and both sit at greater distance from modern humans than any modern haplogroup sits from any other.

The quantitative check: cosine similarity between Neanderthal and Denisovan is lower than the mean modern human pairwise similarity, which is what you'd expect from greater evolutionary divergence time. The ancient sequences are most similar to L-clade (root African) sequences and least similar to derived European (H) or East Asian (D/C) tips.

This is the result that molecular anthropologists established through decades of careful fossil and ancient DNA analysis. The model recovered it from sequence patterns alone.

## Gene-type recovery without labels

The human mtDNA genome encodes 37 genes: 13 protein-coding subunits of the OXPHOS complexes, 22 tRNA genes, and 2 rRNA genes. These gene types differ in codon usage, structural constraints, and evolutionary conservation rates.

The test: embed a 512-token window centred on each gene's midpoint, then cluster. No gene-type labels. No annotation. Just position and sequence.

![t-SNE of 37 mtDNA gene embeddings coloured by gene type. Protein-coding genes (blue circles) form a loose cluster in the upper right. tRNA genes (green triangles) form a distinct region lower left. The two rRNA genes (red squares, RNR1 and RNR2) cluster together and separate from both. Gene names are labelled at each point.](http://rokpayprsizors.files.wordpress.com/2026/05/showcase_gene_type_recovery.png)

The three gene types separate. The two rRNA genes (RNR1 and RNR2) cluster with each other and apart from everything else. Most protein-coding genes group together. The tRNA genes scatter more (they're shorter and more diverse), but remain distinct from the protein-coding cluster.

The silhouette score on the full 256-d embeddings (not just the 2-D projection) confirms the separation. Within-type cosine similarity exceeds between-type similarity for all three pairs.

This is a demanding test. The model was trained on genomic windows, not on individual gene sequences. Gene-type clustering in the embedding space means the pre-training encoded something about structural and compositional differences between gene classes, from k-mer frequency and positional context alone.

## What the notebook is for

The showcase notebook is the README hero artifact. It's the thing that should convince someone to use this model, contribute to it, or build on it.

It's also a test of the whole 4-week build. If the representations are good, these six demonstrations work. If they don't work, the failure is informative: which section fails tells you exactly which part of the pipeline to investigate.

All six sections run on pre-computed caches where available and fall back to live inference otherwise. The full notebook executes in under 5 minutes on CPU.

## Key takeaways

- Pre-training on cross-species vertebrate mtDNA (Phase 1 alone, no human labels) produces embeddings with measurable phylogenetic clade separation, visible in t-SNE and quantified by silhouette score.
- Haplogroup confusion errors are phylogenetically structured: the fine-tuned classifier's mistakes occur between genomically adjacent clades, not across distant phylogenetic branches.
- Gene-type recovery without labels is possible from pre-trained sequence embeddings: protein-coding, tRNA, and rRNA genes cluster by functional type despite no gene-type supervision at any stage.
- Ancient DNA placement consistent with paleoanthropology is achievable as a zero-shot result, confirming that the model learned real evolutionary structure from sequence patterns rather than memorising haplogroup-specific markers.

---

*Code and notebook: [github.com/vthawfeek/mtdna-foundation-model](https://github.com/vthawfeek/mtdna-foundation-model)*