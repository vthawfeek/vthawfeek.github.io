---
title: "Ancient DNA and Foundation Models: An Unexpected Application"
date: 2026-05-29
tags: [mtDNA, foundation model, bioinformatics, machine learning, BERT]
layout: post
---

A model trained on modern vertebrate genomes gets fed a sequence from a 50,000-year-old Neanderthal. It has never seen ancient DNA. It has no evolutionary tree, no labels, no dates. The question: does it know this sequence is different from modern humans, and where in sequence space does it place it?

This is part of a project to build the first dedicated foundation model for mitochondrial DNA. mtDNA mutations drive over 350 inherited diseases including MELAS and Leigh syndrome, and haplogroup structure underpins maternal ancestry analysis. No AI model designed specifically for the circular mitochondrial genome existed before this.

## The setup

Day 20 of the build was supposed to be a validation of what the pre-trained encoder learned, using ancient DNA as a probe. The logic: if the model captured real evolutionary structure from sequence alone, it should place Neanderthal (NC_011137.1, Vindija Cave, Croatia) and Denisovan (FR695060.1, Altai Cave, Russia) in a geometrically coherent position relative to modern humans, without any supervised signal about their identity or age.

These sequences were downloaded directly from NCBI and embedded using `MtDNAEmbedder.from_pretrained("models/phase1_v1")`. No fine-tuning. No labels. No special handling beyond the standard windowing and mean-pooling:

```python
from mtdna_fm.data.ancient_dna import prepare_ancient_sequences
from mtdna_fm.inference.api import MtDNAEmbedder

embedder = MtDNAEmbedder.from_pretrained("models/phase1_v1")
ancient_seqs = prepare_ancient_sequences()   # {"Neanderthal": "ATGC...", "Denisovan": "ATGC..."}
ancient_embs = {k: embedder.embed_genome(v) for k, v in ancient_seqs.items()}
```

One small practical issue: Neanderthal mtDNA is 16,565 bp and Denisovan is 16,570 bp, while the human reference (rCRS) is 16,569 bp. The positional encoding buffer has exactly 16,569 entries. Feeding a 16,570 bp sequence caused an IndexError at position 16,569. The fix is four lines in `embed_genome()`:

```python
genome_length = self.model.config.genome_length
if len(sequence) > genome_length:
    sequence = sequence[:genome_length]
elif len(sequence) < genome_length:
    sequence = sequence + "N" * (genome_length - len(sequence))
```

This makes the embedder robust to any mitochondrial genome, not just the rCRS. Ancient sequences differ from modern ones by a handful of base pairs. Truncating or N-padding at the boundary is biologically reasonable.

## What the numbers say

The embedding comparison uses 100 modern human sequences sampled across 50 haplogroups from the HmtDB test set, plus the two ancient sequences. I used L2 distance, not cosine similarity, for reasons explained below.

**Neanderthal:** mean L2 distance to modern humans = 0.1110
**Denisovan:** mean L2 distance to modern humans = 0.1070
**Modern human pairwise:** mean L2 = 0.0749

Ancient sequences are 1.45-1.48 times farther from modern humans than modern humans are from each other. The model placed them outside the modern human distribution without being told to.

![UMAP of 100 modern human mtDNA sequences (coloured by haplogroup) plus Neanderthal (gold star) and Denisovan (purple star). Ancient sequences appear at the periphery of the modern human cloud, not inside any haplogroup cluster.](http://rokpayprsizors.files.wordpress.com/2026/05/ancient_dna_umap.png)

## Where the result is honest

The nearest-neighbour analysis does not reproduce the phylogenetic tree. The top-5 nearest modern sequences to Neanderthal include P1d1 (Melanesian), B4a1a1 (East Asian), and L0a2a2a (African root), mixed without obvious phylogenetic ordering. Mean L2 to L-haplogroup sequences (African root, 0.1109) is essentially the same as to H (European, 0.1111) or D/C (East Asian, 0.1101).

This is the honest result of Phase 1 pre-training on MLM loss. The model learned k-mer frequency patterns, not haplogroup-discriminative structure. Cross-species vertebrate pre-training teaches the model what mitochondrial DNA looks like in general, not the finer branching topology of the human haplogroup tree. The separation between ancient and modern is real (1.45x), but the model can't tell H from L from D at this stage.

Phase 2 (human-specific pre-training) and haplogroup fine-tuning would be needed to reproduce phylogenetic topology. That is Day 21 and beyond.

## A note on cosine similarity

Before arriving at L2 distances, the analysis started with cosine similarity and produced pairwise values of 1.0000 for every pair. All vectors, modern and ancient, had cosine similarity of 1 to every other vector. This is a known property of pre-trained BERT-style encoders: CLS token embeddings, when mean-pooled across windows, all point in roughly the same direction in the high-dimensional space. The representations occupy a narrow cone.

This does not mean the embeddings carry no information. The L2 distances between embeddings are non-trivial (0.07 to 0.11). It means cosine is the wrong similarity metric for untrained or lightly-trained BERT embeddings. Cosine is appropriate after fine-tuning or contrastive training, which aligns the embedding space so that different classes occupy genuinely different directions. Before that, use L2.

The collapse is fixable and will resolve once haplogroup fine-tuning is applied. The fine-tuned model from Day 16 (haplogroup classification, LoRA r=8) should produce embeddings with non-trivial cosine structure.

## The zero-shot test is still meaningful

What the model got right: ancient sequences are geometrically outside the modern human distribution. They are not embedded inside the H haplogroup cluster or the L3 cluster. They are at the periphery. A model that learned nothing about evolutionary divergence would place them randomly inside the modern cloud. This one doesn't.

What the model got wrong: it can't tell that Neanderthal is more similar to African root haplogroups than to European tip haplogroups. That would require the embedding space to have directional meaning at the haplogroup level, which Phase 1 pre-training does not provide.

The separation between ancient and modern is 1.45x. This is not a spectacular result. It is an honest one. The model captured that 50,000-year-old sequences are measurably different from sequences present in the modern population, based solely on k-mer frequency patterns in a sliding-window MLM objective.

## Key takeaways

- Pre-trained BERT-style models place out-of-distribution sequences outside the training distribution in L2 space, even without any evolutionary supervision: ancient sequences are 1.45-1.48x farther from modern humans than moderns are from each other.
- Cosine similarity is the wrong metric for untrained BERT embeddings: all pairwise cosine similarities are ~1.0 because CLS mean-pooled vectors occupy a narrow cone before fine-tuning. Switch to L2 distance.
- Phase 1 MLM pre-training captures "this sequence is different" but not "which haplogroup is this closer to": haplogroup-level phylogenetic structure requires human-specific pre-training or fine-tuning.
- Variable-length ancient genomes require explicit length normalization: Neanderthal (16,565 bp) and Denisovan (16,570 bp) differ from the 16,569 bp rCRS by 4 and 1 bases respectively; the four-line fix in `embed_genome()` makes the embedder robust to any mitochondrial genome.

---
*Code and data at [github.com/vthawfeek/mtdna-foundation-model](https://github.com/vthawfeek/mtdna-foundation-model)*