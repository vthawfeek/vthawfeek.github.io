---
title: "Why Mitochondrial DNA Needs Its Own Foundation Model"
date: 2026-05-31
tags: [mtDNA, foundation model, bioinformatics, machine learning, BERT]
layout: post
---

The human mitochondrial genome is 16,569 base pairs. Position 1 and position 16,569 are physically adjacent -- they share a phosphodiester bond in the circular chromosome. DNABERT2, HyenaDNA, and Nucleotide Transformer all treat these two positions as 16,568 positions apart.

That is not a minor inaccuracy. It is a structural misrepresentation of the genome. This post explains why it matters, what heteroplasmy adds to the problem, and how this project addresses both.

Mitochondria carry their own genome -- a remnant of the bacterial ancestor that was engulfed by an ancestral eukaryote roughly 1.5 billion years ago. Mutations in that genome cause over 350 inherited diseases, including MELAS (mitochondrial encephalomyopathy), Leigh syndrome, and Leber hereditary optic neuropathy. Copy number changes are biomarkers for cancer and aging. And yet no sequence model has been designed specifically for the properties that make this genome unusual. That gap is the premise of this project.

## The Circular Topology Problem

Standard BERT uses sinusoidal positional encoding. Position embeddings are computed as:

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
```

For a linear sequence, this is fine. The embedding at position 0 and the embedding at position 16,568 are maximally dissimilar, which correctly reflects that these positions are far apart in a linear genome.

But mtDNA is not linear. The D-loop control region spans the junction: it starts near position 16,024 and wraps around through position 576. Transcription initiation sites sit on both sides of that junction. If the model treats position 16,569 as far from position 1, any attention head trying to model D-loop regulation is working with corrupted positional signals.

The fix is to replace the linear angle with a circular one:

```
PE[pos, 2i]   = sin(2pi * pos / 16569 * 1/10000^(2i/d))
PE[pos, 2i+1] = cos(2pi * pos / 16569 * 1/10000^(2i/d))
```

The key change: `pos / 10000^(...)` becomes `(2pi * pos / genome_length) / 10000^(...)`. At `pos = 0` and `pos = 16569`, the angles are identical, making those positions positionally indistinguishable in the encoding -- which is correct.

This is implemented as a non-learnable buffer:

```python
class MtDNACircularPositionalEncoding(nn.Module):
    def __init__(self, genome_length: int, hidden_size: int):
        super().__init__()
        pe = torch.zeros(genome_length, hidden_size)
        position = torch.arange(genome_length).float()
        angle = 2 * torch.pi * position / genome_length
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() *
            (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(angle.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(angle.unsqueeze(1) * div_term)
        self.register_buffer("pe", pe)
```

Non-learnable because circular topology is a biological fact. The model should not be able to unlearn it under gradient pressure.

## Heteroplasmy: When Sequence Identity Breaks Down

Every standard DNA sequence model assumes that each position has one definitive base. That assumption fails for mitochondrial genetics.

Human cells contain hundreds to thousands of mitochondria, each carrying multiple copies of the mtDNA genome. A mutation that arises in one copy will initially be present in a minority of those copies. The result is a mixed population: some copies carry the wild-type base, others carry the mutant. This state is called heteroplasmy, and the fraction of mutant copies is the heteroplasmy level.

Heteroplasmy is clinically critical because the same mutation at different heteroplasmy levels produces different disease severity. The m.3243A>G mutation -- the most common cause of MELAS -- is asymptomatic at low levels and causes severe encephalomyopathy above roughly 70-80%. A model that receives only the sequence and not the heteroplasmy level is missing the feature that predicts clinical outcome.

Existing DNA models cannot represent this. They take a single sequence as input. There is no channel for a continuous per-position value.

mtDNA-FM addresses this with a heteroplasmy projection channel. Alongside the k-mer token IDs, the model accepts a float vector of shape `(seq_len,)` where each value is the local heteroplasmy level, 0.0 to 1.0. This vector is projected into the embedding space and added to the token embeddings before the transformer layers:

```python
# In MtDNAEmbeddings.forward():
kmer_embeds = self.kmer_embedding(input_ids)
pos_embeds = self.circular_pe(position_ids)
if het_values is not None:
    het_embeds = self.het_projection(het_values.unsqueeze(-1))
    return self.layer_norm(kmer_embeds + pos_embeds + het_embeds)
return self.layer_norm(kmer_embeds + pos_embeds)
```

This follows the same pattern as the expression value channel in single-cell foundation models, where continuous gene expression levels are projected alongside discrete gene identifiers. The design treats heteroplasmy as a continuous biological signal, not a discretized categorical state.

## The D-Loop Is Not Like the Rest of the Genome

The mitochondrial D-loop (displacement loop) control region spans roughly positions 576 to 16,024 and contains both promoters and the origin of replication. It is also the most variable region of the mtDNA genome by a wide margin.

The exploratory data analysis notebook (`notebooks/01_data_exploration.ipynb`) computed per-position Shannon entropy across 47,000 human mitochondrial sequences from HmtDB. The D-loop positions show entropy roughly 7 times higher than positions in protein-coding regions such as MT-ND1 or MT-CO1.

![Shannon entropy across 16,569 bp positions. Orange shading marks the D-loop control region (positions 16,024–576). 50 bp moving average applied. D-loop positions reach entropy ~7x higher than coding regions such as MT-ND1 or MT-CO1.](http://rokpayprsizors.files.wordpress.com/2026/05/positional_entropy-3.png)

This matters for masking strategy during pre-training. Standard BERT randomly masks 15% of tokens. If that masking is applied uniformly, the model sees D-loop positions masked far more often than their baseline variability warrants. But the D-loop also contains a homopolymeric C-tract (positions 303-315) that is almost entirely sequencing noise -- the polymerase slips on runs of C, producing variable-length artifacts that are not biologically meaningful. Masking those positions and training the model to predict them would teach it to model technical artifact, not biology.

The masking collator (`MtDNAMaskingCollator`) maintains a blacklist of these positions. Blacklisted positions are never selected for masking, regardless of the 15% sampling rate.

## Why Existing Models Are the Wrong Tool

DNABERT2, HyenaDNA, and Nucleotide Transformer were all trained on nuclear DNA. They are capable models that work well for the problems they were designed for. But they share three properties that limit their usefulness for mtDNA:

**Linear positional encoding.** All three treat the sequence as linear. For nuclear chromosomes, which are linear, this is correct. For the circular mtDNA genome, it is not. No amount of fine-tuning corrects a fundamental topological mismatch in the encoding.

**No heteroplasmy channel.** None accept a continuous per-position signal. A model that can only distinguish "A" from "C" at a position cannot represent "30% A, 70% C."

**Nuclear bias in pre-training.** Nuclear DNA has different compositional properties, different repeat structures, and a very different evolutionary conservation pattern from mtDNA. Human mtDNA is maternally inherited with no recombination, producing strong haplogroup structure that does not exist in the nuclear genome. A model pre-trained on nuclear DNA learns representations appropriate for nuclear sequence. Fine-tuning on mtDNA adjusts the weights, but it cannot fully reorient representations that were shaped by 3 billion bases of the wrong genome.

Pre-training from scratch on vertebrate mtDNA (~117k sequences, cross-species for broader diversity) and then human mtDNA (~35k HmtDB sequences, Phase 2) means the representations are grounded in the right biology from the first gradient step.

## Week 1 in Numbers

Seven days of infrastructure produced the following verified state:

- 101 tests passing, 0 failures
- 152,484 training sequences (117k cross-species vertebrate + 35k human, windowed at 512 tokens during training)
- All 152k sequences exactly 16,569 bp after normalization
- `KmerVocabulary.build(k=6)` produces 4,102 tokens deterministically; encode/decode roundtrip verified
- CI pipeline active on GitHub: two jobs (`lint`, `test`), triggered on every push

![Top 20 haplogroups in the 34,974-sequence HmtDB training corpus. Haplogroup H dominates (most common European lineage). L-root haplogroups represent African ancestral lineages. The stratified split preserves this distribution across train/val/test.](http://rokpayprsizors.files.wordpress.com/2026/05/haplogroup_distribution-2.png)

The model architecture is next. The circular positional encoding is already specified. The two-phase pre-training curriculum (cross-species first, human second) starts in Week 2.

## Key takeaways

- Circular PE makes positions 1 and 16,569 equidistant in encoding space, which is the correct representation of the mtDNA junction; standard sinusoidal PE cannot do this.
- The heteroplasmy channel projects a continuous per-position float into the embedding, allowing the model to condition on the fraction of mutant copies alongside the sequence identity.
- D-loop Shannon entropy is roughly 7x higher than coding regions, and the homopolymeric C-tract (positions 303-315) is blacklisted from masking because it reflects sequencing noise rather than biological variation.
- Nuclear-DNA pre-trained models transfer to mtDNA with fundamental topological and compositional mismatch; pre-training on vertebrate mtDNA from scratch avoids this at the cost of a smaller pre-training corpus.