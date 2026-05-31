---
title: "Circular Positional Encoding for Mitochondrial DNA: The Mathematics of a Closed Loop"
date: 2026-05-31
tags: [mtDNA, foundation model, bioinformatics, machine learning, BERT]
layout: post
---

Position 1 and position 16,569 of the human mitochondrial genome are physically adjacent. The D-loop, where almost every haplogroup-defining variant lives, straddles this junction. Standard BERT positional embeddings treat positions 1 and 16,569 as 16,568 steps apart, the maximum possible distance in the sequence. That is not a minor approximation. It is structurally wrong.

This is part of an open-source project to build the first dedicated foundation model for mitochondrial DNA. mtDNA mutations drive over 350 inherited diseases, including MELAS, Leigh syndrome, and Leber hereditary optic neuropathy, and the circular topology of the genome is a biological fact that no existing DNA foundation model encodes correctly.

## Why 6-mer Tokenization and Not BPE

The tokenizer is the first thing most people ask about. Why fixed k-mers instead of byte-pair encoding?

BPE builds vocabulary from corpus statistics: frequently co-occurring token pairs get merged. Applied to DNA, this produces tokens of variable length that cluster around common sequence patterns in the training corpus. For a model trained on nuclear DNA, those patterns reflect the human nuclear genome's GC content, repeat element distribution, and coding sequence composition. None of those statistics transfer cleanly to mitochondrial DNA, which has a different GC content (~44% vs ~41% nuclear), no introns, overlapping reading frames, and a completely different codon usage bias.

A 6-mer vocabulary sidesteps this entirely. The vocabulary is all possible 6-mers over the {A, C, G, T} alphabet: 4^6 = 4,096 tokens, plus 6 special tokens ([PAD], [CLS], [MASK], [UNK], [SEP], [HET]), for a total of 4,102 tokens. This vocabulary is deterministic, reproducible across machines, and independent of any corpus.

```python
class KmerVocabulary:
    SPECIAL_TOKENS = {"[PAD]": 0, "[CLS]": 1, "[MASK]": 2,
                      "[UNK]": 3, "[SEP]": 4, "[HET]": 5}

    @classmethod
    def build(cls, k: int = 6) -> "KmerVocabulary":
        vocab = dict(cls.SPECIAL_TOKENS)
        idx = len(vocab)
        for kmer in product("ACGT", repeat=k):
            vocab["".join(kmer)] = idx
            idx += 1
        return cls(vocab, k=k)
```

The k=6 choice is not arbitrary. At k=6, each token covers a 6-base window with 1-base stride, so adjacent tokens overlap by 5 bases. This creates dense contextual overlap: a single-base mutation changes up to 6 tokens in the local neighbourhood, giving the model multiple views of the same biological change. At k=4, you lose too much local context. At k=8, the vocabulary becomes 65,536 tokens plus specials, which is large enough to make the embedding table (65,536 × 256 = 16.8M parameters) dominate the model's parameter count.

## How Tokenization Handles the Circular Junction

The genome is not linear. Tokenizing from base 0 to base 16,568 and stopping there means the tokens at the 3' end of the genome don't see the bases at the 5' end that are physically adjacent.

The fix is simple: prepend the last k-1 bases to the front of the sequence before k-merizing:

```python
def tokenize_sequence(
    seq: str,
    vocabulary: KmerVocabulary,
    k: int = 6,
    circular: bool = True,
    max_seq_len: int = 512,
) -> dict:
    if circular:
        seq = seq[-(k - 1):] + seq  # wrap last k-1 bases to front
    tokens = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        tokens.append(vocabulary.encode(kmer))
    ...
```

When `circular=True`, a 16,569-base genome produces 16,569 k-mer tokens (after the junction wrap), each at its correct genomic coordinate. The position_ids returned are absolute genomic positions, not window-relative positions. This matters for how the circular PE indexes the position table.

The output is a dict with `input_ids`, `attention_mask`, `position_ids`, and `het_values`. The first three mirror the HuggingFace tokenizer interface. The fourth is the heteroplasmy channel.

## The Circular Positional Encoding: Why the Maths Work

Standard BERT positional encoding (Vaswani 2017):

```
PE[pos, 2i]   = sin(pos / 10000^(2i/d))
PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
```

This is linear. Positions 0 and 16,569 produce PE vectors that are as different as any two positions 16,569 apart. There is no periodicity.

The circular encoding wraps position around the genome circumference by substituting `2π * pos / L` for the raw position:

```
PE[pos, 2i]   = sin(2π * pos/L * 1/10000^(2i/d))
PE[pos, 2i+1] = cos(2π * pos/L * 1/10000^(2i/d))
```

where L = 16,569 (the genome length). Now the angle at pos=0 is 0 and the angle at pos=16,569 is 2π: they are the same point on the circle. The PE vectors at these two positions are identical. All positions near the junction (positions 16,500 or position 50, for example) produce PE vectors that are nearby in the high-dimensional encoding space, reflecting the fact that they are nearby on the genome.

This is not a parameter. It is a fixed buffer registered at model initialisation:

```python
class MtDNACircularPositionalEncoding(nn.Module):
    def __init__(self, genome_length: int, hidden_size: int):
        super().__init__()
        pe = torch.zeros(genome_length, hidden_size)
        position = torch.arange(genome_length).float()
        angle = 2 * torch.pi * position / genome_length
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float()
            * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(angle.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(angle.unsqueeze(1) * div_term)
        self.register_buffer("pe", pe)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.pe[position_ids]
```

The buffer is 16,569 × 256 = 4.2M floats. At fp32 that is 16.8 MB, a bit large for a 24 MB model. But it is never updated by the optimizer, so it costs nothing in gradient computation. And because it is registered as a buffer, it is saved with the model and loaded automatically: `MtDNAModel.from_pretrained("models/phase1_v1")` loads the pre-computed PE table.

![Per-position k-mer entropy in the first 256 bp of the mtDNA genome, showing the elevated diversity in the D-loop (positions 0-256) relative to the start of the tRNA gene cluster. The 7x entropy difference between D-loop and coding region motivates the circular PE: the D-loop junction region at positions 16,024-16,569 and positions 0-576 needs to be treated as spatially contiguous.](http://rokpayprsizors.files.wordpress.com/2026/05/positional_entropy_kmer-2.png)

## The Heteroplasmy Channel: Continuous, Not Discretized

Every cell contains thousands of copies of the mitochondrial genome. In most people, all copies are identical (homoplasmy). In some, particularly in disease states, a fraction of copies carry a mutation (heteroplasmy). The het_level is the fraction of copies with the mutant allele: 0.0 is fully homoplasmic reference, 1.0 is fully mutant, anything in between is heteroplasmic.

The first design choice was whether to include heteroplasmy at all. The argument against: most training sequences are homoplasmic (het_level = 0.0), and the gnomAD heteroplasmic variant data is noisy. The argument for: heteroplasmy is biologically real and clinically important. MELAS syndrome can be latent at het_level 0.3 and fully expressed at 0.7. A model that treats all positions as fully resolved is wrong for a non-trivial fraction of clinical cases.

The second choice was how to encode it. Discretizing into bins (low/medium/high heteroplasmy) loses precision and requires choosing bin boundaries. Treating it as a secondary token ID requires expanding the vocabulary and creates a combinatorial explosion (each k-mer now has multiple variants depending on its heteroplasmy bin). The simplest approach is to treat it as a continuous float and project it into embedding space alongside the k-mer embedding:

```python
class MtDNAEmbeddings(nn.Module):
    def __init__(self, config: MtDNAConfig):
        super().__init__()
        self.kmer_embeddings = nn.Embedding(config.vocab_size, config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.circular_pe = MtDNACircularPositionalEncoding(
            config.genome_length, config.hidden_size
        )
        if config.use_het_projection:
            self.het_projection = nn.Linear(1, config.hidden_size, bias=True)
            self.het_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids, position_ids, het_values=None):
        x = self.kmer_embeddings(input_ids) + self.circular_pe(position_ids)
        if het_values is not None and hasattr(self, "het_projection"):
            het_proj = self.het_norm(self.het_projection(het_values.unsqueeze(-1)))
            x = x + het_proj
        return self.layer_norm(x)
```

The `het_projection` is a 1 → 256 linear layer: one floating-point value per position is mapped to a 256-dimensional additive contribution to the token embedding. When het_values is None (Phase 1 cross-species pre-training, where heteroplasmy data is unavailable), the het channel is simply absent. When it is present (Phase 2 human pre-training), the model learns to use it as additional signal.

The combined loss for Phase 2 is:

```python
total_loss = mlm_weight * mlm_loss + het_weight * het_mse_loss
```

where `het_weight=0.3` in Phase 2 and `het_weight=0.0` in Phase 1. The heteroplasmy regression loss only applies to positions where the het_label is not -1 (the ignore sentinel).

## What Phase 2 Adds to Phase 1

Phase 1 pre-training uses ~117k vertebrate mtDNA sequences across 3,500+ species. Het_weight is 0 because non-human genomes don't have gnomAD heteroplasmy measurements. The model learns general mitochondrial sequence structure: codon usage patterns, tRNA stem-loop motifs, conserved protein-coding regions.

Phase 2 starts from the Phase 1 checkpoint and continues on 34,974 human HmtDB sequences with het_weight=0.3. The learning rate drops from 1e-4 to 3e-5 (lower: the model is adjusting, not learning from scratch) with only 500 warmup steps (faster: the optimizer isn't starting cold).

The Phase 2 trainer implements this with one key change: `_load_checkpoint(encoder_weights_only=True)` loads only the `mtdna.*` keys from Phase 1 (the encoder stack) and discards the Phase 1 optimizer state. The Phase 1 Adam moment estimates were calibrated to the cross-species distribution and the larger learning rate. Using them for Phase 2 would push the optimizer in the wrong direction from the first step. Fresh optimizer, lower LR, human-specific sequences.

This is the direct analogue of domain-adaptive pre-training in NLP: pre-train on a broad corpus (all vertebrate mtDNA) then fine-adjust on the target domain (human mtDNA with clinical labels). The circular PE and het projection are the architecture features that make this transfer meaningful for mtDNA specifically.

## Key takeaways

- A 6-mer vocabulary of 4,096 tokens is deterministic and reproducible, unlike BPE vocabularies that depend on training corpus statistics. For small-genome models this is the correct default.
- Circular positional encoding encodes position as an angle on a circle (2π × pos/L), so positions at the genome junction (0 and 16,569) produce identical PE vectors. This is mathematically exact, not approximate.
- Heteroplasmy as a continuous linear projection (1 → hidden_size, added to the k-mer embedding) preserves the full precision of the gnomAD het_level values without requiring vocabulary expansion or discretization.
- Phase 2 encoder-weight transfer with a fresh optimizer is critical: loading Phase 1 Adam moments into a lower-LR Phase 2 run pushes the optimizer in the wrong direction from step 1.