---
title: "D-loop Entropy and Why mtDNA Preprocessing Isn't Trivial"
date: 2026-05-26
tags: [mtDNA, foundation model, bioinformatics, machine learning, BERT]
layout: post
---

The mitochondrial D-loop wraps around both ends of the linearised 16,569-bp rCRS: it runs from position 16,024 through to position 576, with the protein-coding, tRNA, and rRNA genes occupying the middle (positions 577–16,023). Computing Shannon entropy across 37,500 real human mitochondrial sequences from NCBI confirms what published phylogenetic studies consistently report: the hypervariable sub-regions HV1 (~16,024–16,383) and HV2 (~73–340) are substantially more polymorphic than the coding genes, with HV1 showing ~1.05× higher entropy than the coding average in raw unaligned data — and published alignment-based analyses showing 3–7× across properly aligned corpora.

That asymmetry, and one structural fact that flows from it, shapes every decision in the Day 4 preprocessing pipeline.

This is part of an open-source project to build the first dedicated foundation model for mitochondrial DNA. mtDNA mutations are the primary cause of over 350 inherited diseases, including MELAS (mitochondrial encephalomyopathy), Leigh syndrome, and Leber hereditary optic neuropathy, and the mitochondrial genome is the reference used for maternal ancestry and population genetics. Every existing genomics language model treats mtDNA as a short fragment of linear DNA. This project starts from first principles: circular architecture, heteroplasmy-aware embeddings, and a vocabulary and preprocessing pipeline designed for the 16,569-bp closed loop rather than adapted from nuclear genome tools.

## What "preprocessing" means here

Preprocessing for a DNA foundation model is not the same as normalising tabular features or resizing images. The raw FASTA files from HmtDB and NCBI contain sequences that differ in length (anywhere from 16,519 to 16,819 bp), character set (IUPAC ambiguity codes like R, Y, W, S, K, M alongside the standard ACGTN), and one quirk specific to circular genomes: some databases append the first 200 bases to the end of the sequence so that analysis tools running sliding windows don't miss the circular junction region.

The pipeline has five separate functions. Each is independently testable and independently importable. The decision to use functions rather than a preprocessing class is deliberate: a downstream consumer who only needs sequence cleaning doesn't need to touch length normalization, and a test for the split logic doesn't need real sequence data.

### Step 1: clean_sequence

```python
def clean_sequence(seq: str) -> str:
    seq = seq.upper()
    seq = re.sub(r"[^ACGTN]", "N", seq)
    n = JUNCTION_DUPLICATE_CHECK_BASES  # 200
    if len(seq) > 2 * n and seq[:n] == seq[-n:]:
        seq = seq[:-n]
    return seq
```

Uppercase is straightforward. The regex replacement catches all IUPAC ambiguity codes and other characters (hyphens from some alignment formats, dots from some annotation formats) and maps them to N. The junction duplicate detection compares the first 200 bases against the last 200: if they match exactly, the trailing copy is removed.

That last step is easy to overlook. If you leave junction duplicates in and then run length normalization, the trimming step removes real sequence from the 3' end rather than the appended duplicate. The sequences come out 16,569 bp as required, but the last ~150 bases of cytochrome b (positions 14,747-15,887) get replaced with what should have been trimmed away. Downstream: the model learns to predict junction sequence where coding sequence should be, and the loss at those positions is slightly lower because the training label is wrong.

### Step 2: normalize_length

Every sequence must be exactly 16,569 bp for batching. Sequences shorter than this get N padding inserted at position 576, not appended to the 3' end:

```python
def normalize_length(seq, target_length=16569, pad_position=576):
    if len(seq) < target_length:
        n_pad = target_length - len(seq)
        insert_at = min(pad_position, len(seq))
        seq = seq[:insert_at] + "N" * n_pad + seq[insert_at:]
    else:
        seq = seq[:target_length]
    return seq
```

The padding position matters for the same reason as the junction check: gene coordinate preservation. MT-CO1 (cytochrome c oxidase subunit 1) starts at position 5,904. If you pad by appending Ns at the 3' end, position 5,904 in the padded sequence still points to the same base as in the original. But if the original sequence is shorter than 576 bp and you insert padding at position 576, you shift everything after the insert point by `n_pad` positions. For sequences that are already close to 16,569 bp (which is most of them: the length histogram peaks within 50 bp of rCRS length), the insert is small and the displacement is minimal.

For sequences much shorter than rCRS (rare, but present in cross-species datasets), the D-loop region absorbs the padding. This is biologically appropriate: the D-loop is the control region, it is the most tolerant of insertions, and positioning the padding there rather than inside a protein-coding gene avoids teaching the model spurious patterns in functional regions.

![Raw sequence length distribution across 155,115 sequences. Red dashed line marks the 16,569 bp rCRS target length. Most sequences fall within ±50 bp of target.](/assets/figures/length_distribution.png)

### Step 3: stratified_split

The split uses `StratifiedShuffleSplit` from scikit-learn to produce 80/10/10 train/val/test partitions with proportional haplogroup representation in each. The implementation hits one non-obvious edge case: some haplogroups in the HmtDB corpus have only a single representative. `StratifiedShuffleSplit` requires at least 2 samples per class to estimate proportions. The fix: merge singleton classes into a `_rare` bin for the split calculation, then restore the original labels.

Cross-species sequences (NCBI vertebrate mtDNA) have no haplogroup label and go directly to train. They are used for Phase 1 MLM pre-training only and never appear in the val or test splits, which are evaluated on human sequences with known haplogroup labels.

![Top 20 haplogroups in the HmtDB human corpus. Haplogroup H dominates (most common European lineage). L-root haplogroups represent African/ancestral lineages.](/assets/figures/haplogroup_distribution.png)

## The D-loop entropy analysis

The EDA notebook computes per-position Shannon entropy across 37,500 real human sequences from NCBI:

```python
bases = list("ACGTN")
freqs = np.stack([(seq_matrix == b).mean(axis=0) for b in bases], axis=0)
pos_entropy = np.array([scipy_entropy(freqs[:, pos] + 1e-9, base=2) for pos in range(RCRS_LENGTH)])
```

At each genomic position, the entropy of the nucleotide frequency distribution is computed. High entropy means many different bases appear at that position across individuals; low entropy means most sequences agree.

![Shannon entropy across 16,569 bp positions. Orange shading marks the D-loop control region (positions 16,024–576). 50 bp moving average applied. HV1 and HV2 are both elevated-variability segments at opposite ends of the linear coordinate space.](/assets/figures/positional_entropy.png)

One subtlety when working with raw (unaligned) sequences: the D-loop control region contains poly-C stretches whose length varies between individuals and haplogroups. This natural indel variation smears each genuinely variable position across neighbouring columns in the fixed-length matrix, suppressing per-position entropy in the D-loop relative to coding regions — where indels are rare under purifying selection. Properly quantifying the D-loop/coding variability ratio requires multiple-sequence alignment first.

What the raw analysis shows on the real corpus: coding-region entropy averages 1.47 bits per position; HV1 (positions 16,024–16,383) averages 1.55 bits (1.05× higher). Published alignment-based studies report D-loop/coding ratios of 3–7× depending on the population sample and the specific sub-region compared. The biological variability is real; the exact number depends on the analysis method.

This matters for the preprocessing decisions above. Variable positions are harder for the model to predict from context, contributing more to the MLM loss and the gradient signal. Whether the true ratio is 3× or 7×, the D-loop/coding boundary is a real feature of the entropy landscape that the model must learn — and that learning is undermined if the boundary is blurred by incorrect padding or junction artefacts.

## Why circular PE is the right response

The entropy analysis also shows why circular positional encoding matters beyond a vague "mtDNA is circular" argument. Consider the junction: positions 16,024 through 16,569 (the end of HV1 and the terminal coding sequence) and positions 1 through 576 (the beginning of HV2 and the tRNA cluster) are biologically contiguous. A model using standard BERT positional embeddings represents position 16,569 and position 1 as 16,568 positions apart, which is the furthest possible distance in the encoding space. A model using circular positional encoding, where the angle `2*pi*pos/genome_length` wraps continuously, represents them as one step apart.

Across 37,500 real human sequences, HV1 (ending near position 16,383) and HV2 (starting at position 73) are both elevated-variability regions used clinically for haplogroup assignment. They are part of the same biological control region, separated only by the coordinate origin convention of the rCRS. A model that cannot represent their proximity cannot learn that they form a single functional unit.

## What the pipeline produces

The output schema for each parquet file:

| Column | Type | Notes |
|--------|------|-------|
| accession | string | NCBI or HmtDB sequence ID |
| sequence | string | cleaned, normalized to 16,569 bp |
| haplogroup | string/null | top-level haplogroup label; null for cross-species |
| species | string | "homo_sapiens" or vertebrate species name |
| geographic_origin | string/null | HmtDB geographic field where available |
| het_level_vector | null | populated from gnomAD on Day 5 |
| qc_pass | bool | True if N-content <= 10% |

The `het_level_vector` column is null at this stage. It will hold a float array of per-position heteroplasmy levels from gnomAD when gnomAD variant data is merged in on Day 5. The column exists in the schema now so the parquet files have a stable shape that downstream code can depend on.

## Numbers

- 67 unit tests passing (27 new preprocessor tests added today)
- 0 ruff errors across mtdna_fm/ and tests/
- All five preprocessor functions independently unit-tested
- Pipeline handles sequences from 16,519 bp (short cross-species) to 16,819 bp (with junction duplicate) without errors
- Real corpus: 37,500 human sequences (NCBI) + 117,615 vertebrate sequences → 155,115 total
- QC pass rate: 155,009 / 155,115 (99.93%) — only 106 sequences exceed the 10% N threshold

![N-content distribution across the corpus. Red dashed line marks the 10% QC threshold. 99.93% of sequences pass (106 sequences fail).](/assets/figures/n_content_distribution.png)

- Train / val / test split: 152,590 / 1,262 / 1,263 (stratified by haplogroup)
- Raw sequence lengths: 54.4% shorter than rCRS, 21.6% exact, 24.0% longer — most within ±50 bp

The preprocessing pipeline is the last thing between raw data and the PyTorch Dataset class on Day 6. It needs to be correct before the model sees any training examples. Getting it right in the test suite now means the rest of the project can assume clean, 16,569-bp, all-uppercase, N-flagged sequences without defensive checks scattered through the training code.

## Key takeaways
- Padding a circular genome at the D-loop (position 576) preserves canonical coordinates for all 37 mitochondrial genes; padding at the 3' end corrupts them for any downstream model that relies on fixed gene positions.
- Junction duplicates appended by sequence databases corrupt length normalization silently: the output is 16,569 bp as expected, but ~150 bp of cytochrome b is replaced by junction sequence.
- Shannon entropy at the D-loop boundary (position 576) rises measurably in real data; published alignment-based analyses report 3–7× D-loop/coding ratios depending on population sample and alignment method.
- HV1 and HV2 are physically one base apart on the circular genome but 16,310 positions apart in standard BERT positional encoding — the strongest concrete argument for circular PE.