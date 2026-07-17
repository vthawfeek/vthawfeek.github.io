---
title: "Two things a 1.3-million-parameter clinical language model taught me"
date: 2026-07-17
tags: [machine learning, LLM, tokenization, clinical trials, interpretability]
layout: post
---
I built a small language model from scratch on ClinicalTrials.gov text: my own byte-pair tokenizer, my own decoder-only transformer, about 1.3 million parameters. I did it so I could open up every step and look at it. The model is deliberately tiny. It produces trial-shaped text with invented specifics, everything it writes is watermarked synthetic, and it is not medical information. The goal was never a smart model. It was a transparent one, and a transparent model turns out to be a good teacher.

Two findings stood out. Neither is new science. Both are the kind of thing that is easy to say and much better to watch.

## 1. Less data can look better, and that is the trap

I trained the same architecture, with the same tokenizer and the same 4,000 training steps, on three sizes of clinical-trial text: 1 MB, 8 MB, and 32 MB. The only thing that changed was how much data the model saw.

Here is the counterintuitive part. The 1 MB model often produced the smoothest-looking output, clean lines like "Age >= 18 to 80 years old". Judged by eye, you would call it the best of the three.

The held-out numbers say the opposite. Measured in bits-per-byte on trials the model never saw during training (lower is better):

| Training data | Bits-per-byte (held-out) |
|---|---|
| 1 MB | 2.24 |
| 8 MB | 1.49 |
| 32 MB | 1.47 |

The 1 MB model looks fluent because, in 4,000 steps, it passes over that tiny corpus about 30 times and memorises it. It is not generalising, it is reciting. On new text it is the worst of the three.

This is why we measure held-out perplexity instead of trusting our eyes. Fluent is not the same as learned. A model that has memorised its training set will demo beautifully and fail in production, and in exactly the settings biomedicine cares about (rare diseases, a novel target, a single site's records), small data makes that the default rather than the exception. In the dashboard you can move the data-volume control and watch the memorisation appear at the low end.

## 2. Tokenization is real, but easy to overclaim

Before a model reads a single character of meaning, GPT-4 included, it chops the text into tokens: sub-word fragments from a fixed vocabulary. The step is invisible, it happens before the "AI" part, and on specialised text it quietly costs money and context.

Here is how real, production tokenizers split one drug name, pembrolizumab:

| Tokenizer | Tokens | Pieces |
|---|---|---|
| GPT-4o (o200k) | 5 | p emb rol iz umab |
| GPT-4 / GPT-3.5 (cl100k) | 6 | p emb rol iz um ab |
| GPT-2 | 6 | p emb rol iz um ab |
| BERT (WordPiece) | 6 | pe mb rol iz uma b |
| My domain tokenizer (clinical, vocab 4,096) | 4 | p emb rol izumab |

Three consequences follow, and they are not equally solid. Separating them is where credibility lives.

Cost and context are facts. Commercial LLMs bill per token and latency scales with tokens; context windows are counted in tokens. On a full eligibility criterion, my domain tokenizer used 30 tokens against 55 for a general-English tokenizer of the same vocabulary size, roughly 45% cheaper with more room for real content. That part is arithmetic.

"Domain always wins" is false, and I want to be exact about it. GPT-4's 100k-plus vocabulary is genuinely competitive. Across a mixed probe set of 90 drug names, gene variants, and eligibility phrases, GPT-4 averaged 4.22 tokens per term against my domain tokenizer's 4.94. A big vocabulary buys back most of the specialised advantage. The clean, controlled result is domain against general at equal vocabulary size (20-45% fewer tokens on biomedical text). It is not "beats GPT-4".

Quality is a flag, not a verdict. It is tempting to conclude that the model also understands fragmented terms worse. Sometimes it does, for rare terms and character-sensitive tasks like exact matching or normalisation. But large models with a lot of training data often reassemble the fragments fine. So treat heavy fragmentation as a reason to test, not as proof of worse quality.

None of this is new. Domain-specific tokenization is why PubMedBERT trained its vocabulary from scratch (https://arxiv.org/abs/2007.15779), and the trade-offs are documented across the tokenizer literature (https://arxiv.org/abs/2310.08754). What I wanted was not a new finding. It was to make the invisible step visible and checkable, on the kind of text drug-discovery teams work with.

## You can look for yourself

So I built a glass-box dashboard. One panel lets you paste your own vocabulary, your drug names and gene variants, and see how GPT-4, GPT-4o, GPT-2, and BERT fragment it and what that costs in tokens and context. Another lets you move the data-volume control and watch a small model tip from generalising into memorising.

The model behind it is tiny on purpose. It demonstrates the mechanism, not intelligence, and everything it generates is watermarked synthetic and is not medical information. This project is not about a smart model. It is about a transparent one.
