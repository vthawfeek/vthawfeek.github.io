---
title: "A glass box for clinical-trial language models: five controls, every step visible"
date: 2026-07-17
tags: [machine learning, LLM, tokenization, clinical trials, interpretability]
layout: post
---
Most of us meet language models as black boxes. Text goes in, text comes out, and the machinery in between gets described with metaphors. I wanted the opposite: a model where every step is visible and checkable, built on text I care about, clinical-trial records. This is the write-up. Try the live demo: https://glass-llm.streamlit.app

One disclaimer up front, because it matters. The model here is about 1.3M parameters (d_model 128, 4 layers, 4 heads, context 128). It is not smart, and it is not meant to be. It produces trial-shaped text with invented specifics, every generation is watermarked synthetic, and it is not medical information. The point is transparency, not capability.

## Built from scratch, on purpose

Two components, both written by hand so there is nothing to hide behind.

A byte-level BPE tokenizer, about 150 lines, no tiktoken or HuggingFace. It learns its vocabulary by repeatedly merging the most frequent adjacent pair of pieces.

A decoder-only transformer, about 1.3M parameters, trained on ClinicalTrials.gov text using all the curated sections: title, conditions, phase, interventions, brief summary, detailed description, primary outcomes, and eligibility criteria. The attention weights and hidden states are plumbed out so the dashboard can show them.

## Five controls, and what each one does

The dashboard is a collapsible panel with five dropdowns. Every dropdown selects a real trained model from a small zoo trained offline, and changing it re-renders the tokenization, the embedding map, the attention heatmaps, the next-token probabilities, and the generated text with the model's actual numbers. Hover the architecture diagram and each step shows its math in plain English.

Training data (clinical trials or general English). The domain model speaks trial; the model trained on English novels drifts into story language on the same prompt.

Data volume (1, 8, or 32 MB). This is the finding I like most, because it is counterintuitive. Same architecture, same tokenizer, same 4,000 steps, only the data changes. The 1 MB model looks the most fluent ("Age >= 18 to 80 years old") but scores the worst on held-out text: bits-per-byte of 2.24, 1.49, and 1.47 as data grows from 1 to 8 to 32 MB. With only 1 MB it passes over the corpus about 30 times and memorises it. Fluent is not the same as learned, which is why we measure held-out perplexity instead of eyeballing output.

Attention heads (4 or 8). The heatmaps change, and the effect on held-out loss is small and honestly non-monotonic. On the tiny-data generic model, 8 heads did slightly worse — 2.64 to 2.78 bits-per-byte — more parameters overfitting harder on 1 MB of text. But on the tiny-data domain model the same change went the other way and helped, 2.24 down to 2.15. So the direction is not a property of the architecture; it depends on the data underneath it. A useful reminder that more is not automatically better, and that "better" here is worth checking rather than assuming.

Fine-tuning (biomarker tagging). More on this below; it is where I was most careful.

Retrieval (RAG). Hallucination against grounding, also below.

## Tokenization: real, but easy to overclaim

A general-purpose tokenizer fragments biomedical terms. Pembrolizumab is 4 tokens with my clinical tokenizer (vocab 4,096), 6 with GPT-4's (p, emb, rol, iz, um, ab), 5 with GPT-4o, 6 with GPT-2 and BERT.

Three claims, not equally solid. Cost and context are facts: a full eligibility criterion is 30 tokens with the domain tokenizer against 55 with a general tokenizer of the same vocabulary size, cheaper with more usable context. "Domain always wins" is false: GPT-4's 100k-plus vocabulary is competitive, averaging 4.22 tokens per term against my 4.94 on a 90-term mixed probe set, so the clean win is domain against general at equal vocab (20-45% fewer), not "beats GPT-4". And quality is a flag, not a verdict: fragmentation is a risk to test for rare and exact-match terms, not proof of worse understanding.

The audit panel makes this practical. Paste your own drug names and gene variants and see how GPT-4, GPT-4o, GPT-2, and BERT split them and what that costs. It is a cheap check, and it is why PubMedBERT trained its own vocabulary (https://arxiv.org/abs/2007.15779); the trade-offs are documented in the tokenizer literature (https://arxiv.org/abs/2310.08754).

## Fine-tuning, honestly: a mechanism, not an extractor

I wanted the fine-tuning control to do something real, so I tried to fine-tune the model to extract biomarkers (EGFR, PD-L1, BRAF V600E) from trial text. It can't, and neither can any model this size. Structured biomarker extraction is a genuinely hard task where the state of the art is fine-tuned 7B-parameter models (Alkhoury et al., npj Digital Medicine 2025, https://pmc.ncbi.nlm.nih.gov/articles/PMC12053753/), and even GPT-4 struggles out of the box.

So I did the honest version. A small task head sits on top of the frozen model, trained with distant supervision from an oncology gene and biomarker lexicon to tag biomarker mentions. The dashboard shows the mechanism, fine-tuning bolting a task onto a base model and lighting up biomarker tokens, and it labels it plainly: the head imitates the lexicon, misses novel terms, and mishandles negation like "EGFR wild-type". It demonstrates how fine-tuning works. It is not a biomarker extractor, and saying so is the point.

## Hallucination against grounding

The retrieval panel is honest about the model's limits. Give it a topic and it will generate a fluent, plausible, completely invented trial. Turn on retrieval, a transparent cosine vector index over 500 real trials (5,960 chunks, the same nearest-neighbour operation FAISS or Chroma perform, kept visible), and the panel shows real records with their NCT IDs and similarity scores. A 1.3M model can't synthesise grounded prose the way a large model would, so the panel shows the mechanism of grounding and says so. One prompt drives every panel, so the same input you tokenize and attend over is the one that retrieves.

## What is new here, and what isn't

Overclaiming is how these projects lose credibility, so let me be plain. Interactive LLM explainers already exist and are excellent (Transformer Explainer is one). Domain tokenization is well established (PubMedBERT). A small model can't extract biomarkers or answer questions; it hallucinates, which is exactly why the retrieval panel exists. What I hadn't seen in one place is the whole thing end to end: a from-scratch tokenizer and model, five controls over data, data volume, architecture, fine-tuning, and retrieval, a live tokenizer audit against real production tokenizers, and a transparent retrieval layer, on clinical-trial text, with every number reproducible. That combination is the contribution.

## Reproduce it

The code (https://github.com/vthawfeek/glass-llm) runs end to end: pull ClinicalTrials.gov, train the tokenizer, train the model zoo, warm the audit tokenizers, launch the dashboard. Seeds are fixed and configs are logged.

Open the glass box: https://glass-llm.streamlit.app
