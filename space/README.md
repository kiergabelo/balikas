---
title: Balikas
emoji: "\U0001F6E1\uFE0F"
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: "5.0.0"
app_file: app.py
pinned: false
tags:
  - hate-speech-detection
  - filipino
  - tagalog
  - cebuano
  - low-resource-nlp
  - cross-lingual
language:
  - tl
  - ceb
license: apache-2.0
short_description: Tagalog hate speech classifier with Cebuano zero-shot eval
---

# Balikas — Filipino Hate Speech Detection

**XLM-RoBERTa-base** fine-tuned on the Cabasag et al. (2019) Tagalog hate speech
corpus, with zero-shot cross-lingual evaluation on a hand-curated Cebuano set.

## Headline numbers

| # | Model | Eval | F1 (hate) | Macro-F1 |
|---|---|---|---|---|
| 1 | TF-IDF + LogReg | 4,232 Tagalog tweets | 0.7489 | 0.7637 |
| 2 | TF-IDF + LogReg | 40 Cebuano sentences | 0.8333 | 0.7917 |
| 3 | XLM-RoBERTa (fine-tuned) | 4,232 Tagalog tweets | _TBD_ | _TBD_ |
| 4 | XLM-RoBERTa → zero-shot CEB | 40 Cebuano sentences | _TBD_ | _TBD_ |

Rows 3–4 come from running `train/finetune_xlm.py` on Colab T4 (~25 min).
No Cebuano training data was used — the transfer is zero-shot via XLM-RoBERTa's
shared subword vocabulary across Tagalog and Cebuano.

## Why this project

Filipino (Tagalog + the Bisayan languages) is low-resource for content
moderation. Big-vendor classifiers cluster it with English. No labeled
Bisaya hate-speech corpus exists on HuggingFace at time of writing
(verified June 2026). Balikas ships an honest cross-lingual study rather
than another API-wrapping demo.

## Methodology

The TF-IDF + LogReg baseline (rows 1–2) was trained in ~0.7s on CPU. The
XLM-RoBERTa fine-tune (rows 3–4) uses the cross-lingual transformer's shared
subword vocabulary for zero-shot transfer from Tagalog to Cebuano — no NLLB
translation, no synthetic data. The Cebuano eval set is a 40-sentence
hand-curated set with documented biases (explicit profanity overlap).

See the source repo for more:
→ **https://github.com/kiergabelo/balikas**

## Dataset & license

- **Dataset:** [jcblaise/hatespeech_filipino](https://huggingface.co/datasets/jcblaise/hatespeech_filipino)
- **Paper:** Cabasag, Chan, Lim, Gonzales, Cheng. "Hate speech in Philippine
  election-related tweets." *Philippine Computing Journal* XIV(1), August 2019.
- **License:** Apache-2.0 (inherits from the dataset license). Code is MIT.
