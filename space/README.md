---
title: Balikas
emoji: "\U0001F6E1\uFE0F"
colorFrom: slate
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
short_description: Tagalog hate speech detection with Cebuano cross-lingual eval
---

# Balikas — Filipino Hate Speech Detection

Demo for the **Tagalog-trained TF-IDF + LogReg baseline** with an honest
cross-lingual evaluation on a small hand-curated Cebuano set.

## Headline numbers

| # | Setup | Eval | F1 (hate) | Macro-F1 |
|---|---|---|---|---|
| 1 | Tagalog in-domain | 4,232 native labeled tweets | **0.7489** | 0.7637 |
| 2 | Zero-shot TL→CEB | 40 hand-curated sentences | **0.8333** | 0.7917 |

Row 2 looks higher than row 1 but **is not** an apples-to-apples win — the
Cebuano eval set is biased toward explicit profanity that overlaps with
Tagalog. The honest read: zero-shot works on clear cases; subtle hate
sarcasm/dog-whistles almost certainly fail. A proper assessment requires a
large native Cebuano benchmark that doesn't yet exist. **That gap is the
research contribution of this project.**

## Why this project

Filipino (Tagalog + the Bisayan languages) is low-resource for content
moderation. Big-vendor classifiers cluster it with English. No labeled
Bisaya hate-speech corpus exists on HuggingFace at time of writing
(verified June 2026). Bantay ships an honest baseline + cross-lingual study
rather than another API-wrapping demo.

## Methodology + caveats

See the source repo for the methodology, the abandoned NLLB translation
pipeline (documented methodological honesty), and the Colab-ready RoBERTa
fine-tune script:

→ **https://github.com/kiergabelo/balikas**

## Dataset & license

- **Dataset:** [jcblaise/hatespeech_filipino](https://huggingface.co/datasets/jcblaise/hatespeech_filipino)
- **Original corpus paper:** Cabasag, Chan, Lim, Gonzales, Cheng. "Hate
  speech in Philippine election-related tweets." *Philippine Computing
  Journal* XIV(1), August 2019.
- **License:** Apache-2.0 (inherits from the dataset license). Code is MIT.
