---
title: Balikas
emoji: "\U0001F6E1\uFE0F"
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version: "5.25.0"
app_file: app.py
pinned: false
tags:
  - hate-speech-detection
  - filipino
  - tagalog
  - cebuano
  - xlm-roberta
  - nlp
language:
  - tl
  - ceb
license: apache-2.0
short_description: Tagalog hate speech classifier with Cebuano zero-shot eval
---

# Balikas — Filipino Hate Speech Detection

**XLM-RoBERTa fine-tuned on 43,892 Filipino social media samples** — Tagalog
election tweets + Filipino TikTok transcriptions including code-switched
Taglish and Cebuano.

## Results

| Model | Training data | F1 (hate) |
|---|---|---|
| TF-IDF + LogReg | 14k Tagalog | 0.749 |
| XLM-R (Tagalog only) | 14k Tagalog | 0.765 |
| **XLM-R (combined)** | **44k Tagalog + TikTok** | **0.917** |

The combined training data pushes Tagalog F1 from 0.76 to 0.92. The TikTok
corpus adds code-switched Filipino text (Taglish + Cebuano) that improves
generalization significantly.

## Dataset

- [jcblaise/hatespeech_filipino](https://huggingface.co/datasets/jcblaise/hatespeech_filipino) — Cabasag et al. 2019
- [SEACrowd/filipino_hatespeech_tiktok](https://huggingface.co/datasets/SEACrowd/filipino_hatespeech_tiktok) — Hernandez et al. 2021

## Code

→ **https://github.com/kiergabelo/balikas**
