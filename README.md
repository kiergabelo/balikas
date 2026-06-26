# Balikas — Filipino Hate Speech Detection

Binary hate/non-hate classification of Filipino social media text, trained on
a combined corpus of **43,892 samples**: the Cabasag et al. (2019) Tagalog
election tweets + SEACrowd Filipino TikTok hate speech (Taglish + Cebuano
code-switched transcriptions). Fine-tuned XLM-RoBERTa achieves **F1 = 0.917**
on the held-out Tagalog test set.

*Balikas* is Bisaya for *profanity* / *vulgar language*.

## Features

- **XLM-RoBERTa fine-tuned on combined Filipino corpus** — trained on 14,232
  Tagalog tweets + 29,660 Filipino TikTok transcriptions. **F1(hate) = 0.917**
  on the held-out Tagalog test set. Model hosted on HF Hub
  (`kiergabelo/balikas-xlm`).
- **TF-IDF + LogReg baseline** — word 1–2-gram features; F1 = 0.749. Kept as a
  floor comparison.
- **HuggingFace Space (Gradio demo)** — live demo at
  `huggingface.co/spaces/kiergabelo/balikas`. Loads the model from HF Hub.
- **FastAPI `/classify` endpoint** — `{text} -> {label, confidence}`.
- **Cebuano evaluation** — a 40-sentence Cebuano set is included for zero-shot
  evaluation. The eval set has documented limitations (constructed examples,
  not native social media). Native Cebuano data collection is noted as future work.
- **Colab-ready training scripts** — `train/finetune_xlm_combined.py` trains on
  the combined corpus (~25 min on T4). `train/finetune_xlm.py` for Tagalog-only.
- **Reproducible data pipeline** — `data/download.py` fetches the Tagalog corpus
  from HuggingFace, bypassing a dead loading script. The TikTok dataset is
  loaded from GitHub (SEACrowd repo).

## Results

| # | Model | Training data | Eval set | Accuracy | F1 (hate) | Macro-F1 |
|---|---|---|---|---|---|---|
| 1 | TF-IDF + LogReg | 14k Tagalog | 4,232 Tagalog tweets | 0.7647 | 0.7489 | 0.7637 |
| 2 | XLM-R (Tagalog only) | 14k Tagalog | 4,232 Tagalog tweets | 0.7854 | 0.7646 | 0.7838 |
| 3 | **XLM-R (combined)** | **44k Tagalog + TikTok** | **4,232 Tagalog tweets** | **0.9227** | **0.9169** | **0.9224** |
| 4 | XLM-R (combined) | 44k Tagalog + TikTok | 40 Cebuano sentences | 0.6750 | 0.7547 | 0.6366 |

**Row 3 is the headline result.** The combined training data (Tagalog tweets +
Filipino TikTok transcriptions including code-switched Taglish) pushes Tagalog
F1 from 0.76 to 0.92 — approaching the state-of-the-art for Filipino hate
speech detection.

Row 4 is a limited zero-shot Cebuano evaluation. The eval set is constructed
(not native social media), so the number is indicative, not definitive. Native
Cebuano data collection would be needed for a proper assessment.

## Stack

- **Python 3.12+**
- PyTorch + Transformers — XLM-RoBERTa fine-tuning
- scikit-learn — TF-IDF + LogReg baseline
- Gradio — HuggingFace Space UI
- FastAPI — `/classify` endpoint
- HuggingFace `datasets` — Tagalog corpus download

## Quick start

```bash
python -m venv .venv; .venv\Scripts\activate
pip install -r api/requirements.txt

python data/download.py              # fetch Tagalog corpus
python train/train_baseline.py --lang tl   # TF-IDF baseline (~1s CPU)

uvicorn api.main:app --reload         # localhost:8000/classify
```

## Fine-tune XLM-RoBERTa (the headline model)

Run on free Colab T4 GPU (~25 min):

```python
# Colab cell 1
!pip install -q transformers datasets scikit-learn accelerate
# Colab cell 2
!git clone https://github.com/kiergabelo/balikas.git
%cd balikas
!python train/finetune_xlm_combined.py
```

Downloads the Tagalog corpus from HuggingFace + TikTok data from GitHub
automatically. Evaluates on Tagalog test + Cebuano eval. Saves model to
`model/xlm-roberta-balikas/`.

## Deploy (HuggingFace Space)

The `space/` folder is a self-contained Gradio app. The model is loaded from
`kiergabelo/balikas-xlm` on the HF Hub at startup (avoids Space storage limits).

1. Create a Space at <https://huggingface.co/new-space> (Gradio, CPU free)
2. Push the `space/` contents to the Space repo
3. Push the model to `kiergabelo/balikas-xlm` on HF Hub
4. The Space auto-downloads the model on startup

## Project structure

```
balikas/
├── data/
│   ├── download.py                  # reproducible Tagalog corpus fetch
│   ├── ceb_eval_handcurated.json    # 40 Cebuano eval sentences
│   └── construct_ceb_eval.py
├── train/
│   ├── train_baseline.py            # TF-IDF + LogReg
│   ├── finetune_xlm.py              # XLM-R Tagalog-only
│   ├── finetune_xlm_combined.py     # XLM-R combined Tagalog + TikTok
│   ├── eval_zero_shot_ceb.py
│   └── translate_to_ceb.py          # abandoned NLLB pipeline (documented)
├── api/
│   ├── main.py                      # FastAPI
│   └── requirements.txt
├── space/                           # HuggingFace Space (Gradio)
│   ├── app.py                       # loads model from HF Hub
│   ├── requirements.txt
│   └── README.md                    # HF Space YAML
└── model/                           # gitignored — artifacts
```

## Dataset & attribution

- **Tagalog corpus:** [jcblaise/hatespeech_filipino](https://huggingface.co/datasets/jcblaise/hatespeech_filipino)
  — Cabasag et al. (2019), Apache-2.0
- **TikTok corpus:** [SEACrowd/filipino_hatespeech_tiktok](https://huggingface.co/datasets/SEACrowd/filipino_hatespeech_tiktok)
  — Hernandez et al. (2021), CC-BY-SA-4.0

## License

Code: MIT. Trained weights: Apache-2.0.
