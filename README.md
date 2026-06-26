# Balikas — Filipino Hate Speech Detection

Binary hate/non-hate classification of Filipino tweets, trained on the
[Cabasag et al. (2019)](https://huggingface.co/datasets/jcblaise/hatespeech_filipino)
Tagalog corpus (~10k tweets from the 2016 Philippine Presidential Elections),
with a zero-shot cross-lingual evaluation on a small hand-curated Cebuano set.

*Balikas* is Bisaya for *profanity* / *vulgar language* — the project is
Cebuano-identified even though its training data is Tagalog, because **no
labeled native Cebuano (Bisaya) hate-speech corpus exists** at the time of
writing (verified on HuggingFace, June 2026). That gap — and the cross-lingual
transfer study built around it — is the core of this project, not an API wrapper.

## Features

- **Tagalog TF-IDF + LogReg baseline** — word 1–2-gram features + Logistic
  Regression; trains in ~0.7s on CPU. In-domain **F1(hate) = 0.749**.
- **Zero-shot cross-lingual evaluation** — the Tagalog model is evaluated on a
  40-sentence hand-curated Cebuano set with *no* Cebuano training.
  **F1 = 0.833**, with the caveats documented below (biased toward explicit
  profanity that overlaps with Tagalog).
- **FastAPI `/classify` endpoint** — `{text, lang} -> {label, confidence, lang}`.
- **HuggingFace Space (Gradio demo)** — the `space/` folder is a self-contained
  Space with the trained model baked in (~880 KB); deploys to HF CPU-free tier.
- **Abandoned NLLB translation pipeline, documented** — `train/translate_to_ceb.py`
  stays in the repo as a documented failed attempt (NLLB-200 dist 600M on CPU
  is impractical at ~14k tweets). Not hidden.
- **Colab-ready XLM-RoBERTa fine-tune script** — `train/finetune_xlm.py` fine-tunes
  `xlm-roberta-base` (cross-lingual, shared subword vocabulary) on a free T4 GPU
  (~25 min). Targets ~0.80-0.85 F1 on Tagalog, evaluates zero-shot Cebuano.
  Replaces the old Tagalog-only `roberta-tagalog` approach (see
  `train/finetune_roberta.py`, deprecated).
- **Reproducible data download** — `data/download.py` fetches and splits the
  corpus directly from HuggingFace, bypassing the repo's dead loading script.
- **Hand-curated Cebuano eval set** — `data/ceb_eval_handcurated.json`, the
  contribution toward filling the missing-Bisaya-corpus gap.

## Results

| # | Setup | Eval set | Accuracy | F1 (hate) | Macro-F1 | Notes |
|---|---|---|---|---|---|---|
| 1 | Tagalog in-domain | 4,232 native labeled tweets (held-out) | 0.7647 | **0.7489** | 0.7637 | TF-IDF (word 1-2gram) + LogReg, ~0.7s CPU |
| 2 | Zero-shot TL→CEB | 40 hand-curated Bisaya sentences | 0.8000 | **0.8333** | 0.7917 | same TF-IDF model, no Cebuano training |
| 3 | XLM-RoBERTa on TL | 4,232 native labeled tweets (held-out) | _TBD_ | _TBD_ | _TBD_ | run `train/finetune_xlm.py` on Colab T4 (~25 min) |
| 4 | XLM-RoBERTa → CEB | 40 hand-curated Bisaya sentences | _TBD_ | _TBD_ | _TBD_ | zero-shot (no Cebuano training — shared subword vocab only) |

### Caveat on the cross-lingual number

Row 2 looks higher than row 1, but it is **not** a real apples-to-apples
win. The Cebuano eval set is hand-curated and biased toward explicit,
unambiguous profanity ("yawa", "ulol", "pisti", "putang ina") that overlaps
heavily with Tagalog profanity. The model catches all 20 hates (recall = 1.0)
but false-positives 8 of 20 non-hates. The honest read: zero-shot transfer via
shared profanity *works on clear cases*; subtle hate — sarcasm, dog-whistles,
code-switched slights — almost certainly fails. A proper assessment requires a
large native-curated Cebuano benchmark that does not yet exist. **That gap is
the research contribution of this project.**

## Stack

- **Python 3.14**
- scikit-learn — TF-IDF + LogReg
- pandas — data wrangling
- FastAPI + uvicorn — `/classify` endpoint
- Gradio — HuggingFace Space UI
- joblib — model serialization
- HuggingFace `datasets` library — source corpus download

## Quick start

```bash
python -m venv .venv; .venv\Scripts\activate
pip install -r api/requirements.txt

python data/download.py              # fetch + parse Tagalog corpus from HuggingFace
python train/train_baseline.py --lang tl   # trains in ~1s on CPU -> model/baseline_tl.joblib
python data/construct_ceb_eval.py     # writes the small Bisaya eval set
python train/eval_zero_shot_ceb.py    # evaluates the TL model on Cebuano
                                     # -> model/zeroshot_ceb_metrics.json

uvicorn api.main:app --reload         # localhost:8000/classify
```

Classify a tweet:

```bash
curl -X POST http://127.0.0.1:8000/classify ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"TANG INA MO talaga eh bobo\", \"lang\":\"tl\"}"
# -> {"label":"hate","label_id":1,"confidence":0.998,"lang":"tl"}

curl -X POST http://127.0.0.1:8000/classify ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Buang kaayo ka, ulol!\", \"lang\":\"tl\"}"
# -> {"label":"hate",...} (note: model is Tagalog-trained; Cebuano is loaded as best-effort zero-shot)
```

## Deploy (HuggingFace Space)

The `space/` folder is a self-contained HuggingFace Spaces app (Gradio +
trained model baked in, ~880 KB). To deploy:

1. Create a free HuggingFace account if you don't have one.
2. Create a new Space at <https://huggingface.co/new-space>:
   - **SDK:** Gradio
   - **License:** Apache-2.0
   - **Hardware:** CPU basic (free)
3. Push the **contents of `space/`** to the HF repo (files at the repo root, not
   nested under a `space/` folder). `space/README.md` already carries the
   required YAML frontmatter (`sdk: gradio`, `app_file: app.py`, tags, etc.):
   ```bash
   git clone https://huggingface.co/spaces/<your-user>/balikas
   Copy-Item -Path "space\*" -Destination balikas-space\ -Recurse -Force
   cd balikas-space
   git add -A; git commit -m "Initial Space: Balikas demo"; git push
   ```
4. Wait ~1–2 min for the Space to build. Public URL:
   `https://huggingface.co/spaces/<your-user>/balikas`.
5. Drop that URL into the portfolio entry's `demo` field.

**Optional custom domain:** `balikas.kierabelo.com` — set a Cloudflare CNAME
to the HF Space URL. Note that custom domains on HuggingFace Spaces require an
HF Pro subscription; if you're on the free tier, the raw `huggingface.co/spaces/...`
URL is the public demo.

## Fine-tune XLM-RoBERTa (the headline model)

The TF-IDF + LogReg baseline is the floor. The headline model is
`xlm-roberta-base` — a cross-lingual transformer whose shared subword
vocabulary spans both Tagalog and Cebuano, allowing zero-shot transfer.

**Run on free Colab T4 GPU (~25 min):**

```python
# Colab cell 1
!pip install -q transformers datasets scikit-learn accelerate
# Colab cell 2
!python train/finetune_xlm.py
```

**Expected:** ~0.80–0.85 F1(hate) on the Tagalog test set, and a Cebuano
zero-shot F1 in the `model/xlm_metrics.json` output that represents the
real cross-lingual transfer gap (no Cebuano training data was used).

**After training:** download the `model/xlm-roberta-balikas/` folder from
Colab, place it in the `space/` folder, and push to your HF Space. The
demo will switch from the old TF-IDF baseline to the fine-tuned model
automatically (`app.py` detects the folder and uses it).

## Project structure

```
balikas/
├── data/
│   ├── download.py                  # reproducible fetch from HF Hub (bypasses dead loading script)
│   ├── splits.json                  # gitignored — Tagalog train/valid/test
│   ├── ceb_eval_handcurated.json    # 40 hand-curated Bisaya sentences (shipped artifact)
│   └── construct_ceb_eval.py        # builder for the above
├── train/
│   ├── train_baseline.py            # TF-IDF + LogReg, ~1s on CPU, --lang tl|ceb
│   ├── eval_zero_shot_ceb.py        # evaluate TL model on Cebuano set
│   ├── finetune_roberta.py         # Colab: roberta-tagalog-base fine-tune
│   └── translate_to_ceb.py         # ABANDONED: NLLB translation pipeline (see README)
├── api/
│   ├── main.py                      # FastAPI /classify {text, lang} -> {label, confidence, lang}
│   └── requirements.txt
├── space/                          # HuggingFace Space (Gradio), self-contained, ships own model copy
│   ├── app.py
│   ├── baseline_tl.joblib
│   ├── requirements.txt
│   └── README.md                    # YAML frontmatter: sdk: gradio, etc.
├── model/                          # gitignored — artifacts (metrics JSON shipped)
│   ├── baseline_tl.joblib
│   ├── baseline_tl_metrics.json
│   └── zeroshot_ceb_metrics.json
└── README.md
```

## Dataset & attribution

Tagalog source corpus:
[jcblaise/hatespeech_filipino](https://huggingface.co/datasets/jcblaise/hatespeech_filipino)
under Apache-2.0. Original data from Cabasag, Chan, Lim, Gonzales, and Cheng,
"Hate speech in Philippine election-related tweets: Automatic detection and
classification using natural language processing," *Philippine Computing
Journal* XIV(1), August 2019.

Cebuano evaluation set (`data/ceb_eval_handcurated.json`) is original to this
project — hand-curated and biased toward explicit profanity (see the caveat
above). Released under MIT.

## License

- **Code:** MIT
- **Trained weights:** Apache-2.0 (inherits the dataset license)
- **Cebuano eval set:** MIT
