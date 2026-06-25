# Balikas — Filipino Hate Speech Detection (Tagalog + Cebuano cross-lingual assessment)

Binary hate/non-hate classification of Filipino tweets, trained on the
[Cabasag et al. (2019)](https://huggingface.co/datasets/jcblaise/hatespeech_filipino)
Tagalog corpus (~10k tweets from the 2016 Philippine Presidential Elections),
with an additional zero-shot cross-lingual evaluation on a small hand-curated
Cebuano set.

Balikas is Bisaya for *profanity* / *vulgar language* — the project is
Cebuano-identified even though its training data is Tagalog, because no
labeled Bisaya hate-speech corpus exists at the time of writing. That gap,
and the cross-lingual transfer study around it, is the core of this project —
not "I called an AI API."

## Why this exists

Filipino is a low-resource language family for content moderation. Big-vendor
classifiers cluster Tagalog/Cebuano with English and miss the code-switched
Taglish/Bislish profanity that characterizes PH political discourse. No
labeled Bisaya hate-speech benchmark exists on HuggingFace (verified June 2026).
This project ships:

1. A publishable Tagalog baseline with measured F1-vs-baseline numbers.
2. An honest cross-lingual transfer study: how well does a Tagalog-trained
   model generalize to **Cebuano**, a related-but-distinct Philippine language,
   given that no Bisaya labels exist?
3. A documented gap: machine-translating labels via NLLB-200 is impractical
   at the volume needed (a 600M NLLB on CPU translates ~14k tweets in many
   hours; see `train/translate_to_ceb.py` for the abandoned attempt). The
   right fix is a multilingual transformer (XLM-RoBERTa) — flagged as
   future work.

## Results

| # | Setup | Eval set | Accuracy | F1 (hate) | Macro-F1 | Notes |
|---|---|---|---|---|---|---|
| 1 | Tagalog in-domain | 4,232 tweets (native labeled, held-out) | 0.7647 | **0.7489** | 0.7637 | TF-IDF (word 1-2gram) + LogReg, 0.7s CPU |
| 2 | Zero-shot TL→CEB | 40 hand-curated Bisaya sentences | 0.8000 | **0.8333** | 0.7917 | same model, no Cebuano training |

### The cross-lingual transfer number is honest about its limits

Result 2 looks better than result 1, but it is **not** a real apples-to-apples
win. The Cebuano eval set is hand-curated and biased toward explicit,
unambiguous profanity ("yawa", "ulol", "pisti", "putang ina") that overlaps
heavily with Tagalog profanity. The model catches all 20 hates (recall = 1.0)
but false-positives 8 of 20 non-hates.

The honest read: zero-shot transfer via shared profanity *works on clear
cases*. Subtle hate — sarcasm, dog-whistles, code-switched slights — almost
certainly fails, and a proper assessment requires a large native-curated
Cebuano benchmark that does not yet exist. **That gap is the research
contribution of this project.**

### Why I abandoned the NLLB translation-transfer approach

`train/translate_to_ceb.py` is left in the repo as a documented attempt.
The script attempts to machine-translate the 14k Tagalog tweets to Cebuano
via `facebook/nllb-200-distilled-600M` and resume-checkpoints after each
split. In practice, on CPU, NLLB-600M translates ~32 tweets per 10 seconds
— fine for a few hundred tweets, but ~50+ minutes wall-clock per split.

Three reasons I dropped it:

1. Translation noise compounds: NLLB makes systematic phonological mistakes
   ("ulol" → "ulol", but slang like "tanga", "buang" is sometimes calqued
   into Tagalog, not Bisaya).
2. A model trained on translated labels isn't a Cebuano model — it's a
   "Tagalog-trained-on-Cebuano-noise" model. The eval against translated
   test data wouldn't tell us anything about real Cebuano.
3. Realistic path forward is multilingual transfer via shared subwords
   (XLM-RoBERTa), not translation-as-augmentation. Leave the latter for
   future work.

## Quick start

**Live demo (HuggingFace Space):** _deploy the `space/` folder to an HF Space
repo (see "Deploy the Space" below) → `https://huggingface.co/spaces/<your-user>/balikas`._

```bash
python -m venv .venv; .venv\Scripts\activate
pip install -r api/requirements.txt

python data/download.py                     # fetch + parse Tagalog corpus from HuggingFace
python train/train_baseline.py --lang tl    # trains in ~1s on CPU -> model/baseline_tl.joblib
python data/construct_ceb_eval.py           # writes the small Bisaya eval set
python train/eval_zero_shot_ceb.py          # evaluates the TL model on Cebuano
                                            # -> model/zeroshot_ceb_metrics.json

uvicorn api.main:app --reload               # localhost:8000/classify
```

## Deploy the Space (the public "website")

The `space/` folder is a self-contained HuggingFace Spaces app (Gradio +
the trained model baked in, ~880 KB). To deploy:

1. Create a free HuggingFace account if you don't have one.
2. Create a new Space at <https://huggingface.co/new-space>:
   - **SDK:** Gradio
   - **License:** Apache-2.0
   - **Hardware:** CPU basic (free)
3. Push the contents of `space/` (not the folder itself — the files at root):
   ```bash
   git clone https://huggingface.co/spaces/<your-user>/balikas
   Copy-Item -Path "space\*" -Destination balikas-space\ -Recurse -Force
   cd balikas-space
   git add -A && git commit -m "Initial Space: Balikas demo" && git push
   ```
4. Wait ~1-2 min for the Space to build. Public URL will be
   `https://huggingface.co/spaces/<your-user>/balikas`.
5. Drop that URL into the portfolio entry's `demo` field.

Classify a tweet:

```bash
curl -X POST http://127.0.0.1:8000/classify ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"TANG INA MO talaga eh bobo\", \"lang\":\"tl\"}"
# -> {"label":"hate","label_id":1,"confidence":0.998,"lang":"tl"}

curl -X POST http://127.0.0.1:8000/classify ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Buang kaayo ka, ulol!\", \"lang\":\"tl\"}"
# -> {"label":"hate","label_id":1,"confidence":0.7xxx,"lang":"tl"}
# (note: model is Tagalog-trained; Cebuano is loaded as best-effort zero-shot)
```

## Fine-tune RoBERTa-Tagalog (the headline model)

The classical TF-IDF + LogReg baseline is the floor. The ceiling is a
fine-tuned `jcblaise/roberta-tagalog-base`. Run on free Colab (T4 GPU):

```python
# Colab cell 1
!pip install -q transformers datasets scikit-learn accelerate
# Colab cell 2
!python train/finetune_roberta.py
```

Expected: ~0.82–0.86 F1(hate) on the Tagalog test set. The fine-tuned
weights + the eval numbers go in `model/` and update this README's table.

## Future work

- Native Cebuano benchmark. Build a large, native-annotated Bisaya hate-speech
  dataset via community annotation. This is the actual bottleneck — not
  modeling.
- XLM-RoBERTa cross-lingual transfer. Fine-tune `xlm-roberta-base` on the
  Tagalog corpus and evaluate *zero-shot* on a proper Cebuano benchmark. The
  shared subword vocabulary should reduce the transfer gap visible in row
  2 of the results table.
- Larger Cebuano eval set. The current 40-sample set is a demonstration,
  not a benchmark.

## Project structure

```
balikas/
├── data/
│   ├── download.py                       # reproducible fetch from HF Hub
│   ├── splits.json                      # gitignored — Tagalog train/valid/test
│   ├── ceb_eval_handcurated.json        # 40 hand-curated Bisaya sentences
│   └── construct_ceb_eval.py            # builder for the above
├── train/
│   ├── train_baseline.py                # TF-IDF + LogReg, ~1s on CPU, --lang tl|ceb
│   ├── eval_zero_shot_ceb.py             # evaluate TL model on Cebuano set
│   ├── finetune_roberta.py              # Colab: roberta-tagalog-base fine-tune
│   └── translate_to_ceb.py              # ABANDONED: NLLB translation pipeline (see README)
├── api/
│   ├── main.py                           # FastAPI /classify {text, lang} -> {label, confidence, lang}
│   └── requirements.txt
├── model/                                # gitignored — artifacts
│   ├── baseline_tl.joblib
│   ├── baseline_tl_metrics.json
│   └── zeroshot_ceb_metrics.json
└── README.md
```

## Dataset & attribution

Tagalog source corpus: [jcblaise/hatespeech_filipino](https://huggingface.co/datasets/jcblaise/hatespeech_filipino)
under Apache-2.0. Original data from Cabasag, Chan, Lim, Gonzales, and Cheng,
"Hate speech in Philippine election-related tweets: Automatic detection and
classification using natural language processing," *Philippine Computing
Journal* XIV(1), August 2019.

Cebuano evaluation set in `data/ceb_eval_handcurated.json` is original to
this project (hand-curated, BIASED toward explicit profanity — see README
caveats). Released under MIT for the code, Apache-2.0 for the trained weights
(inherits from the dataset license).

## License

Code: MIT. Trained weights: Apache-2.0. Cebuano eval set: MIT.
