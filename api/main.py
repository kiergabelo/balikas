"""Bantay API. Serves Cebuano (default) + Tagalog baselines if present."""
import os, joblib
from fastapi import FastAPI
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_DIR = os.path.join(ROOT, "model")
LABELS = {0: "non-hate", 1: "hate"}

models = {}
for lang in ("ceb", "tl"):
    p = os.path.join(MODEL_DIR, f"baseline_{lang}.joblib")
    if os.path.exists(p):
        models[lang] = joblib.load(p)
if not models:
    raise RuntimeError(f"no models in {MODEL_DIR}; run train/train_baseline.py")

app = FastAPI(title="Bantay — Filipino Hate Speech (Tagalog + Cebuano)",
              version="0.2.0")

class ClassifyIn(BaseModel):
    text: str
    lang: str | None = None     # "ceb" | "tl"; default = ceb if present else tl

class ClassifyOut(BaseModel):
    label: str
    label_id: int
    confidence: float
    lang: str

def pick_lang(req_lang):
    if req_lang and req_lang in models:
        return req_lang
    return "ceb" if "ceb" in models else next(iter(models))

@app.get("/health")
def health():
    return {"status": "ok", "models": list(models.keys())}

@app.post("/classify", response_model=ClassifyOut)
def classify(body: ClassifyIn):
    lang = pick_lang(body.lang)
    pipe = models[lang]
    proba = pipe.predict_proba([body.text])[0]
    label_id = int(proba.argmax())
    return ClassifyOut(
        label=LABELS[label_id],
        label_id=label_id,
        confidence=round(float(proba[label_id]), 4),
        lang=lang,
    )
