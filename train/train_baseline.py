"""Train TF-IDF + LogReg baseline on either the Tagalog source or Cebuano
translated splits. Defaults to Cebuano if available.

    python train/train_baseline.py            # ceb if present else tl
    python train/train_baseline.py --lang tl  # force Tagalog
    python train/train_baseline.py --lang ceb # force Cebuano
"""
import argparse, json, os, time, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["tl", "ceb"], default=None,
                    help="dataset to train on; default: ceb if present else tl")
    args = ap.parse_args()

    ceb_path = os.path.join(DATA, "splits_ceb.json")
    tl_path = os.path.join(DATA, "splits.json")
    if args.lang is None:
        lang = "ceb" if os.path.exists(ceb_path) else "tl"
    else:
        lang = args.lang
    path = ceb_path if lang == "ceb" else tl_path
    if not os.path.exists(path):
        raise SystemExit(f"missing: {path} (run data/download.py and train/translate_to_ceb.py first)")

    out_dir = os.path.join(ROOT, "model")
    os.makedirs(out_dir, exist_ok=True)
    model_out = os.path.join(out_dir, f"baseline_{lang}.joblib")
    metrics_out = os.path.join(out_dir, f"baseline_{lang}_metrics.json")

    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    tr, va, te = d["train.csv"], d["valid.csv"], d["test.csv"]
    X_tr = [r["text"] for r in tr + va]; y_tr = [int(r["label"]) for r in tr + va]
    X_te = [r["text"] for r in te];      y_te = [int(r["label"]) for r in te]

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                  max_df=0.95, sublinear_tf=True, strip_accents="unicode")),
        ("clf", LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")),
    ])
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    dt = time.time() - t0

    acc = accuracy_score(y_te, pred)
    p, r, f1, _ = precision_recall_fscore_support(y_te, pred, average="binary", pos_label=1, zero_division=0)
    macro = precision_recall_fscore_support(y_te, pred, average="macro", zero_division=0)[2]
    cm = confusion_matrix(y_te, pred, labels=[0, 1]).tolist()
    joblib.dump(pipe, model_out)
    metrics = dict(lang=lang, acc=acc, f1_hate=f1, precision_hate=p, recall_hate=r,
                   macro_f1=macro, cm=cm, train_seconds=dt,
                   train_size=len(X_tr), test_size=len(X_te))
    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{lang}] trained in {dt:.1f}s on {len(X_tr)} rows")
    print(f"TEST: acc={acc:.4f}  F1(hate)={f1:.4f}  macro-F1={macro:.4f}")
    print(f"precision(hate)={p:.4f}  recall(hate)={r:.4f}")
    print(f"confusion (rows=true 0/1): {cm}")
    print(f"saved model -> {model_out}")

if __name__ == "__main__":
    main()
