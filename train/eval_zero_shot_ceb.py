"""Zero-shot cross-lingual transfer evaluation.

Loads the Tagalog-trained TF-IDF + LogReg baseline (baseline_tl.joblib) and
evaluates it on a small hand-curated Cebuano set. Expected: very low F1,
because TF-IDF features don't cross languages. The drop IS the finding and
motivates switching to a multilingual transformer (XLM-RoBERTa — future work).
"""
import json, os, time, joblib
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL = os.path.join(ROOT, "model", "baseline_tl.joblib")
EVAL = os.path.join(ROOT, "data", "ceb_eval_handcurated.json")
OUT = os.path.join(ROOT, "model", "zeroshot_ceb_metrics.json")

print(f"loading model: {MODEL}")
pipe = joblib.load(MODEL)

with open(EVAL, encoding="utf-8") as f:
    rows = json.load(f)
X = [r["text"] for r in rows]
y_true = [int(r["label"]) for r in rows]
print(f"eval set: {len(X)} rows; positives={sum(y_true)}")

t0 = time.time()
proba = pipe.predict_proba(X)
y_pred = [int(p.argmax()) for p in proba]
dt = time.time() - t0

acc = accuracy_score(y_true, y_pred)
p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                              pos_label=1, zero_division=0)
macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
print(f"\n=== Zero-shot Tagalog model on Cebuano eval (n={len(X)}) ===")
print(f"accuracy={acc:.4f}  F1(hate)={f1:.4f}  macro-F1={macro:.4f}")
print(f"precision(hate)={p:.4f}  recall(hate)={r:.4f}")
print(f"confusion (rows=true 0/1): {cm}")
print("\nclassification report:\n" + classification_report(y_true, y_pred,
    target_names=["non-hate", "hate"], zero_division=0))

# Tagalog reference number
TAG_METRICS = os.path.join(ROOT, "model", "baseline_tl_metrics.json")
tag_f1 = None
if os.path.exists(TAG_METRICS):
    with open(TAG_METRICS) as f:
        tag_f1 = json.load(f).get("f1_hate")
if tag_f1 is not None:
    drop = tag_f1 - f1
    print(f"\nTagalog in-domain F1(hate) = {tag_f1:.4f}")
    print(f"Zero-shot Cebuano F1(hate) = {f1:.4f}")
    print(f"Transfer gap = {drop:+.4f}  (negative = drop)")

metrics = dict(setup="zero-shot Tagalog model -> Cebuano hand-curated eval",
               acc=acc, f1_hate=f1, precision_hate=p, recall_hate=r,
               macro_f1=macro, cm=cm, eval_seconds=dt,
               tagalog_in_domain_f1=tag_f1, transfer_gap=(tag_f1 - f1) if tag_f1 else None,
               n=len(X))
with open(OUT, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nsaved -> {OUT}")
