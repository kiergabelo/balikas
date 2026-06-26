"""
Fine-tune XLM-RoBERTa-base on Tagalog hate speech, then evaluate zero-shot
on the hand-curated Cebuano set. Run on free Colab with a T4 GPU (~25 min).

Colab setup (run in a new cell BEFORE this script):
    !pip install -q transformers datasets scikit-learn torch accelerate

Then run this script:
    !python train/finetune_xlm.py

The script downloads the Tagalog corpus from HuggingFace automatically, so
you don't need to upload any data files. The Cebuano eval set is built-in
(hand-curated, shipped in data/ceb_eval_handcurated.json).

When done, upload the `model/xlm-roberta-balikas/` folder to the HuggingFace
Space repo so the demo uses the fine-tuned model instead of the old TF-IDF
baseline.
"""
import os, json, sys, numpy as np
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import Dataset, load_dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_NAME = "xlm-roberta-base"
OUT_DIR = ROOT / "model" / "xlm-roberta-balikas"
MAX_LEN = 128
SEED = 42

set_seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load Tagalog corpus from HuggingFace (automatic download)
# ═══════════════════════════════════════════════════════════════════════════
print("Loading Tagalog hate speech corpus ...")
try:
    ds = load_dataset("jcblaise/hatespeech_filipino", trust_remote_code=True)
except Exception:
    # fallback: direct CSV download
    import csv, io, urllib.request, zipfile, tempfile
    URL = "https://huggingface.co/datasets/jcblaise/hatespeech_filipino/resolve/main/hatespeech_raw.zip"
    tmp = tempfile.mkdtemp()
    zf = os.path.join(tmp, "data.zip")
    urllib.request.urlretrieve(URL, zf)
    with zipfile.ZipFile(zf) as z:
        z.extractall(tmp)
    def load_csv(name):
        rows = []
        path = os.path.join(tmp, "hatespeech", f"{name}.csv")
        with open(path, encoding="utf-8") as f:
            rdr = csv.reader(f, quotechar='"', delimiter=",",
                             quoting=csv.QUOTE_ALL, skipinitialspace=True)
            next(rdr, None)
            for row in rdr:
                if len(row) == 2:
                    rows.append({"text": row[0], "label": int(row[1])})
        return Dataset.from_list(rows)
    ds = {
        "train": load_csv("train"),
        "valid": load_csv("valid"),
        "test": load_csv("test"),
    }

print(f"  train: {len(ds['train'])}, valid: {len(ds['valid'])}, test: {len(ds['test'])}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Tokenize
# ═══════════════════════════════════════════════════════════════════════════
print(f"Loading tokenizer: {MODEL_NAME}")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(batch):
    return tok(batch["text"], truncation=True, max_length=MAX_LEN)

train_ds = ds["train"].map(encode, batched=True)
valid_ds = ds["valid"].map(encode, batched=True)
test_ds  = ds["test"].map(encode, batched=True)

# ═══════════════════════════════════════════════════════════════════════════
# 3. Load model + train
# ═══════════════════════════════════════════════════════════════════════════
print(f"Loading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", pos_label=1, zero_division=0)
    macro = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)[2]
    return {"accuracy": acc, "f1_hate": f1, "macro_f1": macro}

args = TrainingArguments(
    output_dir=str(OUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_hate",
    greater_is_better=True,
    seed=SEED,
    report_to="none",
)

trainer = Trainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=valid_ds,
    tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)

print("\nTraining ... (~20-25 min on T4)")
trainer.train()

# ═══════════════════════════════════════════════════════════════════════════
# 4. Evaluate on Tagalog test set
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Tagalog test set ===")
tl_results = trainer.evaluate(test_ds)
print(json.dumps(tl_results, indent=2))

# ═══════════════════════════════════════════════════════════════════════════
# 5. Evaluate zero-shot on Cebuano hand-curated set
# ═══════════════════════════════════════════════════════════════════════════
ceb_path = ROOT / "data" / "ceb_eval_handcurated.json"
print(f"\n=== Cebuano zero-shot eval ({ceb_path}) ===")
with open(ceb_path, encoding="utf-8") as f:
    ceb_rows = json.load(f)
ceb_texts = [r["text"] for r in ceb_rows]
ceb_labels = [r["label"] for r in ceb_rows]

ceb_ds = Dataset.from_dict({"text": ceb_texts, "label": ceb_labels})
ceb_ds = ceb_ds.map(encode, batched=True)

ceb_results = trainer.evaluate(ceb_ds)
print(json.dumps({k: v for k, v in ceb_results.items() if "_loss" not in k}, indent=2))

# ═══════════════════════════════════════════════════════════════════════════
# 6. Save
# ═══════════════════════════════════════════════════════════════════════════
trainer.save_model(str(OUT_DIR))
tok.save_pretrained(str(OUT_DIR))
print(f"\nSaved model to: {OUT_DIR}")

metrics = {
    "tagalog_test": tl_results,
    "cebuano_zero_shot": {k: v for k, v in ceb_results.items() if "_loss" not in k},
    "transfer_gap": tl_results.get("eval_f1_hate", 0) - ceb_results.get("eval_f1_hate", 0),
}
with open(ROOT / "model" / "xlm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to model/xlm_metrics.json")
print(f"\n--- DONE ---")
print(f"Tagalog F1: {tl_results.get('eval_f1_hate', '?'):.4f}")
print(f"Cebuano F1: {ceb_results.get('eval_f1_hate', '?'):.4f}")
print(f"Transfer gap: {metrics['transfer_gap']:.4f}")
print(f"\nNext: download model/xlm-roberta-balikas/ and push to HF Space")
