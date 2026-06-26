"""
Fine-tune XLM-RoBERTa on a combined Tagalog + Filipino (incl. Cebuano) hate
speech corpus. Trains on:
  1. Cabasag et al. (2019) — 10k Tagalog election tweets (jcblaise/hatespeech_filipino)
  2. SEACrowd Filipino TikTok hate speech — transcribed videos in Taglish + Cebuano

Then evaluates zero-shot on the hand-curated Cebuano set.

Run on free Colab T4 GPU (~25 min):

    !pip install -q transformers datasets scikit-learn accelerate
    !python train/finetune_xlm_combined.py
"""
import os, json, sys, numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import Dataset, load_dataset, concatenate_datasets

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MODEL_NAME = "xlm-roberta-base"
OUT_DIR = ROOT / "model" / "xlm-roberta-balikas"
MAX_LEN = 128
SEED = 42

set_seed(SEED)

# ═══════════════════════════════════════════════════════════════════════════
# 1. Load Cabasag Tagalog corpus
# ═══════════════════════════════════════════════════════════════════════════
print("Loading Cabasag Tagalog corpus ...")
try:
    ds_tl = load_dataset("jcblaise/hatespeech_filipino", trust_remote_code=True)
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
    ds_tl = {
        "train": load_csv("train"),
        "valid": load_csv("valid"),
        "test": load_csv("test"),
    }
print(f"  Tagalog: train={len(ds_tl['train'])}, valid={len(ds_tl['valid'])}, test={len(ds_tl['test'])}")

# ═══════════════════════════════════════════════════════════════════════════
# 2. Load SEACrowd Filipino TikTok hate speech (Taglish + Cebuano)
# ═══════════════════════════════════════════════════════════════════════════
print("Loading SEACrowd Filipino TikTok hate speech ...")
try:
    ds_tiktok = load_dataset("SEACrowd/filipino_hatespeech_tiktok", trust_remote_code=True)
    tiktok_all = ds_tiktok["train"] if "train" in ds_tiktok else ds_tiktok[list(ds_tiktok.keys())[0]]
    # Normalize column names to {text, label}
    cols = tiktok_all.column_names
    text_col = "text" if "text" in cols else cols[0]
    label_col = "label" if "label" in cols else ("hate" if "hate" in cols else cols[-1])
    tiktok_all = tiktok_all.rename_column(text_col, "text").rename_column(label_col, "label")
    # Ensure labels are 0/1
    def normalize_label(ex):
        v = ex["label"]
        if isinstance(v, str):
            v = 1 if v.lower() in ("1", "true", "yes", "hate") else 0
        ex["label"] = int(v)
        return ex
    tiktok_all = tiktok_all.map(normalize_label)
    # Keep only text + label columns
    keep = {"text", "label"}
    tiktok_all = tiktok_all.remove_columns([c for c in tiktok_all.column_names if c not in keep])
    print(f"  TikTok: {len(tiktok_all)} samples (Taglish + Cebuano)")
except Exception as e:
    print(f"  TikTok load failed: {e}")
    print("  Continuing with Tagalog only")
    tiktok_all = None

# ═══════════════════════════════════════════════════════════════════════════
# 3. Combine datasets
# ═══════════════════════════════════════════════════════════════════════════
print("Combining datasets ...")
combined_train = [ds_tl["train"], ds_tl["valid"]]
if tiktok_all is not None:
    # Split TikTok 80/10/10
    tiktok_split = tiktok_all.train_test_split(test_size=0.2, seed=SEED)
    tiktok_train = tiktok_split["train"]
    tiktok_rest = tiktok_split["test"].train_test_split(test_size=0.5, seed=SEED)
    tiktok_valid = tiktok_rest["train"]
    tiktok_test = tiktok_rest["test"]
    combined_train.append(tiktok_train)
    # Merge TikTok test into the Tagalog test for combined eval
    # (but keep a Tagalog-only eval too for comparison)
    print(f"  TikTok split: train={len(tiktok_train)}, valid={len(tiktok_valid)}, test={len(tiktok_test)}")

train_ds = concatenate_datasets(combined_train)
print(f"  Combined train: {len(train_ds)}")

# Validation = Tagalog valid (+ TikTok valid if available)
valid_sets = [ds_tl["valid"]]
if tiktok_all is not None:
    valid_sets.append(tiktok_valid)
valid_ds = concatenate_datasets(valid_sets)

# Tagalog-only test (for comparison with baseline)
test_tl = ds_tl["test"]

# ═══════════════════════════════════════════════════════════════════════════
# 4. Tokenize
# ═══════════════════════════════════════════════════════════════════════════
print(f"Loading tokenizer: {MODEL_NAME}")
tok = AutoTokenizer.from_pretrained(MODEL_NAME)

def encode(batch):
    return tok(batch["text"], truncation=True, max_length=MAX_LEN)

train_ds = train_ds.map(encode, batched=True)
valid_ds = valid_ds.map(encode, batched=True)
test_tl = test_tl.map(encode, batched=True)

# ═══════════════════════════════════════════════════════════════════════════
# 5. Train
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
    data_collator=DataCollatorWithPadding(tok),
    compute_metrics=compute_metrics,
)

print("\nTraining ... (~20-25 min on T4)")
trainer.train()

# ═══════════════════════════════════════════════════════════════════════════
# 6. Evaluate on Tagalog test
# ═══════════════════════════════════════════════════════════════════════════
print("\n=== Tagalog test set ===")
tl_results = trainer.evaluate(test_tl)
print(json.dumps(tl_results, indent=2))

# ═══════════════════════════════════════════════════════════════════════════
# 7. Evaluate on Cebuano hand-curated set
# ═══════════════════════════════════════════════════════════════════════════
ceb_path = ROOT / "data" / "ceb_eval_handcurated.json"
print(f"\n=== Cebuano eval ({ceb_path}) ===")
with open(ceb_path, encoding="utf-8") as f:
    ceb_rows = json.load(f)
ceb_texts = [r["text"] for r in ceb_rows]
ceb_labels = [r["label"] for r in ceb_rows]
ceb_ds = Dataset.from_dict({"text": ceb_texts, "label": ceb_labels}).map(encode, batched=True)

ceb_results = trainer.evaluate(ceb_ds)
print(json.dumps({k: v for k, v in ceb_results.items() if "_loss" not in k}, indent=2))

# ═══════════════════════════════════════════════════════════════════════════
# 8. Save
# ═══════════════════════════════════════════════════════════════════════════
trainer.save_model(str(OUT_DIR))
tok.save_pretrained(str(OUT_DIR))
print(f"\nSaved to: {OUT_DIR}")

metrics = {
    "training_data": {
        "tagalog_cabasag": len(ds_tl["train"]),
        "tiktok_seacrowd": len(tiktok_train) if tiktok_all else 0,
        "total_train": len(train_ds),
    },
    "tagalog_test": tl_results,
    "cebuano_eval": {k: v for k, v in ceb_results.items() if "_loss" not in k},
}
with open(ROOT / "model" / "xlm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\n--- DONE ---")
print(f"Tagalog F1: {tl_results.get('eval_f1_hate', '?'):.4f}")
print(f"Cebuano F1: {ceb_results.get('eval_f1_hate', '?'):.4f}")
print(f"\nNext: push model/xlm-roberta-balikas/ to HF Hub (kiergabelo/balikas-xlm)")
