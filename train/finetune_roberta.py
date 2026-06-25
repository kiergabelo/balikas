"""Fine-tune RoBERTa-Tagalog on hatespeech_filipino. Run on Colab (T4 GPU).

Goal: beat the TF-IDF + LogReg baseline (F1-hate = 0.749, macro-F1 = 0.764).
Expected: ~0.82-0.86 F1-hate with roberta-tagalog-base.

Colab setup (first cell):
    !pip install -q transformers datasets scikit-learn accelerate

Then:
    !python train/finetune_roberta.py
"""
import os, json, numpy as np
import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding,
                          set_seed)
from datasets import Dataset as HFDataset
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
with open(os.path.join(ROOT, "data", "splits.json"), encoding="utf-8") as f:
    d = json.load(f)

MODEL_NAME = "jcblaise/roberta-tagalog-base"   # pretrained Tagalog RoBERTa
MAX_LEN = 128
SEED = 42
OUT_DIR = os.path.join(ROOT, "model", "roberta-tagalog-finetuned")

set_seed(SEED)

def to_hf(split):
    return HFDataset.from_list([{"text": r["text"], "label": r["label"]} for r in d[split]])

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
def encode(batch):
    return tok(batch["text"], truncation=True, max_length=MAX_LEN)

train_ds = to_hf("train.csv").map(encode, batched=True)
val_ds   = to_hf("valid.csv").map(encode, batched=True)
test_ds  = to_hf("test.csv").map(encode, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def metrics(eval_pred):
    logits, labels = eval_pred
    pred = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, pred)
    p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="binary", pos_label=1, zero_division=0)
    macro = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)[2]
    return {"accuracy": acc, "f1_hate": f1, "macro_f1": macro}

args = TrainingArguments(
    output_dir=OUT_DIR,
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
    train_dataset=train_ds, eval_dataset=val_ds,
    tokenizer=tok, data_collator=DataCollatorWithPadding(tok),
    compute_metrics=metrics,
)

trainer.train()

print("\n=== TEST SET ===")
test_metrics = trainer.evaluate(test_ds)
print(json.dumps(test_metrics, indent=2))

# Save final + export metrics
trainer.save_model(OUT_DIR)
with open(os.path.join(ROOT, "model", "roberta_metrics.json"), "w") as f:
    json.dump(test_metrics, f, indent=2)
print(f"saved -> {OUT_DIR}")

# Confusion matrix on test
preds = trainer.predict(test_ds)
y_pred = np.argmax(preds.predictions, axis=-1)
y_true = np.array([r["label"] for r in d["test.csv"]])
cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
print("confusion (rows=true 0/1):", cm)
