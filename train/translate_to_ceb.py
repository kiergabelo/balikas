"""Translate the Tagalog splits to Cebuano via NLLB-200 distilled.

Cross-lingual transfer: no native Cebuano hate-speech corpus exists, so we
machine-translate the labeled Tagalog corpus (Cabasag et al. 2019) and train
on the translated labels. The translation noise is acknowledged in the README.
"""
import os, json, time, sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "data", "splits.json")
OUT = os.path.join(ROOT, "data", "splits_ceb.json")

MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "tgl_Latn"
TGT_LANG = "ceb_Ceb"
BATCH = 32
MAX_LEN = 128

print(f"loading {MODEL_NAME} ...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()
device = "cpu"  # NLLB CPU is fine for this volume
model.to(device)

# Force the target language BOS token
tgt_lang_id = tok.convert_tokens_to_ids(TGT_LANG)

def translate_batch(texts):
    cleaned = [(t or "").strip() or "." for t in texts]
    tok_in = tok(cleaned, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
    with torch.no_grad():
        gen = model.generate(
            **tok_in,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=64,
            num_beams=1,           # greedy for speed
            do_sample=False,
        )
    return tok.batch_decode(gen, skip_special_tokens=True)

with open(SRC, encoding="utf-8") as f:
    src = json.load(f)

# Resume support: load any partial already-translated splits.
out = {}
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        out = json.load(f)
    print(f"resuming; already have: {list(out.keys())}", flush=True)

remaining = [(k, v) for k, v in src.items() if k not in out]
total_rows = sum(len(v) for _, v in remaining)
print(f"translating {total_rows} rows across {len(remaining)} splits (batch={BATCH}) ...", flush=True)
t0 = time.time()

for split_name, rows in remaining:
    print(f"\n=== {split_name} ({len(rows)} rows) ===", flush=True)
    texts = [r["text"] for r in rows]
    labels = [r["label"] for r in rows]
    translated = []
    for i in tqdm(range(0, len(texts), BATCH), desc=split_name, file=sys.stdout):
        chunk = texts[i:i + BATCH]
        try:
            tr = translate_batch(chunk)
        except Exception as e:
            print(f"  batch {i} failed: {e}; skipping (keeping source)", flush=True)
            tr = chunk
        translated.extend(tr)
    out[split_name] = [{"text": t, "label": l} for t, l in zip(translated, labels)]
    for r in out[split_name][:2]:
        print(f"  [label={r['label']}] {r['text'][:140]}", flush=True)
    # Checkpoint after every split.
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"  checkpoint saved ({len(out[split_name])} rows)", flush=True)

dt = time.time() - t0
print(f"\nDONE in {dt:.1f}s ({dt/60:.1f} min); {dt/max(total_rows,1)*1000:.1f} ms/row")
print(f"saved -> {OUT}")
