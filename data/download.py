"""Fetch + parse jcblaise/hatespeech_filipino from the HF Hub.

Bypasses the dataset's loading script (datasets>=3 dropped script support) by
downloading the raw zip directly and parsing the CSVs the same way the
original loader did.
"""
import os, csv, json, zipfile, urllib.request

URL = "https://huggingface.co/datasets/jcblaise/hatespeech_filipino/resolve/main/hatespeech_raw.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ZIP = os.path.join(HERE, "hatespeech_raw.zip")
EXTRACT = os.path.join(HERE, "hatespeech")
OUT = os.path.join(HERE, "splits.json")

def main():
    if not os.path.exists(ZIP):
        print(f"downloading {URL} ...")
        req = urllib.request.Request(URL, headers={"User-Agent": "alerto-hate/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(ZIP, "wb") as f:
            f.write(r.read())
        print(f"  {os.path.getsize(ZIP)} bytes")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(HERE)

    splits = {}
    for name in ("train", "valid", "test"):
        path = os.path.join(EXTRACT, f"{name}.csv")
        rows = []
        with open(path, encoding="utf-8") as f:
            rdr = csv.reader(f, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
            next(rdr, None)
            for row in rdr:
                if len(row) == 2:
                    rows.append({"text": row[0], "label": int(row[1])})
        splits[f"{name}.csv"] = rows
        from collections import Counter
        c = Counter(r["label"] for r in rows)
        print(f"{name}: {len(rows)} rows, labels={dict(c)}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(splits, f, ensure_ascii=False)
    print(f"saved -> {OUT}")
    print(f"\nDataset: Cabasag et al. (2019), Philippine Computing Journal.")
    print(f"License: Apache-2.0 (per HuggingFace dataset card).")

if __name__ == "__main__":
    main()
