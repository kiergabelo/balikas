"""Headless smoke test for the Gradio app's classify() function.
Boots the model + UI without launching the server.
"""
import os, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.join(HERE, "app.py")

# Load app.py as a module (it runs at import time — model load + gr.Blocks build)
spec = importlib.util.spec_from_file_location("balikas_space_app", APP)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

cases = [
    ("TANG INA MO talaga eh bobo", 1),                # explicit hate
    ("Maayong buntag sa tanan.", 0),                  # benign greeting
    ("Buang kaayo ka, ulol!", 1),                     # Cebuano hate
    ("Salamat kaayo sa tabang nimo kagahapon.", 0),    # Cebuano benign
]
ok = 0
for text, expected in cases:
    label_dict, pred_line = mod.classify(text)
    predicted_id = 1 if label_dict.get("hate", 0) > label_dict.get("non-hate", 0) else 0
    match = predicted_id == expected
    ok += match
    print(f"[{'OK' if match else 'FAIL'}] want={expected} got={predicted_id}  {text[:50]!r}  -> {pred_line}")
print(f"\n{ok}/{len(cases)} cases classified correctly")
sys.exit(0 if ok == len(cases) else 1)
