"""Balikas — Filipino Hate Speech Detection (HuggingFace Spaces, Gradio).

Loads the Tagalog-trained TF-IDF + LogReg baseline and serves a single-input
demo. Cebuano input works as best-effort zero-shot (see main repo README for
the cross-lingual study and its caveats).
"""
import os, joblib
import gradio as gr

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "baseline_tl.joblib")
pipe = joblib.load(MODEL)
LABELS = {0: "non-hate", 1: "hate"}

def classify(text):
    if not text or not text.strip():
        return {LABELS[0]: 0.0, LABELS[1]: 0.0}, "_(empty input)_"
    proba = pipe.predict_proba([text])[0]
    label_id = int(proba.argmax())
    conf = float(proba[label_id])
    pred_line = f"**{LABELS[label_id].upper()}** ({conf:.1%} confidence)"
    return {LABELS[i]: float(p) for i, p in enumerate(proba)}, pred_line

examples = [
    ["TANG INA MO talaga eh bobo"],
    ["Bobo mo naman, ulol ka!"],
    ["Maayong buntag sa tanan, maayo unta ang inyong adlaw."],
    ["Salamat kaayo sa tabang nimo kagahapon."],
    ["Buang kaayo ka, ulol!"],
]

with gr.Blocks(title="Balikas — Filipino Hate Speech") as demo:
    gr.Markdown("# \U0001f6e1\ufe0f Balikas — Filipino Hate Speech Detection")
    gr.Markdown(
        "Tagalog-trained TF-IDF + LogReg baseline. "
        "**F1 (hate) = 0.749** on the 4,232-tweet Cabasag et al. (2019) "
        "held-out test set. Cebuano input is classified **zero-shot** "
        "(transfer F1 = 0.833 on a 40-sentence hand-curated set; see main "
        "repo README for caveats — the number is biased by explicit "
        "profanity overlap)."
    )
    with gr.Row():
        inp = gr.Textbox(placeholder="Type a Filipino tweet or sentence…",
                         label="Input text", lines=3)
    with gr.Row():
        out_label = gr.Label(num_top_classes=2, label="Confidence")
        out_pred = gr.Markdown(label="Prediction")
    btn = gr.Button("Classify", variant="primary")
    btn.click(fn=classify, inputs=inp, outputs=[out_label, out_pred])
    gr.Examples(examples=examples, inputs=inp)
    gr.Markdown(
        "---\n"
        "*Model is Tagalog-trained. Cebuano input is best-effort zero-shot. "
        "No native Bisaya hate-speech benchmark exists (see main repo README "
        "for the methodology discussion).*\n\n"
        "**Dataset:** [jcblaise/hatespeech_filipino](https://huggingface.co/datasets/jcblaise/hatespeech_filipino) "
        "(Apache-2.0, Cabasag et al. 2019).  \n"
        "**Code:** [github.com/kiergabelo/balikas](https://github.com/kiergabelo/balikas)."
    )

if __name__ == "__main__":
    demo.launch()
