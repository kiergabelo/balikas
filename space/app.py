"""Balikas — Filipino Hate Speech Detection (HuggingFace Spaces, Gradio).

XLM-RoBERTa fine-tuned on combined Tagalog + Filipino TikTok corpus.
Loads the model from kiergabelo/balikas-xlm on the HF Hub.
"""
import os
import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "kiergabelo/balikas-xlm"

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
LABELS = {0: "non-hate", 1: "hate"}

def classify(text):
    if not text or not text.strip():
        return {LABELS[0]: 0.0, LABELS[1]: 0.0}, "_(empty input)_"
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1)[0]
    label_id = int(probs.argmax())
    conf = float(probs[label_id])
    pred_line = f"**{LABELS[label_id].upper()}** ({conf:.1%} confidence)"
    return {LABELS[i]: float(probs[i]) for i in range(2)}, pred_line

examples = [
    ["TANG INA MO talaga eh bobo"],
    ["Bobo mo naman, ulol ka!"],
    ["Maayong buntag sa tanan, maayo unta ang inyong adlaw."],
    ["Salamat kaayo sa tabang nimo kagahapon."],
    ["Buang kaayo ka, ulol!"],
    ["Putang ina ng gobyernong ito, walang silbi!"],
    ["Ganahan ko nga mo-eskwela ko karong adlawa."],
]

with gr.Blocks(title="Balikas — Filipino Hate Speech") as demo:
    gr.Markdown("# \U0001f6e1\ufe0f Balikas — Filipino Hate Speech Detection")
    gr.Markdown(
        "XLM-RoBERTa fine-tuned on **43,892 Filipino social media samples** "
        "(Tagalog election tweets + TikTok transcriptions including code-switched "
        "Taglish and Cebuano). **F1 = 0.917** on held-out Tagalog test. "
        "Type any Filipino or Cebuano text to classify it."
    )
    with gr.Row():
        inp = gr.Textbox(placeholder="Type a Filipino or Cebuano tweet...",
                         label="Input text", lines=3)
    with gr.Row():
        out_label = gr.Label(num_top_classes=2, label="Confidence")
        out_pred = gr.Markdown(label="Prediction")
    btn = gr.Button("Classify", variant="primary")
    btn.click(fn=classify, inputs=inp, outputs=[out_label, out_pred])
    gr.Examples(examples=examples, inputs=inp)
    gr.Markdown(
        "---\n"
        "**Model:** XLM-RoBERTa fine-tuned on combined Tagalog + Filipino TikTok corpus.\n\n"
        "**Dataset:** [jcblaise/hatespeech_filipino](https://huggingface.co/datasets/jcblaise/hatespeech_filipino) "
        "(Cabasag et al. 2019) + "
        "[SEACrowd/filipino_hatespeech_tiktok](https://huggingface.co/datasets/SEACrowd/filipino_hatespeech_tiktok) "
        "(Hernandez et al. 2021).\n\n"
        "**Code:** [github.com/kiergabelo/balikas](https://github.com/kiergabelo/balikas)."
    )

if __name__ == "__main__":
    demo.launch()
