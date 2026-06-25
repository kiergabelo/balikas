"""Small hand-curated Cebuano hate speech evaluation set.

This is NOT a representative benchmark. It is a small (~40-sample) set of
unambiguous sentences constructed to demonstrate the zero-shot cross-lingual
transfer gap from a Tagalog-trained model. Each sentence is either clearly
hate (containing Bisaya profanity or election-cycle aggression) or clearly
non-hate (greetings, news-style, neutral statements).

It exists because no labeled Cebuano hate-speech corpus exists at the time
of writing. The honest framing: a Tagalog-trained TF-IDF+LogReg baseline is
expected to fail on this set almost by construction (vocabularies barely
overlap), which IS the finding — classical features can't transfer
zero-shot across Filipino languages. The right fix is a multilingual
transformer (XLM-RoBERTa), noted as future work.
"""
import json, os

# (text, label, note) — label 1 = hate, 0 = non-hate
SAMPLES = [
    # --- non-hate (clearly benign) ---
    ("Maayong buntag sa tanan, maayo unta ang inyong adlaw.", 0, "greeting"),
    ("Salamat kaayo sa tabang nimo kagahapon.", 0, "gratitude"),
    ("Ang bag-ong tulay sa lungsod gisirhan na sa miaging semana.", 0, "news"),
    ("Magtuon ko sa eksamen para sa sunod nga adlaw.", 0, "neutral"),
    ("Mga estudyante sa unibersidad nagtipon karong Sabado.", 0, "neutral"),
    ("Maayong buntag, kumusta ang imong pagpangita og trabaho?", 0, "conversational"),
    ("Naglibang ko sa parke kagahapon gabii.", 0, "neutral"),
    ("Ang akon pamilya nagapuyo sa Cebu.", 0, "neutral"),
    ("Sikat kaayo ang lumpia sa pista.", 0, "neutral"),
    ("Maayo ang panahon karon, uwanan unya gabii.", 0, "neutral"),
    ("Aduna bay magapamaligya og isda diri?", 0, "neutral"),
    ("Ang simbahan sa lungsod daan na kaayo apan gipreserbar.", 0, "neutral"),
    ("Mga bata nagadula sa plasa.", 0, "neutral"),
    ("Ang akon igsoon nagatrabaho sa syudad.", 0, "neutral"),
    ("Nagtuon ko og Sinugbuanon karong tuiga.", 0, "neutral"),
    ("Maayong gabii kaninyong tanan.", 0, "greeting"),
    ("Salamat sa pagtabang sa among komunidad.", 0, "gratitude"),
    ("Ang bag-ong eskwelahan gibuksan na niadtong lunes.", 0, "news"),
    ("Magbakal ko og bag-ong sapatos sa pula nga presyo.", 0, "neutral"),
    ("Mga mag-uuma nag-ani sa uma kung umaga.", 0, "neutral"),
    # --- hate (contains clear Bisaya profanity / aggression, election-themed) ---
    ("Buanga kaayo ka, ulol!", 1, "profanity + insult"),
    ("Yawa ka, ulay sa imong kinabuhi!", 1, "profanity insult"),
    ("Putang ina mo, inyong kandidato bura!", 1, "profanity"),
    ("Animal ka, dili ka tao!", 1, "dehumanizing insult"),
    ("Baboy ka, hilom na lang unta ka.", 1, "dehumanizing insult"),
    ("Buang ka, ulol, tanga! Wala kay nahibaloan!", 1, "profanity chain"),
    ("Ang imong kandidata, hakog kaayo, kawatan!", 1, "political slur"),
    ("Patyon ta ikaw kung dili ka motiil.", 1, "death threat"),
    ("Mga kawatan ang pulitiko, tanan sila ulol!", 1, "political slur chain"),
    ("Pisti, inyong gobyerno way ayo, tanan!", 1, "profanity + broad attack"),
    ("Putang ina inyong pamilya, tanan mogasto sa kwarta sa tawo!", 1, "profanity + accusation"),
    ("Bilat sa imong nawong, bilib kaayo ko nimo ulol!", 1, "profanity + insult"),
    ("Inyong mayor, kawatan, lagpak ang pangulo!", 1, "political slur"),
    ("Tanga ka, wa kay kahibalo sa imong gipanghambael!", 1, "intellectual insult"),
    ("Linte, piste, pisti inyong tanan!", 1, "profanity chain"),
    ("Ulol mong tanan, bilib man mo sa inyong kandidato bura!", 1, "profanity + group attack"),
    ("Pisti, awto mo, paborito mo ang mga lahing bura!", 1, "profanity + racist slur"),
    ("Buwisit mo, ulol ka, yawa, ulol mong pamilya!", 1, "profanity chain"),
    ("Tanga kaayo ang botante, tanan sila ulol!", 1, "voter dehumanization"),
    ("Patyon nominal ra ang dili mosugot, tanan lagpason!", 1, "threat"),
]

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "data", "ceb_eval_handcurated.json")

def main():
    rows = [{"text": t, "label": l, "note": n} for (t, l, n) in SAMPLES]
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    from collections import Counter
    c = Counter(r["label"] for r in rows)
    print(f"wrote {len(rows)} samples -> {OUT}")
    print(f"dist: {dict(c)} (0=non-hate, 1=hate)")

if __name__ == "__main__":
    main()
