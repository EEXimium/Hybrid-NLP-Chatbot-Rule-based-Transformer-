import json
import random
from pathlib import Path

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .rule_based import match_rule  # regex tabanlı kurallar

BASE = Path(__file__).resolve().parents[1]
INTENTS_PATH = BASE / "data" / "intents.json"
MODEL_DIR = BASE / "models" / "transformer_intent"

def load_intents():
    with open(INTENTS_PATH, encoding="utf-8") as f:
        return json.load(f)

class Chatbot:
    """
    Sıra:
      1) rule-based (yüksek kesinlikli cevaplar)
      2) transformer intent sınıflandırma (confidence kontrolü)
      3) fallback
    """
    def __init__(self, threshold: float = 0.40):
        self.intents = load_intents()
        self.intent_map = {i["name"]: i for i in self.intents["intents"]}
        # Transformer modeli yükle
        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Transformer model bulunamadı: {MODEL_DIR}\n"
                f"Önce eğit: python -m app.train_intent_transformer"
            )
        self.tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        self.labels = json.loads((MODEL_DIR / "labels.json").read_text(encoding="utf-8"))
        # labels.json {id: name}; pipeline LABEL_0 yerine id döndürürsek map’leriz
        # eşiği sakla
        self.threshold = threshold

    def _predict_intent(self, text: str):
        inputs = self.tok(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        # labels.json anahtarları str olabilir
        intent = self.labels.get(str(idx), self.labels.get(idx, f"LABEL_{idx}"))
        return intent, conf

    def get_response(self, text: str) -> str:
        # 1) Rule-based
        rb = match_rule(text)
        if rb:
            print("[dbg] rule-based matched")
            return rb

        # 2) Transformer intent
        intent, conf = self._predict_intent(text)
        print(f"[dbg] transformer intent={intent} conf={conf:.2f} (thr={self.threshold:.2f})")
        if conf < self.threshold or intent not in self.intent_map:
            return self.intent_map["fallback"]["responses"][0]

        responses = self.intent_map[intent]["responses"]
        return random.choice(responses) if responses else self.intent_map["fallback"]["responses"][0]

    # CLI geriye uyum: cli.py 'respond' çağırıyor
    def respond(self, text: str) -> str:
        return self.get_response(text)
