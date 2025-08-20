from pathlib import Path
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "models" / "transformer_intent"

class TransformerIntentClassifier:
    def __init__(self, threshold: float = 0.40):
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"Transformer model not found at {MODEL_DIR}. Train it: python -m app.train_intent_transformer")
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
        self.pipe = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True)
        self.labels = json.loads((MODEL_DIR / "labels.json").read_text(encoding="utf-8"))
        self.threshold = threshold

    def predict(self, text: str):
        scores = self.pipe(text)[0]
        best = max(scores, key=lambda x: x['score'])
        label = best['label']
        conf = float(best['score'])
        if label.startswith('LABEL_'):
            idx = int(label.split('_')[-1])
            label = self.labels.get(idx, label)
        return label, conf
