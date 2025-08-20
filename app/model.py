from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parents[1]
PIPE_PATH = BASE / "models" / "intent_pipeline.joblib"

class MLIntentClassifier:
    def __init__(self):
        if not PIPE_PATH.exists():
            raise FileNotFoundError(f"Model not found at {PIPE_PATH}. Train first: python -m app.train_intent")
        self.pipe = joblib.load(PIPE_PATH)

    def predict(self, text: str):
        probs = self.pipe.predict_proba([text])[0]
        labels = self.pipe.classes_
        best_idx = probs.argmax()
        return labels[best_idx], float(probs[best_idx])
