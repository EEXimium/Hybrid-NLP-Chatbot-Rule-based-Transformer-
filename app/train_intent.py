import json
from pathlib import Path
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "intents.json"
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

def load_intents(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["intents"]

def build_dataset(intents):
    X, y = [], []
    for intent in intents:
        name = intent["name"]
        for s in intent["samples"]:
            if s.strip():
                X.append(s)
                y.append(name)
    return X, y

def ensure_nltk():
    try:
        _ = stopwords.words("english")
    except LookupError:
        nltk.download("stopwords")
    try:
        _ = word_tokenize("hi")
    except LookupError:
        nltk.download("punkt")

def main():
    intents = load_intents(DATA)
    X, y = build_dataset(intents)

    ensure_nltk()

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), lowercase=True)),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="ovr")),
    ])

    scores = cross_val_score(pipe, X, y, cv=5, scoring="f1_micro")
    print(f"CV f1_micro: {scores.mean():.3f} (+/- {scores.std():.3f})")

    pipe.fit(X, y)

    import joblib, json as _json
    joblib.dump(pipe, MODELS / "intent_pipeline.joblib")

    labels = sorted(set(y))
    with open(MODELS / "labels.json", "w", encoding="utf-8") as f:
        _json.dump(labels, f, indent=2, ensure_ascii=False)

    y_pred = pipe.predict(X)
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    main()
