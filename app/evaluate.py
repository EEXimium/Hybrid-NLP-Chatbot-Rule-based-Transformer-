import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "intents.json"
OUT_DIR = BASE / "models"
OUT_DIR.mkdir(exist_ok=True)

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

def main():
    intents = load_intents(DATA)
    X, y = build_dataset(intents)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), lowercase=True)),
        ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", multi_class="ovr")),
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    report = classification_report(y_test, y_pred, digits=3)
    (OUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)

    labels = sorted(set(y))
    cm = confusion_matrix(y_test, y_pred, labels=labels, normalize=None)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    plt.figure(figsize=(8, 8))
    disp.plot(include_values=True, xticks_rotation=45, cmap=None)  # no specific colors
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close()

    import joblib
    joblib.dump(pipe, OUT_DIR / "intent_pipeline.joblib")

    print("Saved: models/classification_report.txt, models/confusion_matrix.png, models/intent_pipeline.joblib")

if __name__ == "__main__":
    main()
