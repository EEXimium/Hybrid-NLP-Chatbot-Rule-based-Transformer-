import json
from pathlib import Path
import os
from typing import List, Dict

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "data" / "intents.json"
OUT_DIR = BASE / "models" / "transformer_intent"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = os.environ.get("INTENT_MODEL_NAME", "distilbert-base-uncased")  # örn: 'dbmdz/bert-base-turkish-cased'

def load_intents(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)["intents"]

def build_dataset(intents: List[Dict]):
    texts, labels = [], []
    for it in intents:
        name = it["name"]
        for s in it["samples"]:
            s = (s or "").strip()
            if s:
                texts.append(s)
                labels.append(name)
    return texts, labels

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main():
    intents = load_intents(DATA)
    texts, labels = build_dataset(intents)

    # Label encode
    le = LabelEncoder()
    y = le.fit_transform(labels)
    id2label = {i: lab for i, lab in enumerate(le.classes_)}
    label2id = {lab: i for i, lab in id2label.items()}

    # Basit split
    split = int(0.8 * len(texts)) if len(texts) > 1 else len(texts)
    train_texts, val_texts = texts[:split], texts[split:]
    train_y, val_y = y[:split], y[split:]

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tok(batch["text"], truncation=True, padding="max_length", max_length=64)

    train_ds = Dataset.from_dict({"text": train_texts, "label": train_y}).map(tokenize, batched=True)
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_y}).map(tokenize, batched=True)

    # Sadece modelin beklediği sütunlar kalsın (eski sürümlerde güvenli)
    keep_cols = {"input_ids", "attention_mask", "label"}
    train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in keep_cols])
    val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in keep_cols])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(id2label), id2label=id2label, label2id=label2id
    )

    # Yeni sürüm varsa yeni argümanları kullan; yoksa sade argümanlara düş
    try:
        args = TrainingArguments(
            output_dir=str(OUT_DIR / "runs"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10,
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=4,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )
    except TypeError:
        # Eski transformers için: evaluation/save stratejileri yok → eğitimden sonra ayrı evaluate edeceğiz
        args = TrainingArguments(
            output_dir=str(OUT_DIR / "runs"),
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=4,
            weight_decay=0.01,
            logging_steps=10,
        )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds if len(val_texts) > 0 else None,
        tokenizer=tok,
        compute_metrics=compute_metrics if len(val_texts) > 0 else None,
    )

    trainer.train()

    # Eski sürüm argümanlarında otomatik eval yok → burada çağır
    if len(val_texts) > 0:
        try:
            print(trainer.evaluate())
        except Exception:
            pass

    # Kaydet
    trainer.save_model(str(OUT_DIR))
    tok.save_pretrained(str(OUT_DIR))
    (OUT_DIR / "labels.json").write_text(json.dumps(id2label, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved transformer model to: {OUT_DIR}")

if __name__ == "__main__":
    main()
