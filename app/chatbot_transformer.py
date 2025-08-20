import json, random
from pathlib import Path
from .rule_based import match_rule
from .model_transformer import TransformerIntentClassifier

BASE = Path(__file__).resolve().parents[1]
INTENTS_PATH = BASE / "data" / "intents.json"

def load_intents():
    with open(INTENTS_PATH, encoding="utf-8") as f:
        return json.load(f)

class ChatbotTransformer:
    def __init__(self, threshold: float = 0.40):
        self.intents = load_intents()
        self.intent_map = {i["name"]: i for i in self.intents["intents"]}
        self.clf = TransformerIntentClassifier(threshold=threshold)
        self.threshold = threshold

    def respond(self, text: str) -> str:
        rule = match_rule(text)
        if rule:
            print("[dbg] rule-based matched")
            return rule

        label, conf = self.clf.predict(text)
        print(f"[dbg] transformer label={label} conf={conf:.2f} (thr={self.threshold:.2f})")
        if conf < self.threshold or label not in self.intent_map:
            return self.intent_map["fallback"]["responses"][0]
        responses = self.intent_map[label]["responses"]
        return random.choice(responses)
