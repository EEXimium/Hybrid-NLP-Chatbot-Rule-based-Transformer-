# 🧠 Hybrid NLP Chatbot (Rule-based + Transformer)

This project is a simple **NLP Chatbot** that combines **rule-based logic** with a **transformer-based intent classification model**.  

---

## 🚀 Features
- Hybrid system: Rule-based + Transformer classifier
- Intent classification with DistilBERT (or any Hugging Face model)
- Custom dataset support via `data/intents.json`
- CLI interface for quick testing
- Extensible architecture for future improvements

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/yourusername/nlp-chatbot-starter.git
cd nlp-chatbot-starter
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 📦 Usage

### 1. Training the Model
Before running the chatbot, fine-tune the transformer model on your intents dataset:

```bash
python -m app.train_intent_transformer
```

- The model will be saved inside:  
  `models/transformer_intent/`  
- Labels are exported to `labels.json` for later inference.

---

### 2. Running the Chatbot
After training, launch the chatbot via the CLI:

```bash
python -m app.cli
```

Example session:
```
Simple NLP Chatbot (rule-based + ML). Type 'exit' to quit.
you> hello
bot> Hi there! How can I help you?
you> how are you
bot> I'm just a bunch of code, but thanks for asking!
you> bye
bot> Goodbye! Have a great day.
```

---

### 3. Adding New Intents
To extend the chatbot:
1. Edit [`data/intents.json`](data/intents.json).  
2. Add a new intent with:
   - `name`
   - `samples` (training examples)
   - `responses` (possible replies)  

Example:
```json
{
  "name": "greeting",
  "samples": ["hi", "hello", "hey there"],
  "responses": ["Hello!", "Hi, how can I help you today?"]
}
```
3. Re-run the training script:
```bash
python -m app.train_intent_transformer
```

---

## 📊 Model Performance
The transformer model is evaluated on the validation set with the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

Example results (on sample dataset):
```
Accuracy: 90%
F1-score: 88%
Precision: 87%
Recall: 86%
```

---

## 📌 Future Improvements
- Web interface with **Flask**, **Gradio**, or **Streamlit**.  
- Support for multilingual datasets (e.g., Turkish + English).  
- Integrate context-aware responses (multi-turn conversations).  
- Deployment-ready Docker container.  

---

## 📝 License
This project is released under the MIT License.  
Feel free to use and adapt it for your own projects.  

---

## 👨‍💻 Author
Developed by **Yagiz Efe Atasever** as part of NLP & Chatbot projects.  
