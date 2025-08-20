try:
    from ..app.chatbot import Chatbot
except ImportError:
    from app.chatbot import Chatbot

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="NLP Chatbot API")
bot = Chatbot(threshold=0.25)

class Query(BaseModel):
    text: str

@app.post("/chat")
def chat(q: Query):
    reply = bot.respond(q.text)
    return {"reply": reply}
