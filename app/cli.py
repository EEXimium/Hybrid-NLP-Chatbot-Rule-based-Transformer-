try:
    from .chatbot import Chatbot
except ImportError:
    from app.chatbot import Chatbot

def main():
    print("Simple NLP Chatbot (rule-based + ML). Type 'exit' to quit.")
    bot = Chatbot(threshold=0.25)  # transformer eşiği
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            break
        if text.lower() in {"exit", "quit"}:
            print("bye!")
            break
        reply = bot.respond(text)
        print(f"bot> {reply}")

if __name__ == "__main__":
    main()
