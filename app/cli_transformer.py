try:
    from .chatbot_transformer import ChatbotTransformer
except ImportError:
    from app.chatbot_transformer import ChatbotTransformer

def main():
    print("Transformer NLP Chatbot. Type 'exit' to quit.")
    bot = ChatbotTransformer(threshold=0.25)
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
