import re
from typing import Optional

RULES = [
    # Greeting (TR/EN)
    (re.compile(r"\b(hi|hello|hey|selam|merhaba|good (morning|evening))\b", re.I),
     "Hello! How can I help you today?"),

    # Thanks
    (re.compile(r"\b(thanks|thank you|teşekkür(ler| ederim)?|çok sağol)\b", re.I),
     "You're welcome!"),

    # Opening hours
    (re.compile(r"\b(hours?|open|opening|kaçta|çalışma saat(leri)?|ne zaman açıksınız)\b", re.I),
     "We're open from 09:00 to 18:00 on weekdays."),

    # Location
    (re.compile(r"\b(where|address|konum|nerede?siniz|adres)\b", re.I),
     "We're located in Istanbul."),

    # Pricing
    (re.compile(r"\b(price|pricing|ücret|fiyat|kaça|ne kadar)\b", re.I),
     "You can find our pricing on the website. Need a link?"),

    # Help / capabilities
    (re.compile(r"\b(help|yardım|what can you do|neler yapabilirsin)\b", re.I),
     "I can answer common questions like hours, location, pricing, and more."),
]

def match_rule(text: str) -> Optional[str]:
    for pattern, response in RULES:
        if pattern.search(text):
            return response
    return None
