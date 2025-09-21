import re

def sanitize_input(text: str) -> str:
    if not text or not text.strip():
        return ""

    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Collapse multiple newlines (e.g., lots of blank lines → one newline)
    text = re.sub(r"\n{2,}", "\n", text)

    # 3. Remove HTML tags like <script>, <div>, etc.
    text = re.sub(r"<.*?>", " ", text)

    # 4. Collapse multiple spaces & tabs
    text = re.sub(r"[ \t]+", " ", text)

    # 5. Collapse spaces around punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)

    # 6. Strip leading/trailing whitespace
    text = text.strip()

    return text