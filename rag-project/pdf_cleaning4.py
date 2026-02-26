# pdf_cleaing4.py
import re
from pathlib import Path


def clean_pdf_text(text: str) -> str:
    text = re.sub(
        r"==> picture \[.*?\] (?:intentionally omitted )?<==", "[Image]", text
    )
    text = re.sub(r"\*\*", "", text)
    text = re.sub(r"__", "", text)
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\|\s]*[-=\s]+[\|\s]*$", "", text, flags=re.MULTILINE)
    text = re.sub(r" {2,}", " ", text)
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    return text.strip()
