#!/usr/bin/env python3
# pdf_cleaing4.py
import re
from pathlib import Path


def clean_pdf_text(text: str) -> str:

    # 1. Remove picture placeholders
    text = re.sub(
        r"==> picture \[.*?\] (?:intentionally omitted )?<==", "[Image]", text
    )

    # 2. Remove bold markers
    text = re.sub(r"\*\*", "", text)

    # 3. Remove underscores (underline/italic)
    text = re.sub(r"__", "", text)

    # 4. Remove <br> tags and replace with space
    text = re.sub(r"<br\s*/?>", " ", text)

    # 5. Remove header markers (##, ###, etc.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # 6. Remove ONLY table separator lines (---, ===, |---|)
    text = re.sub(r"^[\|\s]*[-=\s]+[\|\s]*$", "", text, flags=re.MULTILINE)

    # 7. Fix multiple spaces in a row
    text = re.sub(r" {2,}", " ", text)

    # 8. Remove ALL blank lines and trim lines
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        cleaned_lines.append(stripped)

    text = "\n".join(cleaned_lines)

    # 9. Clean up any leading/trailing whitespace
    return text.strip()
