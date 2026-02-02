# text_extraction2.py
import pymupdf.layout
import pymupdf4llm
import fitz
from typing import List, Dict
from pdf_cleaning4 import clean_pdf_text


def load_pdf_as_text(source: bytes) -> List[Dict]:
    pdf_doc = fitz.open(stream=source, filetype="pdf")
    pdf = pymupdf4llm.to_markdown(pdf_doc, page_chunks=True, filetype="pdf")

    pages = []
    for page in pdf:
        text = clean_pdf_text(page["text"])
        if not text:
            continue
        pages.append(
            {
                "page": page["metadata"]["page_number"],
                "content": text,
                "title": page["metadata"]["title"],
                "author": page["metadata"].get("author", ""),
                "file_path": page["metadata"].get("file_path", ""),
            }
        )

    return pages
