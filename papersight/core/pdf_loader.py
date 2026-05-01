import io
import re

from pypdf import PdfReader


def normalize_text(text: str) -> str:
    normalized = text.replace("\x00", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_pages_from_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []
    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        pages.append((page_idx, normalize_text(raw_text)))
    return pages
