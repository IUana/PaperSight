import io
import re
from dataclasses import dataclass
from typing import Any

from pypdf import PdfReader


@dataclass(frozen=True)
class ExtractedFigure:
    page: int
    figure_id: str
    image_bytes: bytes
    mime_type: str
    caption: str | None = None


@dataclass(frozen=True)
class ParsedPdfContent:
    pages: list[tuple[int, str]]
    figures: list[ExtractedFigure]


def normalize_text(text: str) -> str:
    normalized = text.replace("\x00", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def extract_pages_from_pdf(pdf_bytes: bytes) -> list[tuple[int, str]]:
    return extract_pdf_content(pdf_bytes).pages


def extract_pdf_content(pdf_bytes: bytes, max_figures_per_page: int = 4) -> ParsedPdfContent:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages: list[tuple[int, str]] = []
    figures: list[ExtractedFigure] = []

    for page_idx, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned_text = normalize_text(raw_text)
        pages.append((page_idx, cleaned_text))

        figures.extend(_extract_page_figures(page, page_idx, cleaned_text, max_figures_per_page))

    return ParsedPdfContent(pages=pages, figures=figures)


def _extract_page_figures(
    page: Any,
    page_no: int,
    page_text: str,
    max_figures_per_page: int,
) -> list[ExtractedFigure]:
    images = getattr(page, "images", None)
    if images is None:
        return []

    extracted: list[ExtractedFigure] = []
    for idx, image_obj in enumerate(images, start=1):
        if idx > max_figures_per_page:
            break

        image_bytes = _extract_image_bytes(image_obj)
        if not image_bytes:
            continue

        image_name = str(getattr(image_obj, "name", "") or "")
        mime_type = _infer_mime_type(image_name)
        caption = _extract_figure_caption(page_text)

        extracted.append(
            ExtractedFigure(
                page=page_no,
                figure_id=f"p{page_no}_f{idx}",
                image_bytes=image_bytes,
                mime_type=mime_type,
                caption=caption,
            )
        )

    return extracted


def _extract_image_bytes(image_obj: Any) -> bytes | None:
    data = getattr(image_obj, "data", None)
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)

    get_data = getattr(image_obj, "get_data", None)
    if callable(get_data):
        try:
            data = get_data()
            if isinstance(data, bytes):
                return data
            if isinstance(data, bytearray):
                return bytes(data)
        except Exception:
            return None

    return None


def _infer_mime_type(image_name: str) -> str:
    lowered = image_name.lower()
    if lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
        return "image/jpeg"
    if lowered.endswith(".png"):
        return "image/png"
    if lowered.endswith(".webp"):
        return "image/webp"
    if lowered.endswith(".bmp"):
        return "image/bmp"
    if lowered.endswith(".gif"):
        return "image/gif"
    return "image/png"


def _extract_figure_caption(page_text: str) -> str | None:
    if not page_text:
        return None

    for line in page_text.splitlines():
        snippet = line.strip()
        if not snippet:
            continue

        lowered = snippet.lower()
        if lowered.startswith("figure") or lowered.startswith("fig."):
            return snippet[:280]

    for line in page_text.splitlines():
        snippet = line.strip()
        if not snippet:
            continue
        if "pipeline" in snippet.lower():
            return snippet[:280]

    return None
