from io import BytesIO
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from papersight.core.pdf_loader import ExtractedFigure, ParsedPdfContent
from papersight.core.service import PaperSightService
from papersight.core.settings import PaperSightSettings


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class FakeVectorIndex:
    def __init__(self) -> None:
        self.docs: list[Document] = []

    def has_documents(self) -> bool:
        return bool(self.docs)

    def add_documents(self, documents: list[Document]) -> None:
        self.docs.extend(documents)

    def remove_paper(self, paper_id: str) -> int:
        before = len(self.docs)
        self.docs = [doc for doc in self.docs if doc.metadata.get("paper_id") != paper_id]
        return before - len(self.docs)

    def clear(self) -> int:
        removed = len(self.docs)
        self.docs = []
        return removed

    def similarity_search(
        self,
        query: str,
        k: int,
        metadata_filter: dict | None = None,
        fetch_k: int | None = None,
    ):
        query_lower = query.lower()
        candidates = self.docs
        if metadata_filter:
            candidates = [
                doc
                for doc in candidates
                if all(doc.metadata.get(key) == value for key, value in metadata_filter.items())
            ]

        ranked: list[tuple[Document, float]] = []
        for doc in candidates:
            text = doc.page_content.lower()
            if query_lower in text:
                ranked.append((doc, 0.0))

        return ranked[:k]


class FakeFigureAnalyzer:
    def analyze(self, image_bytes: bytes, mime_type: str, caption: str | None = None):
        return {
            "figure_type": "pipeline",
            "is_pipeline": True,
            "confidence": 0.9,
            "steps": ["Encode input", "Aggregate features", "Predict output"],
            "components": ["Encoder", "Aggregator", "Head"],
            "relations": ["Encoder -> Aggregator", "Aggregator -> Head"],
            "summary": caption or "Pipeline diagram with three stages.",
        }


def make_service(tmp_path, parser, figure_analyzer=None):
    settings = PaperSightSettings(data_dir=tmp_path, chunk_size=30, chunk_overlap=10)
    splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=10, separators=[""])
    return PaperSightService(
        settings=settings,
        vector_index=FakeVectorIndex(),
        llm=FakeLLM("test answer [C1]"),
        pdf_parser=parser,
        figure_analyzer=figure_analyzer,
        splitter=splitter,
    )


@pytest.fixture
def workdir():
    base = Path("data") / "test_runs"
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"papersight-test-{uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=True)
    try:
        yield target
    finally:
        rmtree(target, ignore_errors=True)


def test_chunk_overlap_is_effective(workdir):
    def parser(_: bytes):
        return [(1, "A" * 80)]

    service = make_service(workdir, parser)
    upload = BytesIO(b"file-a")
    upload.name = "a.pdf"
    service.ingest_document(upload, title="A")

    chunks = service.vector_index.docs
    assert len(chunks) >= 2
    assert chunks[0].page_content[-10:] == chunks[1].page_content[:10]


def test_dedup_upload_by_content_hash(workdir):
    def parser(_: bytes):
        return [(1, "Dedup Content " * 10)]

    service = make_service(workdir, parser)
    first = BytesIO(b"same-file")
    first.name = "same.pdf"
    second = BytesIO(b"same-file")
    second.name = "same-again.pdf"

    first_result = service.ingest_document(first, title="same")
    second_result = service.ingest_document(second, title="same")

    assert first_result["deduplicated"] is False
    assert first_result["added_chunks"] > 0
    assert second_result["deduplicated"] is True
    assert second_result["added_chunks"] == 0


def test_scope_filter_global_and_single_paper(workdir):
    def parser(payload: bytes):
        return [(1, payload.decode("utf-8"))]

    service = make_service(workdir, parser)
    doc_a = BytesIO("alpha method details".encode("utf-8"))
    doc_a.name = "alpha.pdf"
    doc_b = BytesIO("beta experiment results".encode("utf-8"))
    doc_b.name = "beta.pdf"

    paper_a = service.ingest_document(doc_a, title="Alpha")["paper_id"]
    paper_b = service.ingest_document(doc_b, title="Beta")["paper_id"]

    global_result = service.answer_question("beta", scope_mode="global", top_k=1)
    assert global_result["citations"][0]["paper_id"] == paper_b

    paper_result = service.answer_question("alpha", scope_mode="paper", paper_id=paper_a, top_k=3)
    assert paper_result["citations"]
    assert all(c["paper_id"] == paper_a for c in paper_result["citations"])


def test_fallback_when_no_recall(workdir):
    def parser(_: bytes):
        return [(1, "only known text")]

    service = make_service(workdir, parser)
    upload = BytesIO(b"only-file")
    upload.name = "only.pdf"
    service.ingest_document(upload, title="Only")

    result = service.answer_question("unknown question", scope_mode="global", top_k=3)
    assert result["answer"] == service.settings.fallback_answer
    assert result["citations"] == []
    assert result["used_chunks"] == 0


def test_answer_contract_contains_expected_fields(workdir):
    def parser(_: bytes):
        return [(1, "method and results are here")]

    service = make_service(workdir, parser)
    upload = BytesIO(b"contract")
    upload.name = "contract.pdf"
    service.ingest_document(upload, title="Contract")

    result = service.answer_question("method", scope_mode="global", top_k=1)

    assert "answer" in result
    assert "citations" in result
    assert "used_chunks" in result
    assert result["citations"]
    assert {"paper_id", "page", "chunk_id", "snippet"}.issubset(result["citations"][0].keys())


def test_ingest_adds_text_and_figure_chunks(workdir):
    def parser(_: bytes):
        return ParsedPdfContent(
            pages=[(1, "Method section explains the approach.")],
            figures=[
                ExtractedFigure(
                    page=1,
                    figure_id="p1_f1",
                    image_bytes=b"fake-image",
                    mime_type="image/png",
                    caption="Figure 1: Proposed pipeline.",
                )
            ],
        )

    service = make_service(workdir, parser, figure_analyzer=FakeFigureAnalyzer())
    upload = BytesIO(b"hybrid")
    upload.name = "hybrid.pdf"

    result = service.ingest_document(upload, title="Hybrid")

    assert result["added_text_chunks"] > 0
    assert result["added_figure_chunks"] == 1
    assert result["added_chunks"] == result["added_text_chunks"] + result["added_figure_chunks"]


def test_pipeline_question_prefers_figure_citation(workdir):
    def parser(_: bytes):
        return ParsedPdfContent(
            pages=[(1, "This paper introduces a method for image classification.")],
            figures=[
                ExtractedFigure(
                    page=1,
                    figure_id="p1_f1",
                    image_bytes=b"fake-image",
                    mime_type="image/png",
                    caption="Figure 1: End-to-end pipeline.",
                )
            ],
        )

    service = make_service(workdir, parser, figure_analyzer=FakeFigureAnalyzer())
    upload = BytesIO(b"pipeline-doc")
    upload.name = "pipeline.pdf"
    service.ingest_document(upload, title="Pipeline")

    result = service.answer_question("pipeline", scope_mode="global", top_k=1)

    assert result["citations"]
    assert result["citations"][0]["doc_type"] == "figure"
    assert result["citations"][0]["figure_id"] == "p1_f1"


def test_figure_backfill_is_idempotent(workdir):
    def text_parser(_: bytes):
        return [(1, "text only baseline")]

    service = make_service(workdir, text_parser, figure_analyzer=FakeFigureAnalyzer())
    upload = BytesIO(b"legacy")
    upload.name = "legacy.pdf"
    paper_id = service.ingest_document(upload, title="Legacy")["paper_id"]

    def parser_with_figure(_: bytes):
        return ParsedPdfContent(
            pages=[(1, "text only baseline")],
            figures=[
                ExtractedFigure(
                    page=1,
                    figure_id="p1_f1",
                    image_bytes=b"fake-image",
                    mime_type="image/png",
                    caption="Figure 1: Legacy pipeline.",
                )
            ],
        )

    service.pdf_parser = parser_with_figure
    first = service.backfill_figure_chunks()
    second = service.backfill_figure_chunks()

    paper = service.catalog.get_paper(paper_id)
    assert first["added_figure_chunks"] == 1
    assert second["added_figure_chunks"] == 0
    assert paper is not None
    assert paper["figure_chunk_count"] == 1


def test_delete_paper_removes_catalog_vectors_and_file(workdir):
    def parser(payload: bytes):
        return [(1, payload.decode("utf-8"))]

    service = make_service(workdir, parser)

    doc_a = BytesIO("alpha method details".encode("utf-8"))
    doc_a.name = "alpha.pdf"
    doc_b = BytesIO("beta experiment results".encode("utf-8"))
    doc_b.name = "beta.pdf"

    paper_a = service.ingest_document(doc_a, title="Alpha")["paper_id"]
    paper_b = service.ingest_document(doc_b, title="Beta")["paper_id"]

    paper_a_record = service.catalog.get_paper(paper_a)
    assert paper_a_record is not None
    source_path = Path(paper_a_record["source_path"])
    assert source_path.exists()

    result = service.delete_paper(paper_a)

    assert result["deleted"] is True
    assert result["paper_id"] == paper_a
    assert result["deleted_chunks"] > 0
    assert result["deleted_file"] is True
    assert service.catalog.get_paper(paper_a) is None
    assert service.catalog.get_paper(paper_b) is not None
    assert not source_path.exists() or source_path.stat().st_size == 0
    assert all(doc.metadata.get("paper_id") != paper_a for doc in service.vector_index.docs)


def test_delete_missing_paper_is_safe(workdir):
    def parser(_: bytes):
        return [(1, "baseline text")]

    service = make_service(workdir, parser)
    upload = BytesIO(b"baseline-file")
    upload.name = "baseline.pdf"
    service.ingest_document(upload, title="Baseline")

    before_count = len(service.list_papers())
    result = service.delete_paper("paper_not_exists")

    assert result["deleted"] is False
    assert result["deleted_chunks"] == 0
    assert result["deleted_file"] is False
    assert "未找到" in result["message"]
    assert len(service.list_papers()) == before_count


def test_clear_library_removes_all_papers_vectors_and_files(workdir):
    def parser(payload: bytes):
        return [(1, payload.decode("utf-8"))]

    service = make_service(workdir, parser)

    doc_a = BytesIO("alpha method details".encode("utf-8"))
    doc_a.name = "alpha.pdf"
    doc_b = BytesIO("beta experiment results".encode("utf-8"))
    doc_b.name = "beta.pdf"

    paper_a = service.ingest_document(doc_a, title="Alpha")["paper_id"]
    paper_b = service.ingest_document(doc_b, title="Beta")["paper_id"]
    source_paths = []
    for paper_id in [paper_a, paper_b]:
        record = service.catalog.get_paper(paper_id)
        assert record is not None
        source_paths.append(Path(record["source_path"]))

    result = service.clear_library()

    assert result["cleared_papers"] == 2
    assert result["deleted_chunks"] > 0
    assert result["deleted_files"] == 2
    assert service.list_papers() == []
    assert service.vector_index.docs == []
    for path in source_paths:
        assert not path.exists() or path.stat().st_size == 0
