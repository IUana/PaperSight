from io import BytesIO
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import pytest
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    def similarity_search(self, query: str, k: int, metadata_filter: dict | None = None):
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


def make_service(tmp_path, parser):
    settings = PaperSightSettings(data_dir=tmp_path, chunk_size=30, chunk_overlap=10)
    splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=10, separators=[""])
    return PaperSightService(
        settings=settings,
        vector_index=FakeVectorIndex(),
        llm=FakeLLM("测试回答 [C1]"),
        pdf_parser=parser,
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
