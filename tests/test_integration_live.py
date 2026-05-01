import os
from io import BytesIO
from pathlib import Path
from shutil import rmtree
from subprocess import DEVNULL, run
from uuid import uuid4

import pytest
from langchain_text_splitters import RecursiveCharacterTextSplitter

from papersight.core.service import PaperSightService
from papersight.core.settings import PaperSightSettings


class FakeLLM:
    def invoke(self, _: str) -> str:
        return "集成测试回答 [C1]"


def ollama_embedding_ready() -> bool:
    try:
        check = run(["ollama", "show", "nomic-embed-text"], stdout=DEVNULL, stderr=DEVNULL, check=False)
        return check.returncode == 0
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    os.getenv("PAPERSIGHT_RUN_LIVE") != "1" or not ollama_embedding_ready(),
    reason="需要显式设置 PAPERSIGHT_RUN_LIVE=1 且本地可用 nomic-embed-text。",
)


@pytest.fixture
def workdir():
    base = Path("data") / "test_runs"
    base.mkdir(parents=True, exist_ok=True)
    target = base / f"papersight-live-{uuid4().hex[:8]}"
    target.mkdir(parents=True, exist_ok=True)
    try:
        yield target
    finally:
        rmtree(target, ignore_errors=True)


def test_faiss_reload_and_query(workdir):
    settings = PaperSightSettings(data_dir=workdir, chunk_size=80, chunk_overlap=20)
    splitter = RecursiveCharacterTextSplitter(chunk_size=80, chunk_overlap=20)

    def parser(_: bytes):
        return [(1, "The method uses contrastive learning for representation alignment.")]

    service_first = PaperSightService(
        settings=settings,
        llm=FakeLLM(),
        pdf_parser=parser,
        splitter=splitter,
    )
    upload = BytesIO(b"live-faiss")
    upload.name = "live.pdf"
    service_first.ingest_document(upload, title="Live Paper")

    service_reloaded = PaperSightService(
        settings=settings,
        llm=FakeLLM(),
        pdf_parser=parser,
        splitter=splitter,
    )
    result = service_reloaded.answer_question("contrastive", scope_mode="global", top_k=1)

    assert result["citations"]
    assert result["used_chunks"] == 1
