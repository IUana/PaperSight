import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from papersight.core.catalog import CatalogStore
from papersight.core.pdf_loader import extract_pages_from_pdf, normalize_text
from papersight.core.prompts import build_qa_prompt, build_report_prompt
from papersight.core.settings import PaperSightSettings
from papersight.core.vector_index import FaissVectorIndex


class PaperSightService:
    def __init__(
        self,
        settings: PaperSightSettings | None = None,
        vector_index: Any | None = None,
        llm: Any | None = None,
        pdf_parser: Callable[[bytes], list[tuple[int, str]]] | None = None,
        splitter: RecursiveCharacterTextSplitter | None = None,
    ) -> None:
        self.settings = settings or PaperSightSettings()
        self.data_dir = Path(self.settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir = self.data_dir / self.settings.upload_dir_name
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        catalog_path = self.data_dir / self.settings.catalog_filename
        self.catalog = CatalogStore(catalog_path)

        index_dir = self.data_dir / self.settings.index_dir_name
        self.vector_index = vector_index or FaissVectorIndex(index_dir, self.settings.embedding_model)
        self.llm = llm or ChatOllama(model=self.settings.chat_model, temperature=0)
        self.pdf_parser = pdf_parser or extract_pages_from_pdf
        self.splitter = splitter or RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

    def list_papers(self) -> list[dict[str, Any]]:
        return self.catalog.list_papers()

    def ingest_document(self, file: Any, title: str | None = None) -> dict[str, Any]:
        raw_bytes, original_name = self._read_upload(file)
        if not raw_bytes:
            return {"paper_id": None, "added_chunks": 0, "deduplicated": False, "message": "上传文件为空。"}

        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        existing = self.catalog.get_paper_by_hash(content_hash)
        if existing:
            return {
                "paper_id": existing["paper_id"],
                "added_chunks": 0,
                "deduplicated": True,
                "message": f"文档已存在：{existing['title']}",
            }

        pages = self.pdf_parser(raw_bytes)
        if not any(text.strip() for _, text in pages):
            return {
                "paper_id": None,
                "added_chunks": 0,
                "deduplicated": False,
                "message": "文本提取不足：该PDF可能是扫描件或无可解析文本。",
            }

        paper_id = f"paper_{uuid.uuid4().hex[:10]}"
        safe_title = (title or Path(original_name).stem or paper_id).strip()
        source_path = self.upload_dir / f"{paper_id}.pdf"
        source_path.write_bytes(raw_bytes)

        documents = self._chunk_pages(
            pages=pages,
            paper_id=paper_id,
            title=safe_title,
            source_path=str(source_path),
            content_hash=content_hash,
        )
        if not documents:
            return {
                "paper_id": None,
                "added_chunks": 0,
                "deduplicated": False,
                "message": "文档分块失败：未得到可用文本块。",
            }

        self.vector_index.add_documents(documents)

        paper_record = {
            "paper_id": paper_id,
            "title": safe_title,
            "source_path": str(source_path),
            "content_hash": content_hash,
            "page_count": len(pages),
            "chunk_count": len(documents),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.catalog.add_paper(paper_record, content_hash)
        return {
            "paper_id": paper_id,
            "added_chunks": len(documents),
            "deduplicated": False,
            "message": f"入库成功：{safe_title}（{len(documents)} chunks）",
        }

    def answer_question(
        self,
        question: str,
        scope_mode: str,
        paper_id: str | None = None,
        top_k: int = 5,
    ) -> dict[str, Any]:
        normalized_question = question.strip()
        if not normalized_question:
            return {"answer": "请输入问题。", "citations": [], "used_chunks": 0}
        if not self.vector_index.has_documents():
            return {"answer": self.settings.fallback_answer, "citations": [], "used_chunks": 0}

        metadata_filter = None
        if scope_mode == "paper" and paper_id:
            metadata_filter = {"paper_id": paper_id}

        retrieve_k = max(top_k, 20) if metadata_filter else top_k
        results = self.vector_index.similarity_search(
            query=normalized_question,
            k=retrieve_k,
            metadata_filter=metadata_filter,
        )

        selected = results[:top_k]
        if not selected:
            return {"answer": self.settings.fallback_answer, "citations": [], "used_chunks": 0}

        context, citations = self._build_context(selected)
        prompt = build_qa_prompt(context=context, question=normalized_question, fallback=self.settings.fallback_answer)
        answer = self._invoke_llm(prompt).strip() or self.settings.fallback_answer
        return {"answer": answer, "citations": citations, "used_chunks": len(citations)}

    def generate_report(self, paper_id: str) -> dict[str, Any]:
        paper = self.catalog.get_paper(paper_id)
        if not paper:
            return {
                "report_markdown": "未找到该论文，请先上传并入库。",
                "citations": [],
                "summary_meta": {"paper_id": paper_id, "used_chunks": 0},
            }

        report_query = "research question method experiment setup results limitations reproducibility"
        results = self.vector_index.similarity_search(
            query=report_query,
            k=12,
            metadata_filter={"paper_id": paper_id},
        )
        selected = results[:8]
        if not selected:
            return {
                "report_markdown": self.settings.fallback_answer,
                "citations": [],
                "summary_meta": {"paper_id": paper_id, "title": paper["title"], "used_chunks": 0},
            }

        context, citations = self._build_context(selected)
        prompt = build_report_prompt(context=context, fallback=self.settings.fallback_answer)
        report = self._invoke_llm(prompt).strip() or self.settings.fallback_answer
        return {
            "report_markdown": report,
            "citations": citations,
            "summary_meta": {"paper_id": paper_id, "title": paper["title"], "used_chunks": len(citations)},
        }

    def _read_upload(self, file: Any) -> tuple[bytes, str]:
        if isinstance(file, (bytes, bytearray)):
            return bytes(file), "uploaded.pdf"

        if hasattr(file, "read"):
            raw = file.read()
            if hasattr(file, "seek"):
                file.seek(0)
            name = getattr(file, "name", "uploaded.pdf")
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
            return raw, name

        raise TypeError("Unsupported file type for ingest_document.")

    def _chunk_pages(
        self,
        pages: list[tuple[int, str]],
        paper_id: str,
        title: str,
        source_path: str,
        content_hash: str,
    ) -> list[Document]:
        docs: list[Document] = []
        chunk_seq = 1

        for page_no, text in pages:
            cleaned = normalize_text(text)
            if not cleaned:
                continue

            page_docs = self.splitter.create_documents(
                [cleaned],
                metadatas=[
                    {
                        "paper_id": paper_id,
                        "title": title,
                        "source_path": source_path,
                        "page": page_no,
                        "content_hash": content_hash,
                    }
                ],
            )

            for doc in page_docs:
                doc.metadata["chunk_id"] = f"{paper_id}_c{chunk_seq}"
                docs.append(doc)
                chunk_seq += 1

        return docs

    def _build_context(self, results: list[tuple[Document, float]]) -> tuple[str, list[dict[str, Any]]]:
        context_lines: list[str] = []
        citations: list[dict[str, Any]] = []

        for idx, (doc, score) in enumerate(results, start=1):
            snippet = doc.page_content.strip()
            metadata = doc.metadata
            label = f"C{idx}"
            context_lines.append(
                f"[{label}] paper_id={metadata.get('paper_id')} page={metadata.get('page')} "
                f"chunk_id={metadata.get('chunk_id')}\n{snippet}"
            )
            citations.append(
                {
                    "label": label,
                    "paper_id": metadata.get("paper_id"),
                    "title": metadata.get("title"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "snippet": snippet[:500],
                    "score": score,
                }
            )

        return "\n\n".join(context_lines), citations

    def _invoke_llm(self, prompt: str) -> str:
        result = self.llm.invoke(prompt)
        if isinstance(result, str):
            return result

        content = getattr(result, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            collected: list[str] = []
            for item in content:
                if isinstance(item, str):
                    collected.append(item)
                elif isinstance(item, dict) and "text" in item:
                    collected.append(str(item["text"]))
            return "\n".join(collected)
        return str(result)
