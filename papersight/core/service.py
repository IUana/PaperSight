import hashlib
import re
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
    EN_STOPWORDS = {
        "what",
        "which",
        "when",
        "where",
        "why",
        "how",
        "who",
        "whom",
        "whose",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "a",
        "an",
        "the",
        "of",
        "to",
        "for",
        "in",
        "on",
        "at",
        "and",
        "or",
    }
    ZH_STOPWORDS = {"什么", "如何", "怎么", "为何", "为什么", "请问", "介绍", "一下", "一下子", "是否", "可以"}

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

        candidate_k = max(top_k, self.settings.retrieval_candidate_k)
        fetch_k = max(candidate_k, self.settings.retrieval_fetch_k)
        results = self.vector_index.similarity_search(
            query=normalized_question,
            k=candidate_k,
            metadata_filter=metadata_filter,
            fetch_k=fetch_k,
        )
        keyword_results = self._keyword_search_candidates(
            question=normalized_question,
            metadata_filter=metadata_filter,
            limit=candidate_k,
        )
        merged_results = self._merge_results(results, keyword_results, max_items=max(candidate_k * 2, 40))

        selected = self._rerank_results(normalized_question, merged_results, top_k)
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

    def _rerank_results(
        self,
        question: str,
        results: list[tuple[Document, float]],
        top_k: int,
    ) -> list[tuple[Document, float]]:
        if not results:
            return []

        question_lower = question.lower()
        terms = self._extract_query_terms(question)
        wants_reference = any(
            token in question_lower
            for token in ["reference", "references", "citation", "citations", "文献", "引用", "参考"]
        )

        reranked: list[tuple[float, Document, float]] = []
        for doc, base_score in results:
            text = doc.page_content or ""
            text_lower = text.lower()

            bonus = 0.0
            if question_lower and len(question_lower) <= 64 and question_lower in text_lower:
                bonus += 0.40

            matched_terms = sum(1 for term in terms if term in text_lower)
            bonus += min(0.07 * matched_terms, 0.35)

            penalty = 0.0
            if not wants_reference and self._looks_like_reference_chunk(text):
                penalty += 0.25

            rank_score = base_score - bonus + penalty
            reranked.append((rank_score, doc, base_score))

        reranked.sort(key=lambda item: item[0])
        return [(doc, base_score) for _, doc, base_score in reranked[:top_k]]

    def _extract_query_terms(self, question: str) -> list[str]:
        terms: list[str] = []
        seen: set[str] = set()

        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]{2,}", question.lower()):
            if token in self.EN_STOPWORDS:
                continue
            if token not in seen:
                seen.add(token)
                terms.append(token)

        for token in re.findall(r"[\u4e00-\u9fff]{2,}", question):
            if token in self.ZH_STOPWORDS:
                continue
            if token not in seen:
                seen.add(token)
                terms.append(token)

        return terms[:16]

    def _looks_like_reference_chunk(self, text: str) -> bool:
        lower = text.lower()
        clues = [
            "references",
            "bibliography",
            "doi",
            "arxiv",
            "proceedings",
            "conference on",
            "et al.",
            "pp.",
            "vol.",
            "isbn",
        ]
        clue_hits = sum(1 for clue in clues if clue in lower)
        bracket_hits = len(re.findall(r"\[[A-Za-z0-9+]{2,}\]", text))
        return clue_hits >= 2 or bracket_hits >= 3

    def _keyword_search_candidates(
        self,
        question: str,
        metadata_filter: dict[str, str] | None,
        limit: int,
    ) -> list[tuple[Document, float]]:
        vectorstore = getattr(self.vector_index, "vectorstore", None)
        if vectorstore is None:
            return []

        terms = self._extract_query_terms(question)
        if not terms:
            return []

        docs = list(vectorstore.docstore._dict.values())
        if metadata_filter:
            docs = [
                doc
                for doc in docs
                if all(doc.metadata.get(key) == value for key, value in metadata_filter.items())
            ]

        scored: list[tuple[float, Document]] = []
        for doc in docs:
            text = doc.page_content or ""
            lowered = text.lower()
            hit_count = sum(lowered.count(term) for term in terms)
            if hit_count == 0:
                continue

            page = doc.metadata.get("page")
            page_number = int(page) if isinstance(page, int) else 999
            reference_penalty = 0.15 if self._looks_like_reference_chunk(text) else 0.0
            synthetic_score = 0.95 - min(hit_count, 6) * 0.08 + min(page_number, 60) * 0.001 + reference_penalty
            scored.append((synthetic_score, doc))

        scored.sort(key=lambda item: item[0])
        return [(doc, score) for score, doc in scored[:limit]]

    def _merge_results(
        self,
        semantic_results: list[tuple[Document, float]],
        keyword_results: list[tuple[Document, float]],
        max_items: int,
    ) -> list[tuple[Document, float]]:
        merged: dict[str, tuple[Document, float]] = {}

        def make_key(doc: Document) -> str:
            chunk_id = doc.metadata.get("chunk_id")
            if isinstance(chunk_id, str):
                return chunk_id
            return str(id(doc))

        for doc, score in semantic_results + keyword_results:
            key = make_key(doc)
            existing = merged.get(key)
            if existing is None or score < existing[1]:
                merged[key] = (doc, score)

        merged_items = sorted(merged.values(), key=lambda item: item[1])
        return merged_items[:max_items]
