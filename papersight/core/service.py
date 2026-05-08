import hashlib
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

from papersight.core.catalog import CatalogStore
from papersight.core.pdf_loader import (
    ExtractedFigure,
    ParsedPdfContent,
    extract_pdf_content,
    normalize_text,
)
from papersight.core.prompts import build_qa_prompt, build_report_prompt
from papersight.core.settings import PaperSightSettings
from papersight.core.vector_index import FaissVectorIndex
from papersight.core.vision import FigureAnalysis, OllamaFigureAnalyzer


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
    ZH_STOPWORDS = {
        "什么",
        "如何",
        "怎么",
        "为何",
        "为什么",
        "请问",
        "介绍",
        "一个",
        "一下",
        "是否",
        "可以",
    }
    PIPELINE_HINTS = {
        "pipeline",
        "workflow",
        "framework",
        "architecture",
        "diagram",
        "flowchart",
        "流程",
        "步骤",
        "框架",
        "管线",
        "架构图",
    }

    def __init__(
        self,
        settings: PaperSightSettings | None = None,
        vector_index: Any | None = None,
        llm: Any | None = None,
        pdf_parser: Callable[[bytes], list[tuple[int, str]] | ParsedPdfContent] | None = None,
        figure_analyzer: Any | None = None,
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
        self.pdf_parser = pdf_parser or self._default_pdf_parser

        self.figure_analyzer = figure_analyzer
        if self.figure_analyzer is None and self.settings.enable_figure_parsing:
            self.figure_analyzer = OllamaFigureAnalyzer(self.settings.vision_model)

        self.splitter = splitter or RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        self._backfill_marker = self.data_dir / ".figure_backfill.done"
        self._backfill_started = False
        self._backfill_lock = threading.Lock()
        self._start_background_backfill()

    def list_papers(self) -> list[dict[str, Any]]:
        return self.catalog.list_papers()

    def delete_paper(self, paper_id: str | None) -> dict[str, Any]:
        normalized_paper_id = (paper_id or "").strip()
        if not normalized_paper_id:
            return {
                "deleted": False,
                "paper_id": None,
                "deleted_chunks": 0,
                "deleted_file": False,
                "message": "论文 ID 不能为空。",
            }

        paper = self.catalog.get_paper(normalized_paper_id)
        if paper is None:
            return {
                "deleted": False,
                "paper_id": normalized_paper_id,
                "deleted_chunks": 0,
                "deleted_file": False,
                "message": "未找到该论文。",
            }

        deleted_chunks = self.vector_index.remove_paper(normalized_paper_id)
        self.catalog.remove_paper(normalized_paper_id)
        deleted_file = self._remove_source_file(str(paper.get("source_path") or ""))
        title = str(paper.get("title") or normalized_paper_id)

        return {
            "deleted": True,
            "paper_id": normalized_paper_id,
            "deleted_chunks": deleted_chunks,
            "deleted_file": deleted_file,
            "message": f"已删除论文：{title}（chunk={deleted_chunks}, file={'yes' if deleted_file else 'no'}）",
        }

    def clear_library(self) -> dict[str, Any]:
        papers = self.catalog.list_papers()
        deleted_files = 0
        for paper in papers:
            source_path = str(paper.get("source_path") or "")
            if self._remove_source_file(source_path):
                deleted_files += 1

        deleted_chunks = self.vector_index.clear()
        cleared_papers = self.catalog.clear()
        return {
            "cleared_papers": cleared_papers,
            "deleted_chunks": deleted_chunks,
            "deleted_files": deleted_files,
            "message": f"已清空知识库（papers={cleared_papers}, chunks={deleted_chunks}, files={deleted_files}）。",
        }

    def ingest_document(self, file: Any, title: str | None = None) -> dict[str, Any]:
        raw_bytes, original_name = self._read_upload(file)
        if not raw_bytes:
            return {
                "paper_id": None,
                "added_chunks": 0,
                "added_text_chunks": 0,
                "added_figure_chunks": 0,
                "deduplicated": False,
                "message": "上传文件为空。",
            }

        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        existing = self.catalog.get_paper_by_hash(content_hash)
        if existing:
            return {
                "paper_id": existing["paper_id"],
                "added_chunks": 0,
                "added_text_chunks": 0,
                "added_figure_chunks": 0,
                "deduplicated": True,
                "message": f"文档已存在：{existing['title']}",
            }

        parsed = self._normalize_pdf_parse_result(self.pdf_parser(raw_bytes))
        pages = parsed.pages
        figures = parsed.figures if self.settings.enable_figure_parsing else []

        has_text = any(text.strip() for _, text in pages)
        if not has_text and not figures:
            return {
                "paper_id": None,
                "added_chunks": 0,
                "added_text_chunks": 0,
                "added_figure_chunks": 0,
                "deduplicated": False,
                "message": "文本与图像均未提取到有效内容。",
            }

        paper_id = f"paper_{uuid.uuid4().hex[:10]}"
        safe_title = (title or Path(original_name).stem or paper_id).strip()
        source_path = self.upload_dir / f"{paper_id}.pdf"
        source_path.write_bytes(raw_bytes)

        text_documents = self._chunk_pages(
            pages=pages,
            paper_id=paper_id,
            title=safe_title,
            source_path=str(source_path),
            content_hash=content_hash,
        )

        figure_documents: list[Document] = []
        if figures and self.figure_analyzer is not None:
            figure_documents = self._chunk_figures(
                figures=figures,
                paper_id=paper_id,
                title=safe_title,
                source_path=str(source_path),
                content_hash=content_hash,
            )

        documents = text_documents + figure_documents
        if not documents:
            return {
                "paper_id": None,
                "added_chunks": 0,
                "added_text_chunks": 0,
                "added_figure_chunks": 0,
                "deduplicated": False,
                "message": "文档分块失败：未得到可用内容。",
            }

        self.vector_index.add_documents(documents)

        paper_record = {
            "paper_id": paper_id,
            "title": safe_title,
            "source_path": str(source_path),
            "content_hash": content_hash,
            "page_count": len(pages),
            "chunk_count": len(documents),
            "text_chunk_count": len(text_documents),
            "figure_chunk_count": len(figure_documents),
            "figure_count": len(figures),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self.catalog.add_paper(paper_record, content_hash)
        return {
            "paper_id": paper_id,
            "added_chunks": len(documents),
            "added_text_chunks": len(text_documents),
            "added_figure_chunks": len(figure_documents),
            "deduplicated": False,
            "message": f"入库成功：{safe_title}（text={len(text_documents)}, figure={len(figure_documents)}）",
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

        prefers_pipeline = self._is_pipeline_question(normalized_question)
        candidate_k = max(top_k, self.settings.retrieval_candidate_k)
        fetch_k = max(candidate_k, self.settings.retrieval_fetch_k)

        semantic_results = self.vector_index.similarity_search(
            query=normalized_question,
            k=candidate_k,
            metadata_filter=metadata_filter,
            fetch_k=fetch_k,
        )

        if prefers_pipeline:
            figure_query = f"{normalized_question} pipeline workflow diagram"
            semantic_results += self.vector_index.similarity_search(
                query=figure_query,
                k=candidate_k,
                metadata_filter=metadata_filter,
                fetch_k=fetch_k,
            )

        keyword_results = self._keyword_search_candidates(
            question=normalized_question,
            metadata_filter=metadata_filter,
            limit=candidate_k,
        )
        merged_results = self._merge_results(
            semantic_results,
            keyword_results,
            max_items=max(candidate_k * 2, 40),
        )

        selected = self._rerank_results(
            normalized_question,
            merged_results,
            top_k,
            prefer_pipeline=prefers_pipeline,
        )
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

        report_query = "research question method experiment setup results limitations reproducibility pipeline"
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

    def backfill_figure_chunks(self) -> dict[str, int]:
        if not self.settings.enable_figure_parsing or self.figure_analyzer is None:
            return {"papers_scanned": 0, "papers_updated": 0, "added_figure_chunks": 0}

        stats = {"papers_scanned": 0, "papers_updated": 0, "added_figure_chunks": 0}
        papers = self.catalog.list_papers()

        for paper in papers:
            stats["papers_scanned"] += 1
            paper_id = str(paper.get("paper_id") or "")
            source_path_str = str(paper.get("source_path") or "")
            if not paper_id or not source_path_str:
                continue

            source_path = Path(source_path_str)
            if not source_path.exists():
                continue

            existing_figure_chunks = int(paper.get("figure_chunk_count", 0) or 0)
            if existing_figure_chunks > 0:
                continue

            try:
                raw_bytes = source_path.read_bytes()
                parsed = self._normalize_pdf_parse_result(self.pdf_parser(raw_bytes))
            except Exception:
                continue

            existing_chunk_ids = self._existing_chunk_ids_for_paper(paper_id)
            figure_docs = self._chunk_figures(
                figures=parsed.figures,
                paper_id=paper_id,
                title=str(paper.get("title") or paper_id),
                source_path=source_path_str,
                content_hash=str(paper.get("content_hash") or ""),
                existing_chunk_ids=existing_chunk_ids,
            )
            if figure_docs:
                self.vector_index.add_documents(figure_docs)

            text_count = int(paper.get("text_chunk_count", paper.get("chunk_count", 0)) or 0)
            self.catalog.update_paper(
                paper_id,
                {
                    "text_chunk_count": text_count,
                    "figure_chunk_count": len(figure_docs),
                    "figure_count": len(parsed.figures),
                    "chunk_count": text_count + len(figure_docs),
                    "figure_backfilled_at": datetime.now(timezone.utc).isoformat(),
                },
            )
            stats["papers_updated"] += 1
            stats["added_figure_chunks"] += len(figure_docs)

        return stats

    def _start_background_backfill(self) -> None:
        if not self.settings.enable_figure_parsing or self.figure_analyzer is None:
            return
        if self._backfill_marker.exists():
            return

        with self._backfill_lock:
            if self._backfill_started:
                return
            self._backfill_started = True

        worker = threading.Thread(target=self._run_background_backfill, daemon=True)
        worker.start()

    def _run_background_backfill(self) -> None:
        try:
            self.backfill_figure_chunks()
            timestamp = datetime.now(timezone.utc).isoformat()
            self._backfill_marker.write_text(timestamp, encoding="utf-8")
        except Exception:
            pass

    def _default_pdf_parser(self, pdf_bytes: bytes) -> ParsedPdfContent:
        return extract_pdf_content(pdf_bytes, max_figures_per_page=self.settings.max_figures_per_page)

    def _normalize_pdf_parse_result(self, parsed: list[tuple[int, str]] | ParsedPdfContent) -> ParsedPdfContent:
        if isinstance(parsed, ParsedPdfContent):
            return parsed
        if isinstance(parsed, list):
            return ParsedPdfContent(pages=parsed, figures=[])
        raise TypeError("Unsupported parse result type.")

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
                        "doc_type": "text",
                    }
                ],
            )

            for doc in page_docs:
                doc.metadata["chunk_id"] = f"{paper_id}_t{chunk_seq}"
                docs.append(doc)
                chunk_seq += 1

        return docs

    def _chunk_figures(
        self,
        figures: list[ExtractedFigure],
        paper_id: str,
        title: str,
        source_path: str,
        content_hash: str,
        existing_chunk_ids: set[str] | None = None,
    ) -> list[Document]:
        docs: list[Document] = []
        skip_ids = existing_chunk_ids or set()

        for figure in figures:
            chunk_id = f"{paper_id}_{figure.figure_id}"
            if chunk_id in skip_ids:
                continue

            analysis = self._analyze_figure(figure)
            text = self._build_figure_chunk_text(figure, analysis)
            metadata = {
                "paper_id": paper_id,
                "title": title,
                "source_path": source_path,
                "page": figure.page,
                "content_hash": content_hash,
                "chunk_id": chunk_id,
                "doc_type": "figure",
                "figure_id": figure.figure_id,
                "figure_type": analysis.figure_type,
                "is_pipeline": analysis.is_pipeline,
                "confidence": analysis.confidence,
                "caption": figure.caption,
            }
            docs.append(Document(page_content=text, metadata=metadata))

        return docs

    def _analyze_figure(self, figure: ExtractedFigure) -> FigureAnalysis:
        if self.figure_analyzer is None:
            return FigureAnalysis(
                figure_type="other",
                is_pipeline=False,
                confidence=0.0,
                steps=[],
                components=[],
                relations=[],
                summary=figure.caption or "图像解析不可用。",
            )

        result = self.figure_analyzer.analyze(
            image_bytes=figure.image_bytes,
            mime_type=figure.mime_type,
            caption=figure.caption,
        )

        if isinstance(result, FigureAnalysis):
            return result

        if isinstance(result, dict):
            return FigureAnalysis(
                figure_type=str(result.get("figure_type", "other")),
                is_pipeline=bool(result.get("is_pipeline", False)),
                confidence=self._safe_confidence(result.get("confidence", 0.0)),
                steps=self._normalize_string_list(result.get("steps")),
                components=self._normalize_string_list(result.get("components")),
                relations=self._normalize_string_list(result.get("relations")),
                summary=str(result.get("summary", figure.caption or "图像解析返回为空。")),
            )

        return FigureAnalysis(
            figure_type="other",
            is_pipeline=False,
            confidence=0.0,
            steps=[],
            components=[],
            relations=[],
            summary=figure.caption or "图像解析返回不可识别。",
        )

    def _build_figure_chunk_text(self, figure: ExtractedFigure, analysis: FigureAnalysis) -> str:
        lines = [
            "Figure Analysis",
            f"Figure ID: {figure.figure_id}",
            f"Type: {analysis.figure_type}",
            f"Pipeline: {analysis.is_pipeline}",
            f"Confidence: {analysis.confidence:.2f}",
            f"Summary: {analysis.summary}",
        ]

        if figure.caption:
            lines.append(f"Caption: {figure.caption}")
        if analysis.steps:
            lines.append("Steps: " + " | ".join(analysis.steps))
        if analysis.components:
            lines.append("Components: " + " | ".join(analysis.components))
        if analysis.relations:
            lines.append("Relations: " + " | ".join(analysis.relations))

        return "\n".join(lines)

    def _build_context(self, results: list[tuple[Document, float]]) -> tuple[str, list[dict[str, Any]]]:
        context_lines: list[str] = []
        citations: list[dict[str, Any]] = []

        for idx, (doc, score) in enumerate(results, start=1):
            snippet = doc.page_content.strip()
            metadata = doc.metadata
            label = f"C{idx}"
            context_lines.append(
                f"[{label}] doc_type={metadata.get('doc_type', 'text')} "
                f"paper_id={metadata.get('paper_id')} page={metadata.get('page')} "
                f"chunk_id={metadata.get('chunk_id')} figure_id={metadata.get('figure_id')} "
                f"confidence={metadata.get('confidence')}\n{snippet}"
            )
            citations.append(
                {
                    "label": label,
                    "paper_id": metadata.get("paper_id"),
                    "title": metadata.get("title"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "doc_type": metadata.get("doc_type", "text"),
                    "figure_id": metadata.get("figure_id"),
                    "confidence": metadata.get("confidence"),
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
        prefer_pipeline: bool,
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
            metadata = doc.metadata

            bonus = 0.0
            if question_lower and len(question_lower) <= 64 and question_lower in text_lower:
                bonus += 0.40

            matched_terms = sum(1 for term in terms if term in text_lower)
            bonus += min(0.07 * matched_terms, 0.35)

            if prefer_pipeline:
                if metadata.get("doc_type") == "figure":
                    bonus += 0.25
                if metadata.get("is_pipeline") is True:
                    bonus += self.settings.pipeline_figure_boost

            penalty = 0.0
            if not wants_reference and metadata.get("doc_type") != "figure" and self._looks_like_reference_chunk(text):
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

            metadata = doc.metadata
            page = metadata.get("page")
            page_number = int(page) if isinstance(page, int) else 999
            reference_penalty = 0.15 if self._looks_like_reference_chunk(text) else 0.0

            boost = 0.0
            if metadata.get("doc_type") == "figure":
                boost += 0.12
            if metadata.get("is_pipeline") is True and self._is_pipeline_question(question):
                boost += self.settings.pipeline_figure_boost

            synthetic_score = 0.95 - min(hit_count, 6) * 0.08 + min(page_number, 60) * 0.001 + reference_penalty - boost
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

    def _is_pipeline_question(self, question: str) -> bool:
        lowered = question.lower()
        for token in self.PIPELINE_HINTS:
            if token in lowered or token in question:
                return True
        return False

    def _safe_confidence(self, value: Any) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, number))

    def _normalize_string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized[:24]

    def _existing_chunk_ids_for_paper(self, paper_id: str) -> set[str]:
        vectorstore = getattr(self.vector_index, "vectorstore", None)
        if vectorstore is None:
            return set()

        chunk_ids: set[str] = set()
        for doc in vectorstore.docstore._dict.values():
            metadata = getattr(doc, "metadata", {})
            if metadata.get("paper_id") != paper_id:
                continue
            chunk_id = metadata.get("chunk_id")
            if isinstance(chunk_id, str) and chunk_id:
                chunk_ids.add(chunk_id)
        return chunk_ids

    def _remove_source_file(self, source_path: str) -> bool:
        if not source_path:
            return False

        path = Path(source_path)
        if not path.exists() or not path.is_file():
            return False

        try:
            path.unlink()
            return True
        except OSError:
            # In restricted environments unlink may fail; fall back to truncation.
            try:
                path.write_bytes(b"")
            except OSError:
                return False
        return True
