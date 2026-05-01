from dataclasses import dataclass
from pathlib import Path

FALLBACK_ANSWER = "无法从给定上下文确定答案。"


@dataclass(frozen=True)
class PaperSightSettings:
    data_dir: Path = Path("data")
    index_dir_name: str = "faiss_index"
    catalog_filename: str = "catalog.json"
    upload_dir_name: str = "uploads"
    chat_model: str = "gemma4:e4b"
    vision_model: str = "qwen3-vl:4b"
    embedding_model: str = "nomic-embed-text"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    default_top_k: int = 5
    retrieval_candidate_k: int = 30
    retrieval_fetch_k: int = 200
    enable_figure_parsing: bool = True
    max_figures_per_page: int = 4
    pipeline_figure_boost: float = 0.35
    fallback_answer: str = FALLBACK_ANSWER
