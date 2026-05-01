from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


class FaissVectorIndex:
    def __init__(self, index_dir: Path, embedding_model: str) -> None:
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.vectorstore: FAISS | None = None
        self._load_if_exists()

    def _index_exists(self) -> bool:
        return (self.index_dir / "index.faiss").exists() and (self.index_dir / "index.pkl").exists()

    def _load_if_exists(self) -> None:
        if not self._index_exists():
            return
        self.vectorstore = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

    def has_documents(self) -> bool:
        if self.vectorstore is None:
            return False
        return self.vectorstore.index.ntotal > 0

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        else:
            self.vectorstore.add_documents(documents)
        self.save()

    def similarity_search(
        self,
        query: str,
        k: int,
        metadata_filter: dict[str, str] | None = None,
        fetch_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        if self.vectorstore is None:
            return []
        kwargs: dict[str, object] = {"query": query, "k": k, "filter": metadata_filter}
        if fetch_k is not None:
            kwargs["fetch_k"] = fetch_k
        return self.vectorstore.similarity_search_with_score(**kwargs)

    def save(self) -> None:
        if self.vectorstore is None:
            return
        self.vectorstore.save_local(str(self.index_dir))
