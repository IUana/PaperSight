import json
from pathlib import Path
from typing import Any


class CatalogStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"papers": {}, "hash_to_paper_id": {}})

    def _read(self) -> dict[str, Any]:
        with self.path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _write(self, payload: dict[str, Any]) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def list_papers(self) -> list[dict[str, Any]]:
        data = self._read()
        papers = list(data["papers"].values())
        papers.sort(key=lambda item: item.get("created_at", ""), reverse=True)
        return papers

    def get_paper(self, paper_id: str) -> dict[str, Any] | None:
        data = self._read()
        return data["papers"].get(paper_id)

    def get_paper_by_hash(self, content_hash: str) -> dict[str, Any] | None:
        data = self._read()
        paper_id = data["hash_to_paper_id"].get(content_hash)
        if not paper_id:
            return None
        return data["papers"].get(paper_id)

    def add_paper(self, paper: dict[str, Any], content_hash: str) -> None:
        data = self._read()
        paper_id = paper["paper_id"]
        data["papers"][paper_id] = paper
        data["hash_to_paper_id"][content_hash] = paper_id
        self._write(data)

    def update_paper(self, paper_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
        data = self._read()
        paper = data["papers"].get(paper_id)
        if paper is None:
            return None

        paper.update(updates)
        data["papers"][paper_id] = paper
        self._write(data)
        return paper

    def remove_paper(self, paper_id: str) -> dict[str, Any] | None:
        data = self._read()
        paper = data["papers"].pop(paper_id, None)
        if paper is None:
            return None

        hash_to_paper_id = data["hash_to_paper_id"]
        removable_hashes = [content_hash for content_hash, mapped_id in hash_to_paper_id.items() if mapped_id == paper_id]
        for content_hash in removable_hashes:
            hash_to_paper_id.pop(content_hash, None)

        self._write(data)
        return paper

    def clear(self) -> int:
        data = self._read()
        removed_count = len(data["papers"])
        data["papers"] = {}
        data["hash_to_paper_id"] = {}
        self._write(data)
        return removed_count
