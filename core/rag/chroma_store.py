# -*- coding: utf-8 -*-
"""
ChromaDB-backed RAG store using local Ollama embeddings.

The module imports chromadb lazily so normal evaluation can run without the
extra dependency unless --use-rag is enabled.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Any

logger = logging.getLogger(__name__)


class OllamaEmbeddingFunction:
    """Chroma embedding function that calls Ollama's local /api/embed endpoint."""

    def __init__(
        self,
        model: str = "bge-m3:latest",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def name(self) -> str:
        """Name required by recent ChromaDB embedding function interface."""
        safe_model = self.model.replace(":", "-").replace("/", "-")
        return f"ollama-{safe_model}"

    def default_space(self) -> str:
        """Default distance space for Chroma collections using this embedding."""
        return "cosine"

    def supported_spaces(self) -> list[str]:
        """Distance spaces supported by the embedding function."""
        return ["cosine", "l2", "ip"]

    def get_config(self) -> dict[str, Any]:
        """Serializable config used by ChromaDB when it inspects embedding functions."""
        return {
            "model": self.model,
            "base_url": self.base_url,
            "timeout": self.timeout,
        }

    @classmethod
    def build_from_config(cls, config: dict[str, Any]) -> "OllamaEmbeddingFunction":
        """Rebuild embedding function from ChromaDB config."""
        return cls(
            model=config.get("model", "bge-m3:latest"),
            base_url=config.get("base_url", "http://localhost:11434"),
            timeout=int(config.get("timeout", 120)),
        )

    def __call__(self, input: Iterable[str]) -> list[list[float]]:
        texts = list(input)
        if not texts:
            return []

        payload = json.dumps(
            {
                "model": self.model,
                "input": texts,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama embedding 调用失败，请确认 ollama 已启动且模型 {self.model!r} 已下载: {exc}"
            ) from exc

        data = json.loads(body)
        embeddings = data.get("embeddings")
        if embeddings is None and "embedding" in data:
            embeddings = [data["embedding"]]
        if not isinstance(embeddings, list):
            raise RuntimeError(f"Ollama embedding 返回格式异常: {body[:300]}")
        return embeddings


class ChromaRAGStore:
    """Small wrapper around a Chroma collection for review-rule retrieval."""

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str,
        embedding_model: str,
        ollama_base_url: str,
        seed_path: str | Path | None = None,
    ):
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "未安装 chromadb。请先执行 `pip install -r requirements.txt`，"
                "或单独安装 `pip install chromadb`。"
            ) from exc

        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.seed_path = Path(seed_path) if seed_path else None
        self.embedding_function = OllamaEmbeddingFunction(
            model=embedding_model,
            base_url=ollama_base_url,
        )

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

        if self.seed_path:
            self.seed_if_empty(self.seed_path)

    def seed_if_empty(self, seed_path: Path) -> None:
        """Load JSONL rule chunks into Chroma when the collection is empty."""
        if self.collection.count() > 0:
            return
        if not seed_path.exists():
            logger.warning("RAG seed 文件不存在，跳过初始化: %s", seed_path)
            return

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        with seed_path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                doc_id = str(item.get("id") or f"rule_{line_no}")
                text = str(item["text"]).strip()
                metadata = dict(item.get("metadata") or {})
                ids.append(doc_id)
                documents.append(text)
                metadatas.append(metadata)

        if not documents:
            logger.warning("RAG seed 文件为空: %s", seed_path)
            return

        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        logger.info("RAG 知识库初始化完成: %d 条规则写入 %s", len(documents), self.collection_name)

    def retrieve(self, query: str, dim: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Retrieve dimension-specific rule chunks."""
        if top_k <= 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where={"dimension": dim},
        )
        docs = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        items: list[dict[str, Any]] = []
        for doc, metadata, distance in zip(docs, metadatas, distances):
            items.append(
                {
                    "text": doc,
                    "metadata": metadata or {},
                    "distance": distance,
                }
            )
        return items

    @staticmethod
    def format_context(items: list[dict[str, Any]]) -> str:
        """Format retrieved chunks for prompt injection."""
        if not items:
            return ""

        blocks: list[str] = []
        for idx, item in enumerate(items, 1):
            metadata = item.get("metadata") or {}
            title = metadata.get("title", "审查规则")
            error_type = metadata.get("error_type", "")
            source = metadata.get("source", "local")
            header_parts = [f"知识{idx}", str(title)]
            if error_type:
                header_parts.append(str(error_type))
            header_parts.append(str(source))
            blocks.append(
                "[{0}]\n{1}".format(
                    " / ".join(header_parts),
                    str(item.get("text", "")).strip(),
                )
            )
        return "\n\n".join(blocks)
