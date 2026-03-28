from __future__ import annotations

import re
from pathlib import Path

from llama_index.core import StorageContext, load_index_from_storage

from openclaw_skill.config import SkillConfig
from openclaw_skill.index_build import _setup_embed_model


class RuleRetriever:
    def __init__(self, persist_dir: str = "vector_store/food_label_rules", embed_model: str | None = None):
        self.persist_dir = Path(persist_dir)
        self.embed_model = embed_model or SkillConfig().embedding_model
        self._index = None
        self._docs_cache: list[dict] = []

    def _load(self):
        if self._index is not None:
            return self._index
        if not self.persist_dir.exists():
            raise FileNotFoundError(f"Index not found: {self.persist_dir}")

        # Ensure same embedding backend as index build (avoid falling back to OpenAI default).
        _setup_embed_model(self.embed_model)

        storage = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
        self._index = load_index_from_storage(storage)

        # cache corpus for lexical retrieval
        self._docs_cache = []
        try:
            for node in self._index.docstore.docs.values():
                self._docs_cache.append(
                    {
                        "id": getattr(node, "node_id", ""),
                        "text": node.get_text() if hasattr(node, "get_text") else "",
                        "metadata": getattr(node, "metadata", {}) or {},
                    }
                )
        except Exception:
            self._docs_cache = []

        return self._index

    def _tokenize(self, text: str) -> set[str]:
        return set(t for t in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", text.lower()) if t)

    def _lexical_search(self, query: str, top_k: int) -> list[dict]:
        q_tokens = self._tokenize(query)
        if not q_tokens or not self._docs_cache:
            return []

        scored: list[tuple[float, dict]] = []
        for d in self._docs_cache:
            tokens = self._tokenize(d.get("text", ""))
            if not tokens:
                continue
            overlap = q_tokens.intersection(tokens)
            if not overlap:
                continue
            score = len(overlap) / max(len(q_tokens), 1)
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[dict] = []
        for score, d in scored[:top_k]:
            out.append(
                {
                    "id": d.get("id", ""),
                    "score": score,
                    "text": d.get("text", "")[:500],
                    "metadata": d.get("metadata", {}),
                }
            )
        return out

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        index = self._load()

        # dense retrieval
        retriever = index.as_retriever(similarity_top_k=max(top_k, 8))
        nodes = retriever.retrieve(query)
        dense_sources: list[dict] = []
        for n in nodes or []:
            node = getattr(n, "node", None)
            dense_sources.append(
                {
                    "id": getattr(node, "node_id", "") if node else "",
                    "score": float(getattr(n, "score", 0.0) or 0.0),
                    "text": node.get_text()[:500] if node else "",
                    "metadata": getattr(node, "metadata", {}) if node else {},
                }
            )

        # lexical retrieval
        lexical_sources = self._lexical_search(query, top_k=max(top_k, 8))

        # hybrid fusion
        dense_norm = {}
        if dense_sources:
            max_dense = max(s["score"] for s in dense_sources) or 1.0
            for s in dense_sources:
                dense_norm[s["id"] or s["text"]] = s["score"] / max_dense

        lex_norm = {}
        if lexical_sources:
            max_lex = max(s["score"] for s in lexical_sources) or 1.0
            for s in lexical_sources:
                lex_norm[s["id"] or s["text"]] = s["score"] / max_lex

        merged: dict[str, dict] = {}
        for s in dense_sources:
            key = s["id"] or s["text"]
            merged[key] = {**s, "score_dense": dense_norm.get(key, 0.0), "score_lex": 0.0}
        for s in lexical_sources:
            key = s["id"] or s["text"]
            if key not in merged:
                merged[key] = {**s, "score_dense": 0.0, "score_lex": lex_norm.get(key, 0.0)}
            else:
                merged[key]["score_lex"] = lex_norm.get(key, 0.0)

        # weighted hybrid score
        for s in merged.values():
            s["score"] = 0.7 * float(s.get("score_dense", 0.0)) + 0.3 * float(s.get("score_lex", 0.0))

        sources = sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:top_k]

        answer = "\n".join(s.get("text", "") for s in sources[: min(3, len(sources))])
        return [{"answer": answer, "sources": sources}]

