from __future__ import annotations

from openclaw_skill.retrieval.retriever import RuleRetriever


def test_retrieval_smoke() -> None:
    retriever = RuleRetriever("vector_store/food_label_rules_v4")
    queries = [
        "低糖含量声称标准",
        "反式脂肪酸 0 标示阈值",
        "净含量1000g单位标示要求",
    ]

    for q in queries:
        results = retriever.search(q, top_k=3)
        assert results, f"No result for query: {q}"
        first = results[0]
        assert "sources" in first, f"No sources for query: {q}"
        assert len(first["sources"]) > 0, f"Empty sources for query: {q}"
    for q in queries:
        print(q)