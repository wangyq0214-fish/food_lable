from __future__ import annotations

from openclaw_skill.retrieval.retriever import RuleRetriever


INDEX_DIR = "vector_store/food_label_rules_v4"


def _hit_target_file(sources: list[dict], target_keyword: str) -> bool:
    for s in sources:
        file_name = (s.get("metadata") or {}).get("file_name", "")
        if target_keyword in file_name:
            return True
    return False


def test_retrieval_file_coverage() -> None:
    retriever = RuleRetriever(INDEX_DIR)

    cases = [
        ("GB7718 预包装食品标签通则 食品名称 配料表 净含量 生产者地址", "GB7718"),
        ("预包装食品标签通则 问答 常见问题", "标签通则》问答"),
        ("营养成分表 NRV 能量 蛋白质 脂肪 碳水化合物", "GB 28050"),
        ("营养标签通则 问答 修约间隔", "营养标签通则》问答"),
        ("食品标识监督管理办法 处罚", "食品标识监督管理办法"),
        ("食品添加剂 使用范围 CNS号 INS号", "GB2760"),
    ]

    for query, target_file_kw in cases:
        results = retriever.search(query, top_k=8)
        assert results, f"No retrieval result: {query}"

        first = results[0]
        sources = first.get("sources", [])
        assert sources, f"No sources returned: {query}"

        print(f"\n=== QUERY: {query} ===")
        for i, s in enumerate(sources[:5], start=1):
            md = s.get("metadata") or {}
            print(f"[{i}] score={s.get('score')} file={md.get('file_name')} heading={md.get('heading_key')}")

        assert _hit_target_file(sources, target_file_kw), (
            f"Top-k did not hit target file `{target_file_kw}` for query: {query}"
        )
