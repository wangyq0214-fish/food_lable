from __future__ import annotations

import argparse
import json

from openclaw_skill.config import SkillConfig
from openclaw_skill.graph_workflow import GraphFoodLabelAuditSkill
from openclaw_skill.index_build import build_index, export_chunks_only
from openclaw_skill.workflow import FoodLabelAuditSkill


def main() -> None:
    cfg = SkillConfig()
    parser = argparse.ArgumentParser(description="Run food label audit skill")

    parser.add_argument(
        "source",
        nargs="?",
        help="Path to label source file (.txt/.docx/.jpg/.jpeg/.png/.bmp/.webp)",
    )
    parser.add_argument(
        "--engine",
        choices=["classic", "graph"],
        default="classic",
        help="Execution engine: classic (rule scaffold) or graph (LangGraph multi-agent)",
    )

    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build offline vector index from standards and exit",
    )
    parser.add_argument(
        "--export-chunks-only",
        action="store_true",
        help="Only split one regulation txt and export chunks to txt, without indexing",
    )
    parser.add_argument(
        "--index-source-dir",
        default="data/cleaned",
        help="Source directory for offline index building",
    )
    parser.add_argument(
        "--index-persist-dir",
        default="vector_store/food_label_rules",
        help="Persist directory for offline index storage",
    )
    parser.add_argument(
        "--chunks-source-file",
        default="",
        help="Source txt file for --export-chunks-only",
    )
    parser.add_argument(
        "--chunks-output-file",
        default="outputs/chunks_preview.txt",
        help="Output txt path for chunk export",
    )
    parser.add_argument(
        "--split-mode",
        choices=["law", "standard", "table_aware", "additive"],
        default="law",
        help="Chunk split mode for --export-chunks-only / --build-index",
    )
    parser.add_argument(
        "--embed-model",
        choices=["mock", "text-embedding-v4"],
        default=cfg.embedding_model,
        help="Embedding model for --build-index (default from EMBEDDING_MODEL)",
    )
    parser.add_argument(
        "--chunks-files",
        nargs="*",
        default=None,
        help="Use pre-split chunk preview txt files directly for indexing",
    )
    parser.add_argument(
        "--append-index",
        action="store_true",
        help="Append new docs into existing index persist dir instead of rebuilding",
    )

    parser.add_argument(
        "--debug-ocr",
        action="store_true",
        help="Enable saving raw OCR response to file",
    )
    parser.add_argument(
        "--debug-ocr-file",
        default="outputs/ocr_raw.json",
        help="Custom path for raw OCR response output",
    )
    parser.add_argument(
        "--debug-parsed-file",
        default="",
        help="Optional path to save parsed generic data (full_text/text_blocks/raw_records)",
    )

    args = parser.parse_args()

    if args.export_chunks_only:
        if not args.chunks_source_file:
            parser.error("--chunks-source-file is required when --export-chunks-only is used")
        out_file = export_chunks_only(
            args.chunks_source_file,
            args.chunks_output_file,
            split_mode=args.split_mode,
        )
        print(f"CHUNKS: {out_file}")
        return

    if args.build_index:
        path = build_index(
            args.index_source_dir,
            args.index_persist_dir,
            args.chunks_output_file,
            split_mode=args.split_mode,
            embed_model=args.embed_model,
            chunks_files=args.chunks_files,
            append=args.append_index,
        )
        print(f"INDEX: {path}")
        print(f"CHUNKS: {args.chunks_output_file}")
        return

    if not args.source:
        parser.error("source is required unless --build-index is used")

    skill = GraphFoodLabelAuditSkill() if args.engine == "graph" else FoodLabelAuditSkill()
    result, docx_path = skill.run(
        args.source,
        debug_ocr=args.debug_ocr,
        debug_ocr_file=args.debug_ocr_file,
        debug_parsed_file=(args.debug_parsed_file or None),
    )

    print(json.dumps(result.summary, ensure_ascii=False, indent=2))
    print(f"DOCX: {docx_path}")


if __name__ == "__main__":
    main()
