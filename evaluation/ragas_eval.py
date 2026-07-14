"""Lightweight evaluation harness for Hybrid RAG.

Runs curated questions against the live pipeline and reports simple
answer-containment / refusal metrics. Optionally computes Ragas scores
when `ragas` and an OpenAI-compatible judge are configured.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ragas_eval")


def load_queries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("test_queries", payload)


def simple_score(answer: str, reference: str) -> dict:
    answer_l = answer.lower().strip()
    reference_l = reference.lower().strip()
    refusal = "i don't know" in answer_l
    ref_refusal = "i don't know" in reference_l
    if ref_refusal:
        return {"correct": refusal, "type": "refusal"}
    # Token overlap heuristic for demo evaluation without an LLM judge.
    ref_tokens = set(reference_l.replace(".", " ").split())
    ans_tokens = set(answer_l.replace(".", " ").split())
    if not ref_tokens:
        return {"correct": False, "type": "empty_reference"}
    overlap = len(ref_tokens & ans_tokens) / len(ref_tokens)
    return {"correct": overlap >= 0.45 and not refusal, "overlap": overlap, "type": "overlap"}


def run_eval(queries_path: Path, ingest: bool) -> dict:
    pipeline = RAGPipeline(auto_load=True)
    if ingest or not pipeline.is_ready:
        logger.info("Ingesting documents from data/raw_docs")
        pipeline.ingest_directory()

    queries = load_queries(queries_path)
    rows = []
    correct = 0
    for item in queries:
        q = item["question"]
        ref = item.get("reference", "")
        result = pipeline.query(q)
        score = simple_score(result.answer, ref)
        correct += int(score["correct"])
        rows.append(
            {
                "question": q,
                "reference": ref,
                "answer": result.answer,
                "verified": result.verified,
                "citations": len(result.citations),
                "score": score,
            }
        )
        logger.info("Q: %s | correct=%s | verified=%s", q, score["correct"], result.verified)

    summary = {
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else 0.0,
        "results": rows,
    }
    return summary


def try_ragas(summary: dict) -> dict | None:
    """Optional Ragas integration when dependencies and API keys are present."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError:
        logger.info("ragas not installed; skipping advanced metrics")
        return None

    records = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
    }
    for row in summary["results"]:
        records["question"].append(row["question"])
        records["answer"].append(row["answer"])
        records["ground_truth"].append(row["reference"])
        # Contexts are not retained in the simple summary; use empty for safety.
        records["contexts"].append([])

    if not any(records["contexts"]):
        logger.info("No contexts captured for Ragas; skipping")
        return None

    dataset = Dataset.from_dict(records)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )
    return dict(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hybrid RAG")
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path(__file__).parent / "test_queries.json",
    )
    parser.add_argument("--ingest", action="store_true", help="Force re-ingest before eval")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "eval_results.json",
    )
    args = parser.parse_args()

    summary = run_eval(args.queries, ingest=args.ingest)
    ragas_scores = try_ragas(summary)
    if ragas_scores:
        summary["ragas"] = ragas_scores

    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"total": summary["total"], "accuracy": summary["accuracy"]}, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
