import json
import logging
import sys
from pathlib import Path

import chromadb
import requests

from config import settings

logger = logging.getLogger("ollama_rag_local.eval")

QUESTIONS_FILE = Path("eval_questions.jsonl")
RESULTS_FILE = Path("eval_results.json")
REPORT_FILE = Path("EVAL_REPORT.md")


def load_questions(filepath: Path) -> list[dict]:
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found")
    questions = []
    with filepath.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            questions.append(json.loads(line))
    return questions


def get_embedding(text: str) -> list[float]:
    payload = {"model": settings.embed_model, "prompt": text}
    resp = requests.post(f"{settings.ollama_url}/api/embeddings", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    vec = data.get("embedding")
    if not isinstance(vec, list):
        raise RuntimeError("Embedding response missing vector")
    return vec


def format_source(meta: dict) -> str:
    source = meta.get("source", "unknown")
    page = meta.get("page", "unknown")
    return f"{source}#{page}"


def evaluate_question(collection: chromadb.Collection, question: dict) -> dict:
    q_text = question["question"]
    expected = question.get("expected_sources", [])
    embedding = get_embedding(q_text)
    res = collection.query(
        query_embeddings=[embedding],
        n_results=settings.top_k,
        include=["documents", "metadatas", "distances"],
    )
    metas = res.get("metadatas", [[]])[0]
    sources = [format_source(meta) for meta in metas]
    hit = any(exp in sources for exp in expected)
    return {
        "id": question.get("id"),
        "question": q_text,
        "expected_sources": expected,
        "retrieved_sources": sources,
        "distances": res.get("distances", [[]])[0],
        "retrieval_hit": hit,
    }


def write_report(results: list[dict]) -> None:
    total = len(results)
    hits = sum(1 for r in results if r["retrieval_hit"])
    hit_rate = hits / total if total else 0.0

    lines = [
        "# Local Ollama RAG Evaluation",
        "",
        f"- Questions evaluated: {total}",
        f"- Retrieval hit rate: {hit_rate:.2%}",
        "",
    ]

    failures = [r for r in results if not r["retrieval_hit"]]
    if failures:
        lines.append("## Failures")
        lines.append("")
        for failure in failures:
            lines.append(f"### {failure['id'] or failure['question']}")
            lines.append(f"- Expected: {', '.join(failure['expected_sources']) or 'None'}")
            lines.append(f"- Retrieved: {', '.join(failure['retrieved_sources']) or 'None'}")
            lines.append("")

    lines.append("## Full Results")
    lines.append("")
    for result in results:
        lines.append(f"- {result['id'] or result['question']}: {'HIT' if result['retrieval_hit'] else 'MISS'}")

    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", REPORT_FILE)


def main() -> None:
    if not settings.chroma_dir.exists():
        logger.error("%s not found. Run build_index.py first.", settings.chroma_dir)
        return

    questions = load_questions(QUESTIONS_FILE)
    if not questions:
        logger.warning("No questions found in %s.", QUESTIONS_FILE)
        return

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.get_or_create_collection(name=settings.collection_name)

    results = []
    for question in questions:
        try:
            result = evaluate_question(collection, question)
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            result = {
                "id": question.get("id"),
                "question": question["question"],
                "expected_sources": question.get("expected_sources", []),
                "retrieved_sources": [],
                "distances": [],
                "retrieval_hit": False,
                "error": str(exc),
            }
        results.append(result)

    RESULTS_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Wrote %s", RESULTS_FILE)
    write_report(results)

    print(f"Evaluation complete. Results written to {RESULTS_FILE} and {REPORT_FILE}.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        main()
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except requests.RequestException as exc:
        logger.error("Ollama request failed: %s", exc)
        sys.exit(1)
