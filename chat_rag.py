import json
import logging
import sys
from typing import Iterable

import chromadb
import requests

from config import settings

logger = logging.getLogger("ollama_rag_local.chat_rag")


def format_sources(metadatas: Iterable[dict]) -> str:
    lines = []
    for idx, meta in enumerate(metadatas, start=1):
        source = meta.get("source", "unknown")
        page = meta.get("page", "unknown")
        lines.append(f"- S{idx}: {source} (page {page})")
    return "\n".join(lines)


def build_context(docs: Iterable[str], metadatas: Iterable[dict]) -> str:
    parts = []
    for idx, (doc, meta) in enumerate(zip(docs, metadatas), start=1):
        source = meta.get("source", "unknown")
        page = meta.get("page", "unknown")
        parts.append(f"[S{idx}] {source} page {page}\n{doc}")
    return "\n\n".join(parts)


def build_prompt(question: str, context: str) -> str:
    instruction = (
        "Use the context to answer the question in a grounded way. "
        "Cite each relevant paragraph as [S#]. "
        "If nothing in the context supports the answer, say "
        "\"I couldn't find this in the PDFs.\" "
        "End with a Sources section listing filename and page numbers."
    )
    return f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


def get_embedding(text: str) -> list[float]:
    payload = {"model": settings.embed_model, "prompt": text}
    resp = requests.post(f"{settings.ollama_url}/api/embeddings", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["embedding"]


def ask_chat(prompt: str) -> str:
    payload = {"model": settings.chat_model, "messages": [{"role": "user", "content": prompt}], "stream": False, "format": "json"}
    resp = requests.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    data = resp.json()
    return data.get("message", {}).get("content", "").strip()


def main() -> None:
    if not settings.chroma_dir.exists():
        logger.error("Missing %s. Run build_index.py first.", settings.chroma_dir)
        return

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.get_or_create_collection(name=settings.collection_name)

    print("Local RAG chat. Type a question, or 'quit' to exit.")
    while True:
        question = input("\nQuestion: ").strip()
        if not question or question.lower() in {"quit", "exit"}:
            print("Bye.")
            break

        logger.info("Question: %s", question)
        try:
            query_embedding = get_embedding(question)
        except requests.RequestException as exc:
            logger.error("Failed to retrieve embedding: %s", exc)
            print(f"ERROR: Unable to contact Ollama for embeddings: {exc}")
            continue

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=settings.top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]

        if not docs:
            print("I couldn't find this in the PDFs.")
            continue

        context = build_context(docs, metas)
        prompt = build_prompt(question, context)
        print("\nSources:")
        print(format_sources(metas))
        try:
            answer = ask_chat(prompt)
        except requests.RequestException as exc:
            logger.error("Ollama chat request failed: %s", exc)
            print(f"ERROR: Ollama chat request failed: {exc}")
            continue

        if not answer:
            print("I couldn't find this in the PDFs.")
        else:
            print("\nAnswer:")
            print(answer)
            print("\nSources:")
            print(format_sources(metas))
            logger.info("Answer returned with %d source fragments.", len(metas))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        main()
    except requests.RequestException as exc:
        logger.error("Ollama request failed: %s", exc)
        sys.exit(1)
