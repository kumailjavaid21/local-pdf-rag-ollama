import logging
import sys
from pathlib import Path

import chromadb
import requests
from pypdf import PdfReader
from tqdm import tqdm

from config import settings

logger = logging.getLogger("ollama_rag_local.build_index")


def find_pdfs(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.pdf") if p.is_file())


def extract_pages(pdf_path: Path) -> list[str]:
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def normalize_text(text: str) -> str:
    text = text.replace("-\n", "")
    text = text.replace("\n", " ")
    return " ".join(text.split())


def chunk_text(text: str) -> list[str]:
    chunk_size = settings.chunk_size
    overlap = settings.chunk_overlap
    min_words = settings.min_chunk_words
    if chunk_size <= 0:
        return []
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 3)

    words = text.split()
    if not words:
        return []

    step = max(1, chunk_size - overlap)
    chunks = []
    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        if len(chunk_words) >= min_words:
            chunks.append(" ".join(chunk_words))

    return chunks


def get_embedding(text: str) -> list[float]:
    payload = {"model": settings.embed_model, "prompt": text}
    resp = requests.post(f"{settings.ollama_url}/api/embeddings", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    vec = data.get("embedding")
    if not isinstance(vec, list) or len(vec) == 0:
        snippet = resp.text[:300]
        raise RuntimeError(f"Embedding response missing vector: {snippet}")
    return vec


def ensure_collection(client: chromadb.PersistentClient) -> chromadb.Collection:
    try:
        collection = client.get_collection(name=settings.collection_name)
        space = (collection.metadata or {}).get("hnsw:space")
        if space != "cosine":
            raise RuntimeError("Existing collection uses wrong metric.")
    except Exception:
        collection = client.get_or_create_collection(
            name=settings.collection_name, metadata={"hnsw:space": "cosine"}
        )
    return collection


def main() -> None:
    if not settings.docs_dir.exists():
        logger.error("Docs directory %s missing.", settings.docs_dir)
        return

    pdfs = find_pdfs(settings.docs_dir)
    if not pdfs:
        logger.error("No PDFs found under %s.", settings.docs_dir)
        return

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = ensure_collection(client)

    total_chunks = 0
    for pdf_path in pdfs:
        rel_name = str(pdf_path.relative_to(settings.docs_dir))
        pages = extract_pages(pdf_path)
        for page_idx, page_text in enumerate(pages, start=1):
            clean = normalize_text(page_text)
            chunks = chunk_text(clean)
            total_chunks += len(chunks)

    if total_chunks == 0:
        logger.error("Found PDFs but no extractable text.")
        return

    logger.info("Indexing %d PDFs with %d chunks.", len(pdfs), total_chunks)
    added = 0
    with tqdm(total=total_chunks, desc="Embedding chunks") as pbar:
        for pdf_path in pdfs:
            rel_name = str(pdf_path.relative_to(settings.docs_dir))
            pages = extract_pages(pdf_path)
            for page_idx, page_text in enumerate(pages, start=1):
                clean = normalize_text(page_text)
                chunks = chunk_text(clean)
                for chunk_idx, chunk in enumerate(chunks):
                    doc_id = f"{rel_name}-p{page_idx}-c{chunk_idx}"
                    metadata = {"source": rel_name, "page": page_idx, "chunk_index": chunk_idx}
                    embedding = get_embedding(chunk)
                    collection.add(
                        ids=[doc_id],
                        documents=[chunk],
                        embeddings=[embedding],
                        metadatas=[metadata],
                    )
                    added += 1
                    pbar.update(1)

    logger.info("Indexing complete. Total chunks added: %d", added)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        main()
    except requests.RequestException as exc:
        logger.error("Ollama request failed: %s", exc)
        sys.exit(1)
