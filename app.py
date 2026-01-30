import json
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import chromadb
import requests
import streamlit as st

from config import settings


def ensure_docs_dir() -> Path:
    settings.docs_dir.mkdir(exist_ok=True, parents=True)
    return settings.docs_dir


def save_uploaded_pdfs(files: Iterable[st.runtime.uploaded_file_manager.UploadedFile]):
    docs_dir = ensure_docs_dir()
    saved = []
    for uploaded in files:
        target = docs_dir / uploaded.name
        with target.open("wb") as fh:
            fh.write(uploaded.getbuffer())
        saved.append(target)
    return saved


def build_prompt(question: str, context: str) -> str:
    instruction = (
        "Answer using only the provided context. "
        "Cite sentences with [S#]. If the answer is not covered, "
        "say \"I couldn't find this in the PDFs.\" End with Sources."
    )
    return f"{instruction}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"


def format_sources(metadatas: Iterable[dict]) -> str:
    lines = []
    for idx, meta in enumerate(metadatas, start=1):
        source = meta.get("source", "unknown")
        page = meta.get("page", "unknown")
        lines.append(f"[S{idx}] {source} page {page}")
    return "\n".join(lines)


def get_embedding(text: str) -> list[float]:
    payload = {"model": settings.embed_model, "prompt": text}
    resp = requests.post(f"{settings.ollama_url}/api/embeddings", json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("embedding", [])


def ask_llm(prompt: str) -> str:
    payload = {
        "model": settings.chat_model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "format": "json",
    }
    resp = requests.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def query_documents(question: str, top_k: int):
    if not settings.chroma_dir.exists():
        st.error(f"{settings.chroma_dir} missing—run build_index.py first.")
        return None, None, None
    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = client.get_or_create_collection(name=settings.collection_name)
    embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    return docs, metas, dists


def main():
    st.set_page_config(page_title="Local Ollama RAG", layout="wide")
    st.title("Local Ollama + Chroma RAG")

    with st.sidebar:
        st.header("Setup")
        st.text_input("Ollama URL", value=settings.ollama_url, disabled=True)
        st.text_input("Chat model", value=settings.chat_model, disabled=True)
        st.text_input("Embedding model", value=settings.embed_model, disabled=True)
        st.text_input("Docs dir", value=str(settings.docs_dir), disabled=True)
        st.text_input("Chroma dir", value=str(settings.chroma_dir), disabled=True)
        st.markdown("## Actions")
        uploaded = st.file_uploader("Upload PDFs to docs/", type=["pdf"], accept_multiple_files=True)
        if uploaded:
            saved = save_uploaded_pdfs(uploaded)
            st.success(f"Saved {len(saved)} file(s).")
        if st.button("Rebuild index"):
            st.info("Rebuilding index…")
            subprocess.run([sys.executable, "build_index.py"], check=False)
            st.success("build_index.py finished.")

    st.markdown("## Chat with the indexed PDFs")
    question = st.text_input("Ask a question", "")
    top_k = st.slider("Top K retrievals", min_value=3, max_value=30, value=settings.top_k)
    if st.button("Answer") and question.strip():
        with st.spinner("Searching the index..."):
            docs, metas, dists = query_documents(question, top_k)
        if not docs:
            st.warning("No documents indexed.")
            return
        context = "\n\n".join(
            f"[S{idx + 1}] {meta.get('source', 'unknown')} page {meta.get('page', 'unknown')}\n{doc}"
            for idx, (doc, meta) in enumerate(zip(docs, metas))
        )
        prompt = build_prompt(question, context)
        with st.spinner("Calling Ollama..."):
            try:
                answer = ask_llm(prompt)
            except requests.RequestException as exc:
                st.error(f"Ollama request failed: {exc}")
                return
        if not answer:
            st.info("I couldn't find this in the PDFs.")
        else:
            st.subheader("Answer")
            st.write(answer)
            st.subheader("Sources")
            st.text(format_sources(metas))


if __name__ == "__main__":
    main()
