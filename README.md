# Local RAG with Ollama + ChromaDB (Windows)

This is a fully-local RAG project using Ollama for embeddings + chat, ChromaDB for storage, and PDFs as the knowledge base.

## Folder layout
- `docs/` : put your PDFs here (this folder is scanned recursively)
- `chroma_db/` : persistent ChromaDB storage (created automatically)

## Setup (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull nomic-embed-text
python build_index.py
python chat_rag.py
```

## Setup (CMD)
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull nomic-embed-text
python build_index.py
python chat_rag.py
```

## Rebuild instructions
- If you change chunking settings or see a warning about cosine similarity,
  delete `chroma_db/` and run `python build_index.py` again.

## Run evaluation
Ensure `eval_questions.jsonl` is populated with representative questions, then run:
```powershell
(.venv) PS D:\ollama_rag_local> python eval.py
```
This produces `eval_results.json` (per-question logging) and `EVAL_REPORT.md` (summary + any failures).

## Streamlit UI
Install Streamlit in your virtualenv and run:
```powershell
(.venv) PS D:\ollama_rag_local> streamlit run app.py
```
The sidebar lets you upload PDFs and rebuild the index, while the main area provides a chat prompt and shows retrieved sources.

## Troubleshooting
- Ollama not running: make sure Ollama is started and reachable at `http://localhost:11434`.
- Missing model: run `ollama pull nomic-embed-text` (and `ollama pull gemma3:4b` if needed).
- No PDFs found: create `docs/` and add PDF files.
- No text extracted: some PDFs are scanned images; pypdf can't read image text.
- Empty index: if `build_index.py` says no chunks, check that your PDFs contain selectable text.
- Chroma DB missing: run `python build_index.py` before `python chat_rag.py`.
