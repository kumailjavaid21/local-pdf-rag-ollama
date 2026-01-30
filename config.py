import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    embed_model: str = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    chat_model: str = os.environ.get("CHAT_MODEL", "gemma3:4b")
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    chunk_size: int = int(os.environ.get("CHUNK_SIZE", "180"))
    chunk_overlap: int = int(os.environ.get("CHUNK_OVERLAP", "40"))
    top_k: int = int(os.environ.get("TOP_K", "30"))
    final_k: int = int(os.environ.get("FINAL_K", "6"))
    docs_dir: Path = Path(os.environ.get("DOCS_DIR", "docs"))
    chroma_dir: Path = Path(os.environ.get("CHROMA_DIR", "chroma_db"))
    collection_name: str = os.environ.get("CHROMA_COLLECTION", "local_docs")
    min_chunk_words: int = int(os.environ.get("MIN_CHUNK_WORDS", "40"))


settings = Settings()
