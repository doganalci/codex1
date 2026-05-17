"""Lightweight RAG layer over ChromaDB.

- PDF metni pypdf ile sayfa bazlı çıkarılır; her sayfa metadata['page']
  ile saklanır.
- Embedding modeli config'den okunur; OpenAI uyumlu API üzerinden hesaplanır.
- Koleksiyon ismi kullanıcıdan alınır (uygulama içinde önemli).
"""
from __future__ import annotations

import re
from pathlib import Path

import chromadb
from openai import OpenAI
from pypdf import PdfReader

from .config import settings


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def _chroma():
    return chromadb.PersistentClient(path=str(settings.vectorstore_dir))


def list_collections() -> list[str]:
    return [c.name for c in _chroma().list_collections()]


def collection_info(name: str) -> dict:
    col = _chroma().get_or_create_collection(name)
    return {"name": name, "count": col.count(), "metadata": col.metadata or {}}


def _chunk(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text).strip()
    if not text:
        return []
    out: list[str] = []
    i = 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return out


def _read_pdf(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    pages: list[tuple[int, str]] = []
    for idx, p in enumerate(reader.pages, 1):
        try:
            t = p.extract_text() or ""
        except Exception:
            t = ""
        if t.strip():
            pages.append((idx, t))
    return pages


def _embed(texts: list[str]) -> list[list[float]]:
    resp = _client().embeddings.create(model=settings.embedding_model, input=texts)
    return [d.embedding for d in resp.data]


def ingest_documents(collection_name: str, file_paths: list[Path], batch: int = 64) -> dict:
    """Ingest PDFs/TXTs into a (possibly existing) collection."""
    client = _chroma()
    col = client.get_or_create_collection(
        collection_name,
        metadata={"embedding_model": settings.embedding_model},
    )

    added_chunks = 0
    docs_added: list[str] = []
    existing_ids = set()
    try:
        # crude existing-id check (chroma get without ids returns up to 'limit')
        existing = col.get(include=[])
        existing_ids = set(existing.get("ids", []))
    except Exception:
        pass

    for path in file_paths:
        doc_name = path.name
        if path.suffix.lower() == ".pdf":
            pages = _read_pdf(path)
        else:
            pages = [(1, path.read_text(encoding="utf-8", errors="ignore"))]

        chunks: list[str] = []
        metas: list[dict] = []
        ids: list[str] = []
        for page_no, page_text in pages:
            for ci, ch in enumerate(_chunk(page_text)):
                _id = f"{doc_name}::p{page_no}::c{ci}"
                if _id in existing_ids:
                    continue
                ids.append(_id)
                chunks.append(ch)
                metas.append({"document": doc_name, "page": page_no, "chunk": ci})

        for s in range(0, len(chunks), batch):
            sub_ids = ids[s : s + batch]
            sub_docs = chunks[s : s + batch]
            sub_meta = metas[s : s + batch]
            embs = _embed(sub_docs)
            col.add(ids=sub_ids, documents=sub_docs, embeddings=embs, metadatas=sub_meta)
            added_chunks += len(sub_docs)

        if chunks:
            docs_added.append(doc_name)

    return {"collection": collection_name, "documents": docs_added, "chunks_added": added_chunks}


def retrieve(collection_name: str, query: str, k: int = 8) -> list[dict]:
    col = _chroma().get_or_create_collection(collection_name)
    if col.count() == 0:
        return []
    emb = _embed([query])[0]
    res = col.query(query_embeddings=[emb], n_results=k)
    out: list[dict] = []
    for doc, meta in zip(res["documents"][0], res["metadatas"][0]):
        out.append({"text": doc, "metadata": meta})
    return out
