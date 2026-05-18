"""OpenAI fine-tuning helpers.

Akış:
  1) build_training_jsonl(collection_name, ...) → dokümanın parçalarından
     optimize promtu kullanarak sentetik (user, assistant) çiftleri üret;
     OpenAI sohbet formatında JSONL'e yaz.
  2) start_finetune_job(jsonl_path, base_model) → dosyayı yükle, FT işini başlat.
  3) job_status(job_id), list_jobs(), list_ft_models().

Not: Bu modül LLM'i çağırarak sentetik veri üretir; maliyeti vardır.
"""
from __future__ import annotations

import json
from pathlib import Path

from openai import OpenAI

from .config import settings
from .prompts import OPTIMIZED_PROMPT
from .rag import _chroma  # type: ignore[attr-defined]


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def build_training_jsonl(
    collection_name: str,
    out_path: Path | None = None,
    samples_per_chunk: int = 1,
    max_chunks: int | None = 200,
    base_model: str | None = None,
) -> Path:
    """Bir koleksiyondaki parçalardan sentetik FT verisi üret.

    Her parça için, OPTIMIZED_PROMPT sistem promtu + parça bağlamı verilir;
    LLM'in döndürdüğü JSON cevap "assistant" mesajı olarak kaydedilir.
    """
    col = _chroma().get_or_create_collection(collection_name)
    data = col.get(include=["documents", "metadatas"])
    docs: list[str] = data.get("documents") or []
    metas: list[dict] = data.get("metadatas") or []
    if not docs:
        raise ValueError(f"Koleksiyon boş: {collection_name}")

    if max_chunks:
        docs = docs[:max_chunks]
        metas = metas[:max_chunks]

    out_path = out_path or (settings.export_dir / f"ft_{collection_name}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cli = _client()
    model = base_model or settings.llm_model
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for chunk, meta in zip(docs, metas):
            doc = meta.get("document", "?")
            page = meta.get("page", "?")
            user_msg = (
                f"# Bağlam\n[doküman={doc} sayfa={page}]\n{chunk}\n\n"
                "# Görev\nBu bağlamdan çıkarılabilecek somut ihlal kurallarını üret."
            )
            for _ in range(samples_per_chunk):
                try:
                    resp = cli.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": OPTIMIZED_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )
                    assistant = resp.choices[0].message.content or ""
                    from . import storage
                    storage.record_usage_from_openai(
                        getattr(resp, "usage", None),
                        operation="ft_prep", model=model,
                        collection=collection_name, note=f"chunk {doc}:p{page}",
                    )
                except Exception:
                    continue
                rec = {
                    "messages": [
                        {"role": "system", "content": OPTIMIZED_PROMPT},
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": assistant},
                    ]
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    return out_path


def start_finetune_job(jsonl_path: Path, base_model: str) -> dict:
    cli = _client()
    up = cli.files.create(file=open(jsonl_path, "rb"), purpose="fine-tune")
    job = cli.fine_tuning.jobs.create(training_file=up.id, model=base_model)
    return {"job_id": job.id, "file_id": up.id, "status": job.status, "model": base_model}


def job_status(job_id: str) -> dict:
    j = _client().fine_tuning.jobs.retrieve(job_id)
    return {
        "id": j.id,
        "status": j.status,
        "model": j.model,
        "fine_tuned_model": j.fine_tuned_model,
        "created_at": j.created_at,
        "finished_at": getattr(j, "finished_at", None),
        "error": getattr(j, "error", None),
    }


def list_jobs(limit: int = 10) -> list[dict]:
    cli = _client()
    page = cli.fine_tuning.jobs.list(limit=limit)
    return [
        {
            "id": j.id,
            "status": j.status,
            "model": j.model,
            "fine_tuned_model": j.fine_tuned_model,
            "created_at": j.created_at,
        }
        for j in page.data
    ]
