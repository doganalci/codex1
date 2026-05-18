"""Excel export for violation pools."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import storage
from .config import settings


def export_run(run_id: str) -> Path:
    run = storage.get_run(run_id)
    if not run:
        raise ValueError(f"Run bulunamadı: {run_id}")
    violations = storage.get_violations(run_id)

    v_rows = []
    e_rows = []
    for v in violations:
        v_rows.append(
            {
                "violation_id": v["id"],
                "title": v.get("title"),
                "description": v["description"],
                "category": v.get("category"),
                "severity": v.get("severity"),
                "threshold": v.get("threshold"),
                "evidence_count": len(v.get("evidence", [])),
                "created_at": v.get("created_at"),
            }
        )
        for ev in v.get("evidence", []):
            e_rows.append(
                {
                    "violation_id": v["id"],
                    "document": ev.get("document"),
                    "page": ev.get("page"),
                    "clause": ev.get("clause"),
                    "snippet": ev.get("snippet"),
                }
            )

    meta_rows = [
        {"key": "run_id", "value": run["id"]},
        {"key": "name", "value": run["name"]},
        {"key": "method", "value": run["method"]},
        {"key": "status", "value": run["status"]},
        {"key": "llm_model", "value": run["llm_model"]},
        {"key": "embedding_model", "value": run.get("embedding_model")},
        {"key": "rag_collection", "value": run.get("rag_collection")},
        {"key": "rag_documents", "value": run.get("rag_documents")},
        {"key": "finetune_model_id", "value": run.get("finetune_model_id")},
        {"key": "created_at", "value": run["created_at"]},
        {"key": "updated_at", "value": run["updated_at"]},
        {"key": "prompt", "value": run["prompt"]},
    ]

    out = settings.export_dir / f"violations_{run['name']}_{run['id'][:8]}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(v_rows).to_excel(w, index=False, sheet_name="violations")
        pd.DataFrame(e_rows).to_excel(w, index=False, sheet_name="evidence")
        pd.DataFrame(meta_rows).to_excel(w, index=False, sheet_name="run_meta")
    return out
