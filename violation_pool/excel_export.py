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

    prompt_short = (run["prompt"] or "")[:120].replace("\n", " ")
    v_rows = []
    e_rows = []
    for v in violations:
        v_rows.append(
            {
                "violation_id": v["id"],
                "batch_no": v.get("batch_no"),
                "method": run["method"],
                "title": v.get("title"),
                "description": v["description"],
                "category": v.get("category"),
                "severity": v.get("severity"),
                "threshold": v.get("threshold"),
                "evidence_count": len(v.get("evidence", [])),
                "llm_model": run["llm_model"],
                "prompt_short": prompt_short,
                "created_at": v.get("created_at"),
            }
        )
        for ev in v.get("evidence", []):
            e_rows.append(
                {
                    "violation_id": v["id"],
                    "batch_no": v.get("batch_no"),
                    "document": ev.get("document"),
                    "page": ev.get("page"),
                    "clause": ev.get("clause"),
                    "snippet": ev.get("snippet"),
                }
            )

    tot = storage.usage_totals(pool_run_id=run["id"])
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
        {"key": "token_total", "value": tot["total_tokens"]},
        {"key": "token_prompt", "value": tot["prompt_tokens"]},
        {"key": "token_completion", "value": tot["completion_tokens"]},
        {"key": "llm_calls", "value": tot["calls"]},
    ]

    out = settings.export_dir / f"violations_{run['name']}_{run['id'][:8]}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(v_rows).to_excel(w, index=False, sheet_name="violations")
        pd.DataFrame(e_rows).to_excel(w, index=False, sheet_name="evidence")
        pd.DataFrame(meta_rows).to_excel(w, index=False, sheet_name="run_meta")
    return out
