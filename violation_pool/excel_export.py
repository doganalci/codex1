"""Excel export for violation pools and IFC labels."""
from __future__ import annotations

import json
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


def export_ifc_labels(ifc_model_id: str) -> Path:
    """Bir violated/imported IFC için etiketleri (her ihlal + tüm kanıtları
    ile) Excel olarak indir. Her kanıt satırı ayrı bir row; etiketin
    ihlal_id'si linklenmiş kalır.
    """
    m = storage.get_ifc_model(ifc_model_id)
    if not m:
        raise ValueError(f"IFC bulunamadı: {ifc_model_id}")
    labels = storage.get_ifc_labels(ifc_model_id)

    label_rows = []
    evidence_rows = []
    for l in labels:
        label_rows.append({
            "label_id": l["id"],
            "violation_id": l.get("violation_id"),   # havuzdaki ihlal kodu
            "is_decoy": bool(l.get("is_decoy")),
            "action": l.get("action"),
            "status": l["status"],
            "title": l["title"],
            "category": l["category"],
            "severity": l["severity"],
            "threshold": l["threshold"],
            "ifc_global_id": l["ifc_global_id"],
            "ifc_type": l["ifc_type"],
            "ifc_name": l["ifc_name"],
            "attribute": l["attribute"],
            "value_before": l["value_before"],
            "value_after": l["value_after"],
            "reason": l["reason"],
            "applied_at": l["applied_at"],
        })
        try:
            evs = json.loads(l.get("evidence_json") or "[]")
        except Exception:
            evs = []
        for ev in (evs or []):
            evidence_rows.append({
                "label_id": l["id"],
                "violation_id": l.get("violation_id"),
                "ifc_global_id": l["ifc_global_id"],
                "document": (ev or {}).get("document"),
                "page": (ev or {}).get("page"),
                "clause": (ev or {}).get("clause"),
                "snippet": (ev or {}).get("snippet"),
            })

    meta_rows = [
        {"key": "ifc_model_id", "value": m["id"]},
        {"key": "name", "value": m["name"]},
        {"key": "kind", "value": m["kind"]},
        {"key": "parent_id", "value": m.get("parent_id")},
        {"key": "pool_run_id", "value": m.get("pool_run_id")},
        {"key": "llm_model", "value": m["llm_model"]},
        {"key": "status", "value": m["status"]},
        {"key": "file_path", "value": m["file_path"]},
        {"key": "labels_path", "value": m.get("labels_path")},
        {"key": "graph_path", "value": m.get("graph_path")},
        {"key": "created_at", "value": m["created_at"]},
    ]

    out = settings.export_dir / f"ifc_labels_{m['name']}_{m['id'][:8]}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame(label_rows).to_excel(w, index=False, sheet_name="labels")
        pd.DataFrame(evidence_rows).to_excel(w, index=False, sheet_name="evidence")
        pd.DataFrame(meta_rows).to_excel(w, index=False, sheet_name="ifc_meta")
    return out
