"""İhlal enjeksiyonu: baseline IFC + havuzdan ihlaller → violated IFC + labels.

Akış:
  1) Baseline IFC ifcopenshell ile yüklenir; düzenlenebilir eleman katalogu
     (Door/Window/Slab/Wall/Stair) çıkarılır.
  2) Her ihlal için LLM'e: "şu açıklamaya uyan bir hedef seç ve hangi
     attribute'u nasıl değiştireceğini söyle (JSON)".
  3) Önerilen değişiklik ifcopenshell ile uygulanır; öncesi/sonrası kaydedilir.
  4) Violated .ifc + .labels.json + .meta.json yazılır.
"""
from __future__ import annotations

import json
import random
import re
import uuid
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from . import storage
from .config import settings


INJECT_SYSTEM_PROMPT = """Sen bir BIM denetim aracısın. Sana bir IFC eleman
katalogu ve bir 'ihlal kuralı' verilir. Görevin: katalogdan EN UYGUN tek bir
hedef eleman seç ve hangi sayısal/string attribute'u hangi yeni değere
çekersen bu ihlali oluşturacağını söyle.

Kurallar:
- Sadece geçerli IFC attribute adı kullan (örn. OverallWidth, OverallHeight,
  Elevation, NominalHeight).
- new_value gerçekçi ve ihlali gerçekleştirecek seviyede olsun
  (örn. 'Kapı genişliği < 70 cm' → OverallWidth = 0.60).
- Uygulanabilir hedef yoksa applicable=false döndür ve reason yaz.

Çıktıyı SADECE şu JSON ile döndür:
{
  "applicable": true|false,
  "target_guid": "GUID",
  "ifc_type": "IfcDoor",
  "attribute": "OverallWidth",
  "new_value": 0.60,
  "rationale": "kısa açıklama",
  "reason": null
}"""


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


_INTERESTING = ("IfcDoor", "IfcWindow", "IfcWall", "IfcWallStandardCase",
                "IfcSlab", "IfcStair", "IfcStairFlight", "IfcRailing",
                "IfcBuildingStorey", "IfcSpace", "IfcRamp")


def _catalog(ifc_file) -> list[dict]:
    out: list[dict] = []
    for t in _INTERESTING:
        for el in ifc_file.by_type(t):
            attrs: dict = {}
            for a in ("OverallWidth", "OverallHeight", "NominalHeight",
                      "Elevation", "Name", "PredefinedType"):
                v = getattr(el, a, None)
                if v is not None:
                    attrs[a] = v
            out.append({
                "guid": el.GlobalId,
                "type": el.is_a(),
                "name": getattr(el, "Name", None),
                "attrs": attrs,
            })
    return out


def _short_catalog(cat: list[dict], limit: int = 60) -> str:
    sample = cat[:limit]
    lines = []
    for it in sample:
        kv = ", ".join(f"{k}={v}" for k, v in it["attrs"].items())
        lines.append(f"- {it['type']} guid={it['guid']} name={it['name']} {{{kv}}}")
    if len(cat) > limit:
        lines.append(f"... (+{len(cat)-limit} daha)")
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError("LLM JSON dönmedi")
    return json.loads(m.group(0))


def _propose_edit(violation: dict, cat: list[dict], model: str) -> dict:
    user = (
        "İhlal:\n"
        + json.dumps({k: violation.get(k) for k in ("title", "description",
                                                    "category", "severity",
                                                    "threshold")},
                     ensure_ascii=False, indent=2)
        + "\n\nKatalog:\n" + _short_catalog(cat)
    )
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": INJECT_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return _parse_json(resp.choices[0].message.content or "")


def _apply_edit(ifc_file, guid: str, attribute: str, new_value):
    """Hedef elemanın attribute'unu set et. Sayısal tip kontrolü minimum."""
    el = ifc_file.by_guid(guid)
    if el is None:
        raise ValueError(f"GUID bulunamadı: {guid}")
    if not hasattr(el, attribute):
        raise ValueError(f"{el.is_a()} üzerinde {attribute} yok")
    before = getattr(el, attribute)
    # cast: string elemanlar string kalır; sayısal alanlar float'a çekilir
    if isinstance(before, (int, float)) and not isinstance(new_value, (int, float)):
        try:
            new_value = float(new_value)
        except Exception:
            pass
    setattr(el, attribute, new_value)
    return before, new_value


def inject_violations(
    *,
    baseline_id: str,
    violations: list[dict],
    pool_run_id: str | None,
    model: str | None = None,
    selection_filter: dict | None = None,
) -> dict:
    """violations: havuzdan seçilmiş ihlal dict'leri (storage.get_violations
    çıktısı formatı).
    selection_filter: {"category": "Erişilebilirlik"} veya {"types": ["IfcDoor"]}
       — şimdilik bilgi amaçlı, label'a yazılır.
    """
    import ifcopenshell  # local import

    model = model or settings.ifc_llm_model
    base = storage.get_ifc_model(baseline_id)
    if not base:
        raise ValueError("Baseline IFC bulunamadı")

    src = ifcopenshell.open(base["file_path"])
    cat = _catalog(src)

    out_id = str(uuid.uuid4())
    out_dir = settings.ifc_dir / "violated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ifc = out_dir / f"{out_id}.ifc"
    out_lab = out_dir / f"{out_id}.labels.json"
    out_meta = out_dir / f"{out_id}.meta.json"

    labels: list[dict] = []
    applied = skipped = 0
    for v in violations:
        try:
            sug = _propose_edit(v, cat, model)
        except Exception as e:
            labels.append({**_label_base(v), "status": "skipped",
                           "reason": f"LLM hata: {e}",
                           "applied_at": datetime.utcnow().isoformat(timespec="seconds")})
            skipped += 1
            continue

        if not sug.get("applicable"):
            labels.append({**_label_base(v), "status": "skipped",
                           "reason": sug.get("reason") or "uygulanabilir hedef yok",
                           "applied_at": datetime.utcnow().isoformat(timespec="seconds")})
            skipped += 1
            continue

        try:
            before, after = _apply_edit(src, sug["target_guid"],
                                        sug["attribute"], sug["new_value"])
        except Exception as e:
            labels.append({**_label_base(v), "status": "skipped",
                           "reason": f"uygulama hatası: {e}",
                           "applied_at": datetime.utcnow().isoformat(timespec="seconds")})
            skipped += 1
            continue

        target = src.by_guid(sug["target_guid"])
        labels.append({
            **_label_base(v),
            "ifc_global_id": sug["target_guid"],
            "ifc_type": sug.get("ifc_type") or target.is_a(),
            "ifc_name": getattr(target, "Name", None),
            "attribute": sug["attribute"],
            "value_before": before,
            "value_after": after,
            "status": "applied",
            "reason": sug.get("rationale"),
            "applied_at": datetime.utcnow().isoformat(timespec="seconds"),
        })
        applied += 1

    src.write(str(out_ifc))

    summary = {"requested": len(violations), "applied": applied, "skipped": skipped}
    labels_doc = {
        "ifc_file": out_ifc.name,
        "baseline_id": baseline_id,
        "violated_id": out_id,
        "pool_run_id": pool_run_id,
        "llm_model": model,
        "selection_filter": selection_filter or {},
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
        "summary": summary,
        "labels": labels,
    }
    out_lab.write_text(json.dumps(labels_doc, ensure_ascii=False, indent=2),
                       encoding="utf-8")
    out_meta.write_text(json.dumps({
        "ifc_id": out_id, "kind": "violated", "baseline_id": baseline_id,
        "pool_run_id": pool_run_id, "llm_model": model,
        "summary": summary,
        "created_at": labels_doc["created_at"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    status = "ok" if applied > 0 and skipped == 0 else ("partial" if applied > 0 else "invalid")

    graph_path: str | None = None
    try:
        from . import ifc_graph
        gp = out_dir / f"{out_id}.graph.json"
        ifc_graph.build_and_save(out_ifc, gp)
        graph_path = str(gp)
    except Exception:
        graph_path = None

    mid = storage.create_ifc_model(
        kind="violated", name=base["name"] + ".violated", parent_id=baseline_id,
        llm_model=model, prompt=None, pool_run_id=pool_run_id,
        params={"summary": summary, "selection_filter": selection_filter or {}},
        file_path=str(out_ifc), meta_path=str(out_meta), labels_path=str(out_lab),
        graph_path=graph_path, status=status, error=None,
    )
    storage.add_ifc_labels(mid, labels)
    return {
        "ifc_model_id": mid, "ifc_path": str(out_ifc),
        "labels_path": str(out_lab), "meta_path": str(out_meta),
        "summary": summary,
    }


def _label_base(v: dict) -> dict:
    return {
        "violation_id": v.get("id"),
        "title": v.get("title"),
        "category": v.get("category"),
        "severity": v.get("severity"),
        "threshold": v.get("threshold"),
        "evidence": v.get("evidence") or [],
    }


def pick_violations(pool_violations: list[dict], n: int,
                    category: str | None = None,
                    seed: int | None = None) -> list[dict]:
    pool = pool_violations
    if category:
        pool = [v for v in pool if (v.get("category") or "").lower() == category.lower()]
    if seed is not None:
        random.seed(seed)
    return random.sample(pool, min(n, len(pool)))
