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
katalogu ve bir 'ihlal kuralı' verilir. Görevin, ihlali oluşturacak EN UYGUN
tek bir EYLEM önermek.

İKİ EYLEM TÜRÜ VAR:

A) modify_attribute — Mevcut bir elemanın bir attribute'unu değiştir.
   (Örn: kapı genişliğini 70 cm'in altına indir.)
   Çıktı şeması:
   {
     "applicable": true,
     "action": "modify_attribute",
     "target_guid": "GUID",
     "ifc_type": "IfcDoor",
     "attribute": "OverallWidth",
     "new_value": 0.60,
     "rationale": "kısa açıklama"
   }

B) add_obstruction — Yeni bir IfcColumn (engel) ekleyerek geçişi/manevrayı
   engelle. Hedef elemanın (kapı / koridor / merdiven) önüne ya da içine
   bir kolon yerleştirilir. Eleman boyutu standartı sağlıyor olabilir ama
   kolon yüzünden işlevsel olarak ihlal oluşur.
   Çıktı şeması:
   {
     "applicable": true,
     "action": "add_obstruction",
     "reference_guid": "GUID",       // yanına/önüne konulacak eleman
     "ifc_type": "IfcDoor|IfcSpace|IfcStair|IfcWall",
     "obstruction_size": [0.30, 0.30, 2.50],  // x, y, z (m)
     "offset": 0.50,                  // referans elemandan mesafe (m)
     "rationale": "kısa açıklama"
   }

İHLAL TÜRÜNDEN EYLEM SEÇİMİ:
- "Kapı genişliği < X cm"             → modify_attribute (OverallWidth)
- "Kapı yüksekliği < X cm"            → modify_attribute (OverallHeight)
- "Tavan yüksekliği < X cm"           → modify_attribute (NominalHeight)
- "Pencere boyutu < X"                → modify_attribute
- "Asansör kabin boyutu < X"          → modify_attribute
- "Kapı önünde manevra alanı yetersiz" → add_obstruction (referans: kapı)
- "Koridorda engel/kolon"              → add_obstruction (referans: koridor/space)
- "Geçiş alanında sabit obje"          → add_obstruction (referans: kapı/space)
- "Merdiven başında engel"             → add_obstruction (referans: merdiven)
- "Rampa önünde engel"                 → add_obstruction (referans: rampa)

UYGULANAMAZSA:
{"applicable": false, "reason": "kısa neden"}

Sadece JSON döndür, başka hiçbir metin yazma."""


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


def _propose_edit(violation: dict, cat: list[dict], model: str,
                  usage_meta: dict | None = None) -> dict:
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
    storage.record_usage_from_openai(
        getattr(resp, "usage", None),
        operation="ifc_inject", model=model,
        note=(violation.get("title") or violation.get("id") or "")[:80],
        **(usage_meta or {}),
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


# ----- Obstruction (kolon) ekleme -----
def _world_xy_of(element) -> tuple[float, float]:
    """IfcProduct'ın yaklaşık world XY konumu (placement zinciri toplanır)."""
    x = y = 0.0
    placement = getattr(element, "ObjectPlacement", None)
    while placement is not None:
        rel = getattr(placement, "RelativePlacement", None)
        if rel is not None:
            loc = getattr(rel, "Location", None)
            if loc is not None and getattr(loc, "Coordinates", None):
                cc = loc.Coordinates
                x += cc[0] if len(cc) > 0 else 0.0
                y += cc[1] if len(cc) > 1 else 0.0
        placement = getattr(placement, "PlacementRelTo", None)
    return float(x), float(y)


def _ref_dir_xy_of(element) -> tuple[float, float]:
    placement = getattr(element, "ObjectPlacement", None)
    if not placement:
        return (1.0, 0.0)
    rel = getattr(placement, "RelativePlacement", None)
    if not rel:
        return (1.0, 0.0)
    ref = getattr(rel, "RefDirection", None)
    if not ref:
        return (1.0, 0.0)
    r = ref.DirectionRatios
    dx = r[0] if len(r) > 0 else 1.0
    dy = r[1] if len(r) > 1 else 0.0
    n = (dx * dx + dy * dy) ** 0.5 or 1.0
    return (dx / n, dy / n)


def _get_context(ifc_file):
    """Mevcut IFC dosyasından owner/body_ctx/storey objelerini topla."""
    import ifcopenshell  # noqa
    owners = ifc_file.by_type("IfcOwnerHistory")
    storeys = ifc_file.by_type("IfcBuildingStorey")
    sub = [c for c in ifc_file.by_type("IfcGeometricRepresentationSubContext")
           if getattr(c, "ContextIdentifier", "") == "Body"]
    if not sub:
        sub = ifc_file.by_type("IfcGeometricRepresentationContext")
    if not (owners and storeys and sub):
        return None, None, None
    return owners[0], sub[0], storeys[0]


def _add_column_obstruction(
    ifc_file, owner, body_ctx, storey,
    world_x: float, world_y: float,
    size_x: float = 0.30, size_y: float = 0.30, height: float = 2.50,
    name: str = "Obstruction Column",
):
    """world (x,y) konumuna IfcColumn ekler; storey'e contain edilir."""
    import ifcopenshell.guid as _gid

    def _pt(x, y, z=0.0):
        return ifc_file.create_entity("IfcCartesianPoint",
                                      Coordinates=(float(x), float(y), float(z)))

    def _dir(x, y, z):
        return ifc_file.create_entity("IfcDirection",
                                      DirectionRatios=(float(x), float(y), float(z)))

    def _axis(loc, z=(0, 0, 1), x=(1, 0, 0)):
        return ifc_file.create_entity("IfcAxis2Placement3D",
                                      Location=loc,
                                      Axis=_dir(*z),
                                      RefDirection=_dir(*x))

    placement = ifc_file.create_entity(
        "IfcLocalPlacement",
        PlacementRelTo=storey.ObjectPlacement,
        RelativePlacement=_axis(_pt(world_x, world_y, 0.0)),
    )
    profile = ifc_file.create_entity(
        "IfcRectangleProfileDef",
        ProfileType="AREA", XDim=float(size_x), YDim=float(size_y),
    )
    extrude_axis = _axis(_pt(0, 0, 0))
    solid = ifc_file.create_entity(
        "IfcExtrudedAreaSolid",
        SweptArea=profile, Position=extrude_axis,
        ExtrudedDirection=_dir(0, 0, 1), Depth=float(height),
    )
    rep = ifc_file.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=body_ctx,
        RepresentationIdentifier="Body",
        RepresentationType="SweptSolid",
        Items=[solid],
    )
    shape = ifc_file.create_entity("IfcProductDefinitionShape",
                                   Representations=[rep])
    column = ifc_file.create_entity(
        "IfcColumn",
        GlobalId=_gid.new(), OwnerHistory=owner, Name=name,
        ObjectPlacement=placement, Representation=shape,
    )
    # Storey'e contain et (varsa mevcut Rel'e ekle, yoksa yeni Rel yarat)
    contained = False
    for r in ifc_file.by_type("IfcRelContainedInSpatialStructure"):
        if r.RelatingStructure == storey:
            related = list(r.RelatedElements or ())
            related.append(column)
            r.RelatedElements = tuple(related)
            contained = True
            break
    if not contained:
        ifc_file.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=_gid.new(), OwnerHistory=owner,
            RelatingStructure=storey, RelatedElements=[column],
        )
    return column


def _apply_add_obstruction(ifc_file, sug: dict) -> dict:
    """sug: add_obstruction önerisi. Yeni kolon ekler ve label bilgisini döner."""
    owner, body_ctx, storey = _get_context(ifc_file)
    if not (owner and body_ctx and storey):
        raise ValueError("IFC bağlam objeleri eksik (owner/body_ctx/storey)")
    ref_guid = sug.get("reference_guid") or sug.get("target_guid")
    ref = ifc_file.by_guid(ref_guid) if ref_guid else None
    if ref is None:
        raise ValueError(f"Referans GUID bulunamadı: {ref_guid}")

    ref_x, ref_y = _world_xy_of(ref)
    dx, dy = _ref_dir_xy_of(ref)
    # Kapı için: dik yönü hesapla (perpendicular). Yoksa kendi referans yönünde.
    nx, ny = -dy, dx
    offset = float(sug.get("offset", 0.5) or 0.5)
    cx = ref_x + nx * offset
    cy = ref_y + ny * offset

    sz = sug.get("obstruction_size") or [0.30, 0.30, 2.50]
    sx = float(sz[0]) if len(sz) > 0 else 0.30
    sy = float(sz[1]) if len(sz) > 1 else 0.30
    sh = float(sz[2]) if len(sz) > 2 else 2.50

    name = f"Obstacle [auto] near {getattr(ref, 'Name', None) or ref.is_a()}"
    column = _add_column_obstruction(
        ifc_file, owner, body_ctx, storey, cx, cy,
        size_x=sx, size_y=sy, height=sh, name=name,
    )
    return {
        "ifc_global_id": column.GlobalId,
        "ifc_type": "IfcColumn",
        "ifc_name": name,
        "attribute": "[ADDED]",
        "value_before": None,
        "value_after": f"IfcColumn size=({sx},{sy},{sh}) at ({cx:.2f},{cy:.2f}) "
                        f"offset={offset} near {ref.GlobalId}",
    }


def inject_violations(
    *,
    baseline_id: str,
    violations: list[dict],
    pool_run_id: str | None,
    model: str | None = None,
    selection_filter: dict | None = None,
    decoy_ratio: float = 0.20,
    decoy_seed: int | None = None,
) -> dict:
    """violations: havuzdan seçilmiş ihlal dict'leri (storage.get_violations
    çıktısı formatı).
    decoy_ratio: gerçek ihlal sayısının yüzdesi kadar SAHTE (decoy) etiket
        eklenir. IFC modifiye edilmez; etiket "ihlal değil" olarak işaretlenir.
        Test amaçlı (downstream sistem decoy'u gerçek ihlalden ayırabiliyor mu?).
    """
    import ifcopenshell  # local import
    import random

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
    inject_meta_ifc_id = out_id

    labels: list[dict] = []
    applied = skipped = 0
    inject_meta = {"pool_run_id": pool_run_id, "ifc_model_id": inject_meta_ifc_id}
    for v in violations:
        try:
            sug = _propose_edit(v, cat, model, usage_meta=inject_meta)
        except Exception as e:
            labels.append({**_label_base(v), "status": "skipped",
                           "reason": f"LLM hata: {e}", "is_decoy": False,
                           "applied_at": datetime.utcnow().isoformat(timespec="seconds")})
            skipped += 1
            continue

        if not sug.get("applicable"):
            labels.append({**_label_base(v), "status": "skipped",
                           "reason": sug.get("reason") or "uygulanabilir hedef yok",
                           "is_decoy": False,
                           "action": sug.get("action") or "modify_attribute",
                           "applied_at": datetime.utcnow().isoformat(timespec="seconds")})
            skipped += 1
            continue

        action = sug.get("action") or "modify_attribute"
        try:
            if action == "add_obstruction":
                info = _apply_add_obstruction(src, sug)
                lbl_extra = info
            else:
                # default: modify_attribute
                before, after = _apply_edit(src, sug["target_guid"],
                                             sug["attribute"], sug["new_value"])
                target = src.by_guid(sug["target_guid"])
                lbl_extra = {
                    "ifc_global_id": sug["target_guid"],
                    "ifc_type": sug.get("ifc_type") or target.is_a(),
                    "ifc_name": getattr(target, "Name", None),
                    "attribute": sug["attribute"],
                    "value_before": before,
                    "value_after": after,
                }
        except Exception as e:
            labels.append({**_label_base(v), "status": "skipped",
                           "reason": f"uygulama hatası: {e}",
                           "is_decoy": False, "action": action,
                           "applied_at": datetime.utcnow().isoformat(timespec="seconds")})
            skipped += 1
            continue

        labels.append({
            **_label_base(v),
            **lbl_extra,
            "status": "applied",
            "is_decoy": False,
            "action": action,
            "reason": sug.get("rationale"),
            "applied_at": datetime.utcnow().isoformat(timespec="seconds"),
        })
        applied += 1

    # ----- Decoys (sahte ihlaller) -----
    decoys_added = 0
    decoy_ratio = max(0.0, float(decoy_ratio or 0.0))
    n_decoys_target = int(round(len(violations) * decoy_ratio))
    if n_decoys_target > 0 and cat:
        rng = random.Random(decoy_seed)
        used = {l.get("ifc_global_id") for l in labels
                if l.get("status") == "applied" and l.get("ifc_global_id")}
        cand = [c for c in cat if c.get("guid") and c["guid"] not in used
                # Sadece görsel olarak anlamlı eleman tipleri
                and c["type"] in ("IfcDoor", "IfcWindow", "IfcWall",
                                  "IfcWallStandardCase", "IfcSlab",
                                  "IfcStair", "IfcRailing", "IfcRamp")]
        rng.shuffle(cand)
        for c in cand[:n_decoys_target]:
            labels.append({
                "violation_id": None,
                "title": f"[DECOY] Sahte etiket: {c['type']}",
                "category": "Decoy",
                "severity": None,
                "threshold": None,
                "evidence": [],
                "ifc_global_id": c["guid"],
                "ifc_type": c["type"],
                "ifc_name": c.get("name"),
                "attribute": None,
                "value_before": None,
                "value_after": None,
                "status": "decoy",
                "is_decoy": True,
                "action": "decoy",
                "reason": "Sahte (honeypot) etiket — gerçek ihlal değil; "
                          "test için yerleştirildi.",
                "applied_at": datetime.utcnow().isoformat(timespec="seconds"),
            })
            decoys_added += 1

    src.write(str(out_ifc))

    summary = {"requested": len(violations), "applied": applied,
               "skipped": skipped, "decoys": decoys_added,
               "decoy_ratio": decoy_ratio}
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
        id=out_id,
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
