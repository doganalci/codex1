"""IFC baseline üretimi.

İki mod:
  - "parametric" (varsayılan, ÖNERİLEN): LLM yalnızca küçük bir JSON spec
    verir; geometriyi ifcopenshell ile programatik olarak biz inşa ederiz.
    Geçerli IFC garantilenir, ifcopenshell.geom her elemanı tessellate eder.
  - "raw": LLM doğrudan tam IFC4 STEP metni üretir (eski deneysel yol).
    gpt-4o-mini için güvenilir değildir; geometri çoğunlukla boş kalır.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .config import settings


IFC_SYSTEM_PROMPT = """Sen IFC4 (ISO 16739) STEP/SPF formatında geçerli bir IFC
dosyası üreten bir mühendissin. Çıktın TAM, eksiksiz bir .ifc dosyası olacak.

KESİN KURALLAR:
- Çıktı yalnızca SPF (STEP) metni olsun. Markdown kod bloğu ya da açıklama YAZMA.
- Dosya `ISO-10303-21;` ile başlasın, `END-ISO-10303-21;` ile bitsin.
- FILE_SCHEMA(('IFC4')) kullan.
- ZORUNLU ENTİTELER (eksiksiz olmalı, yoksa cevap REDDEDİLİR):
  * 1 IfcProject (Units: IfcUnitAssignment ile METRE/RADIAN)
  * 1 IfcSite
  * 1 IfcBuilding
  * 1+ IfcBuildingStorey
  * 1+ IfcSpace (oda)
  * IfcRelAggregates ile hiyerarşi: Project→Site→Building→Storey
  * IfcRelContainedInSpatialStructure ile Storey→fiziksel elemanlar
  * 4+ IfcWallStandardCase (her duvar IfcExtrudedAreaSolid ile geometriye sahip)
  * 1+ IfcSlab (zemin)
  * 2+ IfcDoor (her birinin IfcRelFillsElement ile bir IfcOpeningElement'i)
  * 2+ IfcWindow
  * IfcRelSpaceBoundary ile duvarların hangi Space'i sınırladığı
  * Her IfcProduct için IfcLocalPlacement (IfcAxis2Placement3D + IfcCartesianPoint + IfcDirection)
  * 1 IfcOwnerHistory
- Tüm GlobalId'ler 22 karakter benzersiz IfcGloballyUniqueId.
- Eleman boyutları KESİNLİKLE ihlal İÇERMESİN (cömert):
  * Kapı genişliği ≥ 1.00 m, yüksekliği ≥ 2.10 m
  * Pencere ≥ 1.20 × 1.20 m
  * Tavan yüksekliği ≥ 3.00 m
  * Duvar kalınlığı ≥ 0.20 m
  * Koridor genişliği ≥ 1.50 m (varsa)
- Geometri tutarlı (placement zinciri, units 'METRE').

Cevap olarak yalnızca dosya içeriğini döndür."""


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:ifc|step|spf)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def _try_open(ifc_path: Path) -> tuple[bool, str | None]:
    try:
        import ifcopenshell  # local import to avoid hard dep at import time
        f = ifcopenshell.open(str(ifc_path))
        # quick sanity
        if not f.by_type("IfcProject"):
            return False, "IfcProject bulunamadı"
        return True, None
    except Exception as e:
        return False, str(e)


def _ask_llm(user_prompt: str, model: str, retry_error: str | None = None,
             note: str | None = None, ifc_model_id: str | None = None) -> str:
    msgs = [{"role": "system", "content": IFC_SYSTEM_PROMPT}]
    if retry_error:
        msgs.append({
            "role": "user",
            "content": user_prompt
            + "\n\nÖnceki denemede parse hatası: " + retry_error
            + "\nDosyayı düzelt; yine sadece SPF metni döndür.",
        })
    else:
        msgs.append({"role": "user", "content": user_prompt})
    resp = _client().chat.completions.create(
        model=model,
        messages=msgs,
        temperature=0.4,
        max_tokens=16000,
    )
    from . import storage
    storage.record_usage_from_openai(
        getattr(resp, "usage", None),
        operation="ifc_gen", model=model,
        ifc_model_id=ifc_model_id,
        note=(note + (" [retry]" if retry_error else "")) if note else None,
    )
    return _strip_fences(resp.choices[0].message.content or "")


IFC_SPEC_PROMPT = """Sen bir mimar yardımcısısın. Sözel isteğe karşılık,
PARAMETRİK bir konut için JSON spec döndürürsün. Geometri programatik
olarak inşa edilecek; sen yalnızca spec verirsin (IFC YAZMA).

Şema:
{
  "name": "string",
  "storey_height": 3.0,         // metre, >= 3.0
  "wall_thickness": 0.20,        // metre, >= 0.20
  "rooms": [
    {"name": "string", "origin": [x, y], "size": [w, l]}
  ],
  "openings": [
    {"room": "string", "side": "south|north|east|west",
     "type": "door|window",
     "width": 1.0, "height": 2.2,
     "sill": 0.0, "offset": 1.2,
     "is_exterior": true,           // dış cephede mi?
     "name": "string"
    }
  ]
}

ZORUNLU İÇERİK:
- En az şu odalar bulunsun: Salon, Mutfak, Banyo (veya WC) ve 1-3 Yatak Odası.
  Opsiyonel: Koridor, Antre.
- HER odanın en az 1 kapısı olsun (oda↔dış mekan veya oda↔başka oda).
- TAM olarak 1 adet DIŞ KAPI (giriş) olsun ve is_exterior=true ile işaretlensin.
  Bu kapı bina çeperinde, dışarıya açılan bir duvarda olsun (komşu oda yok).
- HER odanın dışa bakan en az 1 duvarında pencere olsun (is_exterior=true).
- Mutfak ve banyoda mutlaka en az 1 pencere olsun.
- İç kapılar (oda↔oda arası) is_exterior=false.

DIŞ DUVAR TESPİTİ (önemli):
- Bir odanın bir kenarı (south/north/east/west) başka bir odanın kenarıyla
  TEMAS ETMİYORSA o kenar DIŞ duvardır.
- Önce odaları planla (origin+size çakışmasın, komşu odalar paylaşan
  kenarlara hizalı olsun); sonra her odanın dış kenarlarını belirle;
  pencereleri ve dış kapıyı yalnızca o kenarlara yerleştir.
- İç kapıları, iki odanın paylaştığı kenara (her iki odadan biri için)
  yerleştir; is_exterior=false.

ÖLÇÜLER (bilinçli olarak fazlasıyla mevzuata uygun, ihlal İÇERMESİN):
- Dış kapı: width >= 1.10 m, height >= 2.20 m, sill = 0.
- İç kapı: width >= 1.00 m, height >= 2.10 m, sill = 0.
- Pencere: width >= 1.20 m, height >= 1.20 m, sill 0.6 - 1.2 m.
- offset = duvar başlangıcından açıklığın başlangıcına metre; açıklık
  duvar uzunluğunu aşmasın.

KISITLAR:
- Tüm odalar dikdörtgen; origin/size metre cinsinden, çakışmasın.
- 3-6 oda arası ver.

Cevap olarak SADECE JSON döndür, başka metin yazma."""


def _ask_llm_spec(user_prompt: str, model: str,
                  ifc_model_id: str | None = None,
                  note: str | None = None) -> dict:
    resp = _client().chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": IFC_SPEC_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.6,
        response_format={"type": "json_object"},
    )
    from . import storage
    storage.record_usage_from_openai(
        getattr(resp, "usage", None),
        operation="ifc_gen_spec", model=model,
        ifc_model_id=ifc_model_id, note=note,
    )
    text = resp.choices[0].message.content or "{}"
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(m.group(0)) if m else {}


def generate_baseline(
    *,
    name: str,
    seed_prompt: str,
    model: str | None = None,
    mode: str = "parametric",  # "parametric" | "raw"
) -> dict:
    """Tek bir baseline IFC üret.

    mode='parametric' (varsayılan): LLM yalnızca JSON spec verir; geometri
    ifcopenshell ile inşa edilir. Geçerli IFC garantilenir.
    mode='raw': LLM tam IFC4 STEP üretir (eski yol; çoğunlukla geometri yok).
    """
    from . import storage  # circular avoidance
    from . import ifc_template

    model = model or settings.ifc_llm_model
    ifc_id = str(uuid.uuid4())
    out_dir = settings.ifc_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    ifc_path = out_dir / f"{ifc_id}.ifc"
    meta_path = out_dir / f"{ifc_id}.meta.json"

    status: str
    err: str | None = None
    spec: dict | None = None

    if mode == "parametric":
        try:
            spec = _ask_llm_spec(seed_prompt, model, ifc_model_id=ifc_id, note=name)
            if not spec or not spec.get("rooms"):
                spec = ifc_template.EXAMPLE_SPEC
                err = "LLM spec boş/eksik; fallback örnek spec kullanıldı"
            ifc_template.build_house_ifc(spec, ifc_path)
            ok, parse_err = _try_open(ifc_path)
            status = "ok" if ok else "invalid"
            if parse_err:
                err = (err + "; " if err else "") + parse_err
        except Exception as e:
            # Son çare: örnek spec ile en azından bir IFC üret
            try:
                ifc_template.build_house_ifc(ifc_template.EXAMPLE_SPEC, ifc_path)
                ok, parse_err = _try_open(ifc_path)
                status = "ok" if ok else "invalid"
                err = f"LLM/spec hatası ({e}); fallback örnek spec kullanıldı"
            except Exception as e2:
                status = "invalid"
                err = f"parametrik üretim başarısız: {e2}"
    else:
        # Eski "raw" yol — LLM tam IFC text yazsın
        text = _ask_llm(seed_prompt, model, note=name, ifc_model_id=ifc_id)
        ifc_path.write_text(text, encoding="utf-8")
        ok, err = _try_open(ifc_path)
        if not ok:
            text2 = _ask_llm(seed_prompt, model, retry_error=err, note=name,
                              ifc_model_id=ifc_id)
            ifc_path.write_text(text2, encoding="utf-8")
            ok, err = _try_open(ifc_path)
        status = "ok" if ok else "invalid"

    meta = {
        "ifc_id": ifc_id,
        "name": name,
        "kind": "baseline",
        "mode": mode,
        "llm_model": model,
        "prompt": seed_prompt,
        "spec": spec,
        "status": status,
        "error": err,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                         encoding="utf-8")

    graph_path: str | None = None
    if status == "ok":
        try:
            from . import ifc_graph
            gp = out_dir / f"{ifc_id}.graph.json"
            ifc_graph.build_and_save(ifc_path, gp)
            graph_path = str(gp)
        except Exception:
            graph_path = None

    mid = storage.create_ifc_model(
        id=ifc_id,
        kind="baseline", name=name, parent_id=None,
        llm_model=model, prompt=seed_prompt, pool_run_id=None,
        params={"mode": mode, "spec": spec},
        file_path=str(ifc_path), meta_path=str(meta_path),
        labels_path=None, graph_path=graph_path,
        status=status, error=err,
    )
    return {"ifc_model_id": mid, "ifc_path": str(ifc_path),
            "meta_path": str(meta_path), "graph_path": graph_path,
            "status": status, "error": err, "mode": mode}


def import_real_ifc(
    *,
    src_path: str | Path,
    name: str | None = None,
) -> dict:
    """Dışarıdan gerçek bir IFC dosyasını projeye al; graph ve meta üret;
    storage'a kind='imported' olarak kaydet.
    """
    from . import storage, ifc_graph as _ifc_graph
    import shutil

    src = Path(src_path)
    if not src.exists():
        raise FileNotFoundError(str(src))

    ifc_id = str(uuid.uuid4())
    out_dir = settings.ifc_dir / "imports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ifc_path = out_dir / f"{ifc_id}.ifc"
    shutil.copyfile(src, ifc_path)

    ok, err = _try_open(ifc_path)
    status = "ok" if ok else "invalid"

    graph_path = None
    if ok:
        try:
            gp = out_dir / f"{ifc_id}.graph.json"
            _ifc_graph.build_and_save(ifc_path, gp)
            graph_path = str(gp)
        except Exception:
            graph_path = None

    meta_path = out_dir / f"{ifc_id}.meta.json"
    meta = {
        "ifc_id": ifc_id,
        "name": name or src.name,
        "kind": "imported",
        "source_filename": src.name,
        "status": status,
        "error": err,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    mid = storage.create_ifc_model(
        id=ifc_id,
        kind="imported", name=(name or src.name), parent_id=None,
        llm_model="-", prompt=None, pool_run_id=None,
        params={"source_filename": src.name},
        file_path=str(ifc_path), meta_path=str(meta_path),
        labels_path=None, graph_path=graph_path,
        status=status, error=err,
    )
    return {
        "ifc_model_id": mid, "ifc_path": str(ifc_path),
        "meta_path": str(meta_path), "graph_path": graph_path,
        "status": status, "error": err,
    }


def generate_baselines(
    *, n: int, seed_prompt: str, model: str | None = None,
    name_prefix: str = "House",
    mode: str = "parametric",
) -> list[dict]:
    """N adet farklı baseline IFC üret. Çeşitlilik için her birine küçük bir
    varyasyon ipucu eklenir."""
    results = []
    variations = [
        "tek katlı, dikdörtgen plan, 3 oda + salon",
        "tek katlı, L şeklinde plan, 2 oda + salon + mutfak",
        "iki katlı, kare plan, alt kat salon+mutfak, üst kat 2 yatak odası",
        "tek katlı geniş plan, 4 oda + salon + 2 banyo",
        "iki katlı, dikdörtgen, üst katta balkon",
    ]
    for i in range(n):
        var = variations[i % len(variations)]
        prompt = (
            seed_prompt
            + f"\n\nBu örnek için varyasyon: {var}. Boyutlar yine cömert (ihlalsiz) olsun."
        )
        results.append(generate_baseline(
            name=f"{name_prefix}-{i+1:02d}", seed_prompt=prompt,
            model=model, mode=mode,
        ))
    return results
