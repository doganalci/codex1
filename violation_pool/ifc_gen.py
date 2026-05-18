"""IFC baseline üretimi (LLM tam STEP text yazar, ifcopenshell validate eder).

Akış:
  1) LLM'e IFC4 SPF (STEP) formatında minimal, geçerli, ihlal içermeyen bir
     ev modeli isteriz. Tüm ölçüler bilinçli olarak fazladır.
  2) ifcopenshell ile dosyayı parse etmeyi deneriz.
  3) Hatalıysa parse mesajını LLM'e geri verip 1 kez retry.
  4) Sonuçta ifc_path + meta_path döner; status='ok'|'invalid'|'partial'.

Not: LLM'in tam STEP üretmesi zordur — uzun ve hata-meyilli. Bu modülde
maliyet ve hata payı için tolerans var; retry başarısız olursa raw çıktı
yine kaydedilir, status='invalid' işaretlenir, kullanıcı UI'da görür.
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
- En az şu objeleri içersin: IfcProject, IfcSite, IfcBuilding, IfcBuildingStorey,
  IfcWallStandardCase (en az 4), IfcSlab (zemin), IfcDoor (en az 2),
  IfcWindow (en az 2), ilgili IfcLocalPlacement, IfcAxis2Placement3D,
  IfcCartesianPoint, IfcDirection, IfcOwnerHistory, IfcUnitAssignment.
- Tüm GUID'ler 22 karakter base64 (IfcGloballyUniqueId) olsun, birbirinden farklı.
- Eleman boyutları cömert olsun (KESİNLİKLE ihlal içermesin):
  * Kapı genişliği ≥ 1.00 m, yüksekliği ≥ 2.10 m
  * Pencere ≥ 1.20 × 1.20 m
  * Tavan yüksekliği ≥ 3.00 m
  * Duvar kalınlığı ≥ 0.20 m
  * Koridor genişliği ≥ 1.50 m (varsa)
- Geometri tutarlı olsun (placement zinciri, units 'METRE').

Sadece dosya içeriğini döndür."""


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


def _ask_llm(user_prompt: str, model: str, retry_error: str | None = None) -> str:
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
        max_tokens=12000,
    )
    return _strip_fences(resp.choices[0].message.content or "")


def generate_baseline(
    *,
    name: str,
    seed_prompt: str,
    model: str | None = None,
) -> dict:
    """Tek bir baseline IFC üret. Sonuç: storage'a kayıt + dosya yolları."""
    from . import storage  # circular avoidance

    model = model or settings.ifc_llm_model
    ifc_id = str(uuid.uuid4())
    out_dir = settings.ifc_dir / "baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    ifc_path = out_dir / f"{ifc_id}.ifc"
    meta_path = out_dir / f"{ifc_id}.meta.json"

    text = _ask_llm(seed_prompt, model)
    ifc_path.write_text(text, encoding="utf-8")
    ok, err = _try_open(ifc_path)
    if not ok:
        # one retry with parse error feedback
        text2 = _ask_llm(seed_prompt, model, retry_error=err)
        ifc_path.write_text(text2, encoding="utf-8")
        ok, err = _try_open(ifc_path)

    status = "ok" if ok else "invalid"
    meta = {
        "ifc_id": ifc_id,
        "name": name,
        "kind": "baseline",
        "llm_model": model,
        "prompt": seed_prompt,
        "status": status,
        "error": err,
        "created_at": datetime.utcnow().isoformat(timespec="seconds"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    mid = storage.create_ifc_model(
        kind="baseline", name=name, parent_id=None,
        llm_model=model, prompt=seed_prompt, pool_run_id=None,
        params=None, file_path=str(ifc_path), meta_path=str(meta_path),
        labels_path=None, status=status, error=err,
    )
    return {"ifc_model_id": mid, "ifc_path": str(ifc_path),
            "meta_path": str(meta_path), "status": status, "error": err}


def generate_baselines(
    *, n: int, seed_prompt: str, model: str | None = None,
    name_prefix: str = "House",
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
            name=f"{name_prefix}-{i+1:02d}", seed_prompt=prompt, model=model
        ))
    return results
