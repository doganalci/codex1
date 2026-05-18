"""Otomatik dataset üretim pipeline'ı.

Tek bir çağrıda:
  - N baseline IFC üretir (parametrik, varyasyonlu)
  - Her baseline için M ihlalli varyant üretir (her varyant farklı
    tohum + farklı rastgele ihlal seçimi)
  - Ortalama K ihlal + decoy_ratio kadar decoy etiketler

Toplam üretilen ihlalli IFC = N × M.
İlerleme `progress_callback(phase, current, total, message)` ile rapor edilir.
"""
from __future__ import annotations

import random
from typing import Callable

from . import ifc_gen, ifc_inject, storage


ProgressCB = Callable[[str, int, int, str], None]


def estimate_tokens(
    *, n_baselines: int, variants_per_baseline: int,
    violations_per_variant: int,
) -> dict:
    """Çok kaba bir tahmin (gpt-4o-mini fiyatlarıyla)."""
    spec_per_baseline = 1500          # input + output
    propose_per_violation = 1500      # her _propose_edit çağrısı
    n_inj = n_baselines * variants_per_baseline * violations_per_variant
    total = n_baselines * spec_per_baseline + n_inj * propose_per_violation
    return {
        "estimated_total_tokens": total,
        "llm_calls": n_baselines + n_inj,
        "violated_ifcs": n_baselines * variants_per_baseline,
    }


def run_pipeline(
    *,
    pool_run_id: str,
    n_baselines: int = 4,
    variants_per_baseline: int = 3,
    violations_per_variant: int = 10,
    decoy_ratio: float = 0.20,
    baseline_seed_prompt: str = (
        "Erişilebilirlik ve kullanılabilirlik açısından sorunsuz, "
        "mevzuata fazlasıyla uygun küçük bir konut spec'i üret."
    ),
    baseline_variations: list[str] | None = None,
    baseline_mode: str = "parametric",
    ifc_model: str | None = None,
    inject_model: str | None = None,
    fill_from_pool: bool = True,
    progress_callback: ProgressCB | None = None,
    name_prefix: str = "Auto",
    existing_baseline_ids: list[str] | None = None,
) -> dict:
    """Tüm uçtan uca dataset oluşturma.

    existing_baseline_ids verilirse yeni baseline üretmez; mevcut ID'leri
    kullanır (n_baselines yok sayılır).
    """
    def _emit(phase: str, cur: int, total: int, msg: str = "") -> None:
        if progress_callback:
            progress_callback(phase, cur, total, msg)

    pool_vs = storage.get_violations(pool_run_id)
    if not pool_vs:
        raise ValueError("Boş havuz")

    results: dict = {
        "baselines": [],
        "variated": [],
        "errors": [],
        "summary": {},
    }

    # ---- 1) Baseline aşaması ----
    baseline_ids: list[str] = []
    if existing_baseline_ids:
        # mevcut baseline'lar
        for bid in existing_baseline_ids:
            m = storage.get_ifc_model(bid)
            if not m or m["status"] != "ok":
                results["errors"].append({"phase": "baseline_load", "id": bid,
                                          "error": "bulunamadı veya invalid"})
                continue
            baseline_ids.append(bid)
            results["baselines"].append({"ifc_model_id": bid,
                                          "status": m["status"],
                                          "name": m["name"],
                                          "reused": True})
            _emit("baseline_reuse", len(baseline_ids), len(existing_baseline_ids),
                  m["name"])
    else:
        variations = baseline_variations or ifc_gen.DEFAULT_VARIATIONS
        for i in range(n_baselines):
            _emit("baseline", i + 1, n_baselines, f"{name_prefix}-{i+1:03d}")
            try:
                brief = variations[i % len(variations)] if variations else None
                r = ifc_gen.generate_baseline(
                    name=f"{name_prefix}-{i+1:03d}",
                    seed_prompt=baseline_seed_prompt,
                    model=ifc_model, mode=baseline_mode,
                    variation_brief=brief,
                )
                results["baselines"].append({**r, "reused": False})
                if r["status"] == "ok":
                    baseline_ids.append(r["ifc_model_id"])
                else:
                    results["errors"].append({"phase": "baseline", "i": i,
                                              "error": r.get("error")})
            except Exception as e:
                results["errors"].append({"phase": "baseline", "i": i,
                                          "error": str(e)})

    if not baseline_ids:
        results["summary"] = {"baselines_ok": 0, "variated": 0,
                              "errors": len(results["errors"])}
        return results

    # ---- 2) Enjeksiyon aşaması: her baseline × M varyant ----
    total_inj = len(baseline_ids) * variants_per_baseline
    done = 0
    for bi, bid in enumerate(baseline_ids):
        for vi in range(variants_per_baseline):
            done += 1
            seed = bi * 1000 + vi * 37 + 1
            try:
                rng = random.Random(seed)
                picked = rng.sample(
                    pool_vs, min(violations_per_variant, len(pool_vs))
                )
                _emit("inject", done, total_inj,
                      f"baseline {bi+1}/{len(baseline_ids)} varyant {vi+1}/{variants_per_baseline}")
                out = ifc_inject.inject_violations(
                    baseline_id=bid, violations=picked,
                    pool_run_id=pool_run_id,
                    model=inject_model,
                    decoy_ratio=decoy_ratio,
                    decoy_seed=seed,
                    fill_from_pool=fill_from_pool,
                    selection_filter={
                        "auto_pipeline": True,
                        "baseline_idx": bi, "variant": vi,
                        "violations_per_variant": violations_per_variant,
                    },
                )
                results["variated"].append(out)
            except Exception as e:
                results["errors"].append({"phase": "inject",
                                          "baseline_id": bid,
                                          "variant": vi,
                                          "error": str(e)})

    results["summary"] = {
        "baselines_ok": len(baseline_ids),
        "variated": len(results["variated"]),
        "errors": len(results["errors"]),
        "total_applied": sum(o["summary"]["applied"] for o in results["variated"]),
        "total_skipped": sum(o["summary"]["skipped"] for o in results["variated"]),
        "total_decoys": sum(o["summary"]["decoys"] for o in results["variated"]),
        "total_replaced_from_pool": sum(
            o["summary"].get("replaced_from_pool", 0) for o in results["variated"]
        ),
    }
    return results
