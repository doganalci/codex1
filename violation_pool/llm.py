"""LLM client and JSON extraction helpers."""
from __future__ import annotations

import json
import re
from typing import Iterable

from openai import OpenAI

from . import storage
from .config import settings
from .prompts import NAIVE_PROMPT, OPTIMIZED_PROMPT, build_user_message


def _client() -> OpenAI:
    return OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def _extract_json(text: str) -> dict:
    """Extract first JSON object from a string (tolerant to code fences / prefixes)."""
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        return json.loads(fence.group(1))
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("LLM cevabında JSON bulunamadı")


def _normalize(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    for it in items or []:
        desc = (it.get("description") or "").strip()
        if not desc:
            continue
        out.append(
            {
                "title": (it.get("title") or "").strip() or None,
                "description": desc,
                "category": (it.get("category") or "").strip() or None,
                "severity": (it.get("severity") or "").strip() or None,
                "threshold": it.get("threshold") if it.get("threshold") not in ("", None) else None,
                "evidence": it.get("evidence") or [],
            }
        )
    return out


def _count_suffix(n: int, avoid: list[str] | None = None) -> str:
    s = f"\n\nTam olarak {n} adet ihlal üret."
    if avoid:
        # Sadece title bazlı kısa liste; çok uzun olmasın diye ilk 50 ile sınırla
        s += (
            "\nAşağıdaki ihlalleri TEKRARLAMA, bunlardan farklı olanları üret:\n- "
            + "\n- ".join(avoid[:50])
        )
    return s


def generate_naive(
    user_prompt: str,
    model: str | None = None,
    n: int = 20,
    avoid_titles: list[str] | None = None,
    usage_meta: dict | None = None,
) -> list[dict]:
    """Method 1: send the user prompt as-is, with a thin JSON instruction."""
    msg = (
        user_prompt.strip()
        + _count_suffix(n, avoid_titles)
        + '\n\nLütfen sonucu yalnızca şu JSON formatında ver: '
          '{"violations":[{"description":"..."}]}'
    )
    eff_model = model or settings.llm_model
    resp = _client().chat.completions.create(
        model=eff_model,
        messages=[{"role": "user", "content": msg}],
        temperature=0.4,
    )
    storage.record_usage_from_openai(
        getattr(resp, "usage", None), operation="gen_naive",
        model=eff_model, **(usage_meta or {}),
    )
    data = _extract_json(resp.choices[0].message.content or "")
    return _normalize(data.get("violations", []))


def generate_optimized(
    user_prompt: str,
    model: str | None = None,
    n: int = 20,
    avoid_titles: list[str] | None = None,
    usage_meta: dict | None = None,
    _operation: str = "gen_optimized",
) -> list[dict]:
    """Method 2: optimized system prompt + user prompt, no context."""
    user_msg = user_prompt.strip() + _count_suffix(n, avoid_titles)
    eff_model = model or settings.llm_model
    resp = _client().chat.completions.create(
        model=eff_model,
        messages=[
            {"role": "system", "content": OPTIMIZED_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    storage.record_usage_from_openai(
        getattr(resp, "usage", None), operation=_operation,
        model=eff_model, **(usage_meta or {}),
    )
    data = _extract_json(resp.choices[0].message.content or "")
    return _normalize(data.get("violations", []))


def generate_finetuned(
    user_prompt: str,
    ft_model_id: str,
    n: int = 20,
    avoid_titles: list[str] | None = None,
    usage_meta: dict | None = None,
) -> list[dict]:
    """Method 4: same optimized prompt as method 2, but call a fine-tuned model."""
    return generate_optimized(
        user_prompt, model=ft_model_id, n=n, avoid_titles=avoid_titles,
        usage_meta=usage_meta, _operation="gen_finetuned",
    )


def generate_rag(
    user_prompt: str,
    context_chunks: list[dict],
    model: str | None = None,
    n: int = 20,
    avoid_titles: list[str] | None = None,
    usage_meta: dict | None = None,
) -> list[dict]:
    """Method 3: same optimized prompt as method 2 + RAG context."""
    user_msg = build_user_message(user_prompt, context_chunks) + _count_suffix(n, avoid_titles)
    eff_model = model or settings.llm_model
    resp = _client().chat.completions.create(
        model=eff_model,
        messages=[
            {"role": "system", "content": OPTIMIZED_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    storage.record_usage_from_openai(
        getattr(resp, "usage", None), operation="gen_rag",
        model=eff_model, **(usage_meta or {}),
    )
    data = _extract_json(resp.choices[0].message.content or "")
    return _normalize(data.get("violations", []))


def default_prompt_for(method: str) -> str:
    from .config import METHOD_NAIVE
    return NAIVE_PROMPT if method == METHOD_NAIVE else (
        "Yapı denetim mevzuatı kapsamında ihlal kuralları üret. "
        "Erişilebilirlik, yangın güvenliği, statik, elektrik, mekanik "
        "alanlarını kapsayacak şekilde geniş bir liste hazırla."
    )
