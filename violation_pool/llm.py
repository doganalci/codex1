"""LLM client and JSON extraction helpers."""
from __future__ import annotations

import json
import re
from typing import Iterable

from openai import OpenAI

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


def generate_naive(user_prompt: str, model: str | None = None) -> list[dict]:
    """Method 1: send the user prompt as-is, with a thin JSON instruction.

    Kullanıcının yazdığı promt korunur; sadece çıktıyı parse edebilmek için
    minimum bir JSON yönergesi eklenir.
    """
    msg = (
        user_prompt.strip()
        + '\n\nLütfen sonucu yalnızca şu JSON formatında ver: {"violations":[{"description":"..."}]}'
    )
    resp = _client().chat.completions.create(
        model=model or settings.llm_model,
        messages=[{"role": "user", "content": msg}],
        temperature=0.3,
    )
    data = _extract_json(resp.choices[0].message.content or "")
    return _normalize(data.get("violations", []))


def generate_optimized(user_prompt: str, model: str | None = None) -> list[dict]:
    """Method 2: optimized system prompt + user prompt, no context."""
    resp = _client().chat.completions.create(
        model=model or settings.llm_model,
        messages=[
            {"role": "system", "content": OPTIMIZED_PROMPT},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    data = _extract_json(resp.choices[0].message.content or "")
    return _normalize(data.get("violations", []))


def generate_rag(user_prompt: str, context_chunks: list[dict], model: str | None = None) -> list[dict]:
    """Method 3: same optimized prompt as method 2 + RAG context."""
    user_msg = build_user_message(user_prompt, context_chunks)
    resp = _client().chat.completions.create(
        model=model or settings.llm_model,
        messages=[
            {"role": "system", "content": OPTIMIZED_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
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
