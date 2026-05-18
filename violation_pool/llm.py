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


def _extract_json_robust(text: str) -> dict:
    """LLM cevabı `max_tokens` ile kesilirse, son tamamlanan obje'ye kadar
    olan kısmı kurtarmaya çalışır."""
    try:
        return _extract_json(text)
    except (json.JSONDecodeError, ValueError):
        pass
    # `"violations": [` arrayini bul, son komplet `}` 'ye kadar al, kapatıp parse et
    m = re.search(r'"violations"\s*:\s*\[', text)
    if not m:
        raise ValueError("Kesik JSON: 'violations' bulunamadı")
    start = m.end()
    depth = 0
    in_string = False
    escape = False
    last_complete_end = -1
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                last_complete_end = i + 1
        elif ch == "]" and depth == 0:
            break
    if last_complete_end < 0:
        raise ValueError("Kesik JSON: tamamlanmış violation objesi yok")
    repaired = text[:last_complete_end] + "]}"
    return json.loads(repaired)


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


def _count_suffix(n: int, avoid: list[str] | None = None,
                  avoid_recent: int = 80) -> str:
    s = f"\n\nTam olarak {n} adet ihlal üret."
    if avoid:
        # Son `avoid_recent` ihlali yasakla — en yeni başlıklar tekrar etmesin diye
        tail = avoid[-avoid_recent:] if len(avoid) > avoid_recent else avoid
        s += (
            "\nAşağıdaki ihlalleri TEKRARLAMA, bunlardan farklı olanları üret:\n- "
            + "\n- ".join(tail)
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
        max_tokens=16000,
    )
    storage.record_usage_from_openai(
        getattr(resp, "usage", None), operation="gen_naive",
        model=eff_model, **(usage_meta or {}),
    )
    data = _extract_json_robust(resp.choices[0].message.content or "")
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
        max_tokens=16000,
    )
    storage.record_usage_from_openai(
        getattr(resp, "usage", None), operation=_operation,
        model=eff_model, **(usage_meta or {}),
    )
    data = _extract_json_robust(resp.choices[0].message.content or "")
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
        max_tokens=16000,
    )
    storage.record_usage_from_openai(
        getattr(resp, "usage", None), operation="gen_rag",
        model=eff_model, **(usage_meta or {}),
    )
    data = _extract_json_robust(resp.choices[0].message.content or "")
    return _normalize(data.get("violations", []))


CHUNK_SIZE = 40


def generate_chunked(
    method: str,
    user_prompt: str,
    *,
    model: str | None = None,
    n: int = 20,
    avoid_titles: list[str] | None = None,
    usage_meta: dict | None = None,
    context_chunks: list[dict] | None = None,
    ft_model_id: str | None = None,
    chunk_size: int = CHUNK_SIZE,
    progress_callback=None,
) -> list[dict]:
    """N büyükse birden çok çağrıda üretir. Aralarda gelen başlıkları
    avoid_titles'a eklemeye devam eder; sonraki partilerde tekrar
    üretilmemesi için."""
    from .config import METHOD_NAIVE, METHOD_OPTIMIZED, METHOD_RAG, METHOD_FINETUNE

    accumulated: list[dict] = []
    avoid = list(avoid_titles or [])
    remaining = int(n)
    chunk_size = max(1, int(chunk_size))
    total_chunks = (remaining + chunk_size - 1) // chunk_size
    chunk_idx = 0
    while remaining > 0:
        batch = min(chunk_size, remaining)
        chunk_idx += 1
        if progress_callback:
            try:
                progress_callback(chunk_idx, total_chunks, batch)
            except Exception:
                pass
        if method == METHOD_NAIVE:
            items = generate_naive(user_prompt, model=model, n=batch,
                                   avoid_titles=avoid, usage_meta=usage_meta)
        elif method == METHOD_OPTIMIZED:
            items = generate_optimized(user_prompt, model=model, n=batch,
                                        avoid_titles=avoid, usage_meta=usage_meta)
        elif method == METHOD_RAG:
            items = generate_rag(user_prompt, context_chunks or [], model=model,
                                  n=batch, avoid_titles=avoid, usage_meta=usage_meta)
        elif method == METHOD_FINETUNE:
            if not ft_model_id:
                raise ValueError("ft_model_id gerekli")
            items = generate_finetuned(user_prompt, ft_model_id=ft_model_id,
                                        n=batch, avoid_titles=avoid,
                                        usage_meta=usage_meta)
        else:
            raise ValueError(f"Bilinmeyen yöntem: {method}")
        if not items:
            break  # boş cevap → erken çık (sonsuz döngüye girme)
        accumulated.extend(items)
        # Yeni başlıkları avoid'a ekle (next partide tekrar gelmesin)
        for it in items:
            t = (it.get("title") or it.get("description") or "").strip()
            if t and t not in avoid:
                avoid.append(t)
        remaining -= batch
    return accumulated


def default_prompt_for(method: str) -> str:
    from .config import METHOD_NAIVE
    return NAIVE_PROMPT if method == METHOD_NAIVE else (
        "Yapılı çevrede erişilebilirlik ve kullanılabilirlik ihlali kuralları "
        "üret. Kapı/koridor, rampa, merdiven, korkuluk/küpeşte, asansör, "
        "tuvalet/banyo, otopark, uyarı yüzeyleri, yönlendirme/işaretleme, "
        "kontrast ve aydınlatma, manevra alanları, eşik/kot farkları gibi "
        "alanları geniş biçimde kapsa."
    )
