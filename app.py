"""Violation Pool Builder — Streamlit UI.

Üç yöntemle ihlal havuzu üretir:
  1) Naive prompt          (LLM, ek işlem yok)
  2) Optimize prompt       (LLM)
  3) RAG + optimize prompt (doküman bazlı)

Yöntem 2 ve 3 aynı (kullanıcının girdiği) promtu kullanır — modeller arası
adil karşılaştırma için kritik.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from violation_pool import excel_export, finetune, ifc_gen, ifc_inject, llm, rag, storage
from violation_pool.config import (
    METHOD_FINETUNE,
    METHOD_LABELS,
    METHOD_NAIVE,
    METHOD_OPTIMIZED,
    METHOD_RAG,
    settings,
)
from violation_pool.prompts import OPTIMIZED_PROMPT


st.set_page_config(page_title="İhlal Havuzu", layout="wide")
storage.init_db()


# ---------- helpers ----------
def _violations_df(items: list[dict]) -> pd.DataFrame:
    rows = []
    for v in items:
        ev = v.get("evidence") or []
        rows.append(
            {
                "batch": v.get("batch_no"),
                "title": v.get("title"),
                "description": v.get("description"),
                "category": v.get("category"),
                "severity": v.get("severity"),
                "threshold": v.get("threshold"),
                "evidence": "; ".join(
                    f"{(e.get('document') or '?')}:p{e.get('page') or '?'}"
                    for e in ev
                )
                if ev
                else "",
                "created_at": v.get("created_at"),
            }
        )
    return pd.DataFrame(rows)


def _reset_draft():
    for k in ("draft_run_id", "draft_items", "draft_method", "draft_is_append"):
        st.session_state.pop(k, None)


# ---------- sidebar: model/embedding/collection ----------
st.sidebar.header("Ayarlar")
st.sidebar.text_input("LLM model", key="llm_model", value=settings.llm_model)
st.sidebar.text_input("Embedding model", key="embedding_model", value=settings.embedding_model)
st.sidebar.caption(f"API base: {settings.openai_base_url}")
if not settings.openai_api_key:
    st.sidebar.warning("OPENAI_API_KEY tanımlı değil (.env).")

st.sidebar.divider()
st.sidebar.subheader("Geçmiş Havuzlar")
runs = storage.list_runs()
if runs:
    for r in runs[:20]:
        with st.sidebar.expander(
            f"[{r['status']}] {r['name']} · {METHOD_LABELS.get(r['method'], r['method'])}",
            expanded=False,
        ):
            st.write(f"id: `{r['id'][:8]}`")
            st.write(f"model: `{r['llm_model']}`")
            if r.get("rag_collection"):
                st.write(f"collection: `{r['rag_collection']}`")
            st.write(f"updated: {r['updated_at']}")
            cols = st.columns(3)
            if cols[0].button("Aç", key=f"open_{r['id']}"):
                st.session_state["view_run_id"] = r["id"]
            if cols[1].button("Excel", key=f"xls_{r['id']}"):
                out = excel_export.export_run(r["id"])
                st.success(f"Yazıldı: {out}")
            if cols[2].button("Sil", key=f"del_{r['id']}"):
                storage.delete_run(r["id"])
                st.rerun()
else:
    st.sidebar.caption("Henüz havuz yok.")


# ---------- main ----------
st.title("İhlal Havuzu & IFC Stüdyosu")

top_pool, top_ifc = st.tabs(["İhlal Havuzu Oluşturma", "İhlalli IFC Oluşturma"])

with top_pool:
    tab_run, tab_view, tab_rag, tab_ft = st.tabs(
        ["Çalıştır", "Havuzu Görüntüle", "RAG / Doküman", "Fine-tune"]
    )

# =========================================================
# RAG management tab
# =========================================================
with tab_rag:
    st.subheader("Doküman yükleme & vektör koleksiyonu")
    st.caption(
        "Yüklenen PDF/TXT belgeleri, verdiğin **koleksiyon adı** altında "
        "vektör veritabanına (Chroma) kalıcı olarak kaydedilir. Aynı adı "
        "tekrar verirsen üzerine ekleme yapılır."
    )

    cols = st.columns([2, 1])
    coll_name = cols[0].text_input(
        "Koleksiyon adı", value="default", help="Aynı adı tekrar kullanırsan ekleme yapılır."
    )
    existing = rag.list_collections()
    cols[1].metric("Mevcut koleksiyon", len(existing))
    if existing:
        st.caption("Var olanlar: " + ", ".join(existing))

    uploads = st.file_uploader(
        "PDF veya TXT dosyaları", type=["pdf", "txt"], accept_multiple_files=True
    )
    if st.button("Yükle & vektörle", disabled=not uploads):
        paths: list[Path] = []
        for uf in uploads:
            p = settings.data_dir / uf.name
            p.write_bytes(uf.getbuffer())
            paths.append(p)
        with st.spinner("Belgeler ayrıştırılıyor ve embedding hesaplanıyor..."):
            info = rag.ingest_documents(coll_name.strip(), paths)
        st.success(
            f"`{info['collection']}` koleksiyonuna {info['chunks_added']} parça "
            f"eklendi (belgeler: {', '.join(info['documents']) or '-'})."
        )

    st.divider()
    sel = st.selectbox("Koleksiyon bilgisi", [""] + existing)
    if sel:
        st.json(rag.collection_info(sel))


# =========================================================
# Run tab
# =========================================================
with tab_run:
    st.subheader("Çalıştırma")

    mode = st.radio(
        "Mod",
        ["Yeni havuz", "Mevcut havuza ekle"],
        horizontal=True,
        key="run_mode",
    )

    append_run: dict | None = None
    if mode == "Mevcut havuza ekle":
        saved = [r for r in storage.list_runs() if r["status"] == "saved"]
        if not saved:
            st.warning("Henüz kaydedilmiş havuz yok.")
            st.stop()
        opt = {
            r["id"]: f"{r['name']} · {METHOD_LABELS.get(r['method'], r['method'])} · "
                     f"{r['llm_model']} · {r['updated_at']}"
            for r in saved
        }
        sel = st.selectbox(
            "Devam edilecek havuz", list(opt.keys()), format_func=lambda k: opt[k]
        )
        append_run = storage.get_run(sel)
        st.caption(
            "Konfigürasyon (yöntem, promt, model, koleksiyon, FT model) bu havuzdan "
            "alınır; yeni ihlaller aynı Excel dosyasına eklenir."
        )
        with st.expander("Bu havuzun konfigürasyonu", expanded=False):
            st.json({k: append_run.get(k) for k in (
                "method", "llm_model", "embedding_model", "rag_collection",
                "finetune_model_id", "prompt"
            )})

    if append_run:
        method = append_run["method"]
        name = append_run["name"]
        st.info(
            f"Yöntem: **{METHOD_LABELS.get(method, method)}**  ·  "
            f"havuz: **{name}** (`{append_run['id'][:8]}`)"
        )
    else:
        method = st.radio(
            "Yöntem",
            [METHOD_NAIVE, METHOD_OPTIMIZED, METHOD_RAG, METHOD_FINETUNE],
            format_func=lambda m: METHOD_LABELS[m],
            horizontal=False,
        )
        name = st.text_input("Çalıştırma adı", value="run-1")

    n_violations = st.number_input(
        "Üretilecek ihlal sayısı", min_value=1, max_value=200, value=20, step=1
    )

    # Prompt: Method 1 has its own; Methods 2, 3 and 4 share the same prompt
    # to allow fair comparison across models.
    if append_run:
        prompt = append_run["prompt"]
        with st.expander("Kullanılan promt (kilitli)", expanded=False):
            st.code(prompt)
    elif method == METHOD_NAIVE:
        st.caption("Yöntem 1: senin yazdığın promt LLM'e olduğu gibi gider.")
        prompt = st.text_area(
            "Promt (Yöntem 1)",
            value=llm.default_prompt_for(METHOD_NAIVE),
            height=160,
            key="prompt_naive",
        )
    else:
        st.caption(
            "Yöntem 2, 3 ve 4 **aynı promtu** kullanır. Promtu burada bir kez yaz; "
            "üç yöntemde de aynısı kullanılır (model karşılaştırması adil olsun)."
        )
        prompt = st.text_area(
            "Promt (Yöntem 2 & 3 & 4 ortak)",
            value=st.session_state.get("shared_prompt", llm.default_prompt_for(METHOD_OPTIMIZED)),
            height=160,
            key="shared_prompt",
        )
        with st.expander("Sistem promtu (sabit, optimize edilmiş)"):
            st.code(OPTIMIZED_PROMPT, language="markdown")

    rag_collection = None
    top_k = 8
    ft_model_id = None
    if method == METHOD_RAG:
        if append_run:
            rag_collection = append_run.get("rag_collection")
            st.info(f"Koleksiyon: `{rag_collection}` (kilitli)")
        else:
            colls = rag.list_collections()
            if not colls:
                st.warning("RAG için önce 'RAG / Doküman' sekmesinden bir koleksiyon oluştur.")
            rag_collection = st.selectbox("RAG koleksiyonu", colls)
        top_k = st.slider("Getirilen parça sayısı (top-k)", 3, 20, 8)
    elif method == METHOD_FINETUNE:
        if append_run:
            ft_model_id = append_run.get("finetune_model_id")
            st.info(f"FT model: `{ft_model_id}` (kilitli)")
        else:
            st.caption(
                "Standart dokümanlarla eğitilmiş fine-tuned model id'sini gir. "
                "FT işini 'Fine-tune' sekmesinden başlatabilirsin."
            )
            ft_model_id = st.text_input(
                "Fine-tuned model id",
                placeholder="ft:gpt-4o-mini-2024-07-18:org::id",
                key="ft_model_id_input",
            )

    # Resume vs fresh — only for unsaved drafts in "new pool" mode
    existing_draft = None
    resume = False
    if not append_run:
        existing_draft = next(
            (
                r
                for r in storage.list_runs()
                if r["status"] == "draft" and r["name"] == name and r["method"] == method
            ),
            None,
        )
        if existing_draft:
            choice = st.radio(
                f"Bu isimde **yarım kalmış** ({existing_draft['id'][:8]}) bir çalıştırma var. Ne yapayım?",
                ["Kaldığı yerden devam et", "Tamamen baştan başla"],
                horizontal=True,
            )
            resume = choice.startswith("Kaldığı")

    if st.button("Çalıştır", type="primary"):
        if method == METHOD_RAG and not rag_collection:
            st.error("RAG yönteminde koleksiyon seçmelisin.")
            st.stop()
        if method == METHOD_FINETUNE and not ft_model_id:
            st.error("Fine-tune yönteminde FT model id girmelisin.")
            st.stop()

        sidebar_llm = st.session_state.get("llm_model") or settings.llm_model
        effective_llm = (
            append_run["llm_model"] if append_run
            else (ft_model_id if method == METHOD_FINETUNE else sidebar_llm)
        )

        if append_run:
            run_id = append_run["id"]
        elif resume and existing_draft:
            run_id = existing_draft["id"]
            st.info(f"Devam ediliyor: {run_id[:8]}")
        else:
            if existing_draft:
                storage.delete_run(existing_draft["id"])
            run_id = storage.create_run(
                name=name,
                method=method,
                prompt=prompt,
                llm_model=effective_llm,
                embedding_model=(
                    st.session_state.get("embedding_model") or settings.embedding_model
                )
                if method == METHOD_RAG
                else None,
                rag_collection=rag_collection if method == METHOD_RAG else None,
                rag_documents=None,
                finetune_model_id=ft_model_id if method == METHOD_FINETUNE else None,
            )

        # Append modunda, mevcut başlıkları LLM'e "tekrarlama" diye veriyoruz
        avoid_titles = None
        if append_run:
            avoid_titles = [
                (v.get("title") or v.get("description") or "")[:120]
                for v in storage.get_violations(run_id)
            ]

        try:
            with st.spinner("LLM çalışıyor..."):
                if method == METHOD_NAIVE:
                    items = llm.generate_naive(
                        prompt, model=effective_llm, n=int(n_violations),
                        avoid_titles=avoid_titles,
                    )
                elif method == METHOD_OPTIMIZED:
                    items = llm.generate_optimized(
                        prompt, model=effective_llm, n=int(n_violations),
                        avoid_titles=avoid_titles,
                    )
                elif method == METHOD_RAG:
                    chunks = rag.retrieve(rag_collection, prompt, k=top_k)
                    if not chunks:
                        st.warning("RAG koleksiyonu boş veya eşleşme yok.")
                    items = llm.generate_rag(
                        prompt, chunks, model=effective_llm, n=int(n_violations),
                        avoid_titles=avoid_titles,
                    )
                else:  # METHOD_FINETUNE
                    items = llm.generate_finetuned(
                        prompt, ft_model_id=effective_llm, n=int(n_violations),
                        avoid_titles=avoid_titles,
                    )
        except Exception as e:
            st.error(f"Hata: {e}")
            st.stop()

        added = storage.add_violations(run_id, items)
        st.session_state["draft_run_id"] = run_id
        st.session_state["draft_method"] = method
        st.session_state["draft_is_append"] = bool(append_run)
        if append_run:
            st.success(
                f"{added} ihlal **eklendi**. Toplam: "
                f"{storage.count_violations(run_id)}. Aşağıdan onaylayıp Excel'i tazele."
            )
        else:
            st.success(f"{added} ihlal üretildi. Aşağıdan inceleyip kaydet/iptal et.")

    # Preview & confirm
    draft_id = st.session_state.get("draft_run_id")
    is_append = st.session_state.get("draft_is_append", False)
    if draft_id:
        run = storage.get_run(draft_id)
        if run:
            st.divider()
            label = "Mevcut havuza eklendi" if is_append else "Taslak"
            st.subheader(f"{label}: {run['name']} ({run['id'][:8]})")
            vs = storage.get_violations(draft_id)
            st.dataframe(_violations_df(vs), use_container_width=True)

            if is_append:
                last_batch = max((v.get("batch_no") or 1) for v in vs) if vs else 1
                c1, c2 = st.columns(2)
                if c1.button("Excel'i tazele (onayla)", type="primary"):
                    out = excel_export.export_run(draft_id)
                    st.success(f"Excel güncellendi: {out}")
                    _reset_draft()
                    st.rerun()
                if c2.button(f"Son batch'i geri al (batch={last_batch})"):
                    storage.delete_batch(draft_id, last_batch)
                    _reset_draft()
                    st.rerun()
            elif run["status"] == "draft":
                c1, c2, c3 = st.columns(3)
                if c1.button("Kaydet (onayla)", type="primary"):
                    storage.mark_saved(draft_id)
                    out = excel_export.export_run(draft_id)
                    st.success(f"Kaydedildi. Excel: {out}")
                    _reset_draft()
                    st.rerun()
                if c2.button("Excel önizleme"):
                    out = excel_export.export_run(draft_id)
                    st.info(f"Excel yazıldı: {out}")
                if c3.button("İptal et (sil)"):
                    storage.delete_run(draft_id)
                    _reset_draft()
                    st.rerun()


# =========================================================
# View tab
# =========================================================
with tab_view:
    view_id = st.session_state.get("view_run_id")
    runs_all = storage.list_runs()
    options = {r["id"]: f"{r['name']} · {METHOD_LABELS.get(r['method'], r['method'])} · {r['status']}" for r in runs_all}
    sel_id = st.selectbox(
        "Havuz seç",
        list(options.keys()) or [""],
        index=(list(options.keys()).index(view_id) if view_id in options else 0) if options else 0,
        format_func=lambda k: options.get(k, "-"),
    )
    if sel_id:
        run = storage.get_run(sel_id)
        st.markdown(
            f"**{run['name']}** · `{run['id'][:8]}` · "
            f"yöntem: {METHOD_LABELS.get(run['method'], run['method'])} · "
            f"durum: {run['status']}"
        )
        meta_cols = st.columns(4)
        meta_cols[0].metric("LLM", run["llm_model"])
        meta_cols[1].metric("Embedding", run.get("embedding_model") or "-")
        meta_cols[2].metric("Koleksiyon", run.get("rag_collection") or "-")
        meta_cols[3].metric("FT model", run.get("finetune_model_id") or "-")
        with st.expander("Kullanılan promt"):
            st.code(run["prompt"])
        vs = storage.get_violations(sel_id)
        st.metric("İhlal sayısı", len(vs))
        st.dataframe(_violations_df(vs), use_container_width=True)
        if st.button("Excel'e aktar", key=f"view_xls_{sel_id}"):
            out = excel_export.export_run(sel_id)
            st.success(f"Yazıldı: {out}")


# =========================================================
# Fine-tune tab
# =========================================================
with tab_ft:
    st.subheader("Fine-tune (Yöntem 4)")
    st.caption(
        "Standart dokümanlardan sentetik eğitim verisi (JSONL) üretip "
        "OpenAI fine-tune işini başlatır. Bittiğinde alınan model id "
        "'Çalıştır' sekmesinde Yöntem 4 için kullanılır."
    )

    colls = rag.list_collections()
    if not colls:
        st.warning("Önce 'RAG / Doküman' sekmesinden bir koleksiyon hazırla.")
    coll = st.selectbox("Kaynak koleksiyon", colls, key="ft_coll")
    c1, c2, c3 = st.columns(3)
    max_chunks = c1.number_input("Maks. parça", min_value=10, max_value=2000, value=200, step=10)
    samples_per_chunk = c2.number_input("Parça başına örnek", 1, 5, 1)
    base_model = c3.text_input("Base model", value=settings.llm_model)

    if st.button("Eğitim verisi (JSONL) üret", disabled=not coll):
        with st.spinner("Sentetik veri üretiliyor (LLM çağrıları)..."):
            try:
                p = finetune.build_training_jsonl(
                    coll,
                    samples_per_chunk=int(samples_per_chunk),
                    max_chunks=int(max_chunks),
                    base_model=base_model,
                )
                st.session_state["ft_jsonl_path"] = str(p)
                st.success(f"JSONL hazır: {p}")
            except Exception as e:
                st.error(f"Hata: {e}")

    jsonl_path = st.session_state.get("ft_jsonl_path")
    if jsonl_path:
        st.code(f"jsonl: {jsonl_path}")
        if st.button("Fine-tune işini başlat"):
            try:
                with st.spinner("Dosya yükleniyor ve FT işi oluşturuluyor..."):
                    info = finetune.start_finetune_job(Path(jsonl_path), base_model)
                st.success(f"Job: {info['job_id']}  ·  status: {info['status']}")
            except Exception as e:
                st.error(f"Hata: {e}")

    st.divider()
    st.markdown("**Fine-tune işleri**")
    if st.button("Listele / yenile"):
        try:
            st.session_state["ft_jobs"] = finetune.list_jobs(limit=20)
        except Exception as e:
            st.error(f"Hata: {e}")
    jobs = st.session_state.get("ft_jobs", [])
    if jobs:
        st.dataframe(pd.DataFrame(jobs), use_container_width=True)

    job_id = st.text_input("Job id (durum sorgu)")
    if job_id and st.button("Durumu getir"):
        try:
            st.json(finetune.job_status(job_id.strip()))
        except Exception as e:
            st.error(f"Hata: {e}")


# =========================================================
# İhlalli IFC Oluşturma — üst seviye 2. sekme
# =========================================================
with top_ifc:
    ifc_t1, ifc_t2, ifc_t3 = st.tabs(
        ["Baseline IFC üret", "İhlal enjekte et", "IFC'leri görüntüle"]
    )

    # -------- Baseline üretimi --------
    with ifc_t1:
        st.subheader("İhlalsiz baseline IFC üret")
        st.caption(
            "LLM tam IFC4 STEP metni üretir; ifcopenshell ile parse edilir. "
            "Parse başarısızsa 1 retry yapılır. Tüm boyutlar bilinçli olarak "
            "cömert tutulur — bu dosyalarda ihlal olmamalı."
        )
        c1, c2 = st.columns([2, 1])
        bn_prefix = c1.text_input("İsim öneki", value="House")
        bn_count = c2.number_input("Adet", 1, 20, 4)
        bn_model = st.text_input("IFC LLM modeli", value=settings.ifc_llm_model)
        bn_prompt = st.text_area(
            "Promt (baseline)",
            value=(
                "Tek aileli, küçük bir konutun tam IFC4 dosyasını üret. "
                "Tüm boyutlar mevzuata fazlasıyla uygun (ihlalsiz) olsun. "
                "Sadece geçerli SPF metni döndür."
            ),
            height=120,
        )
        if st.button("Baseline IFC'leri üret", type="primary"):
            with st.spinner(f"{bn_count} adet baseline üretiliyor (LLM)..."):
                try:
                    res = ifc_gen.generate_baselines(
                        n=int(bn_count), seed_prompt=bn_prompt,
                        model=bn_model.strip() or None, name_prefix=bn_prefix,
                    )
                    df = pd.DataFrame([{
                        "id": r["ifc_model_id"][:8],
                        "status": r["status"],
                        "ifc": r["ifc_path"],
                        "error": r["error"],
                    } for r in res])
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Hata: {e}")

    # -------- Enjeksiyon --------
    with ifc_t2:
        st.subheader("Baseline'a havuzdan ihlal enjekte et")

        baselines = [m for m in storage.list_ifc_models("baseline")
                     if m["status"] == "ok"]
        if not baselines:
            st.warning("Önce geçerli (status=ok) baseline IFC üretmelisin.")
        bo = {m["id"]: f"{m['name']} · {m['id'][:8]} · {m['created_at']}"
              for m in baselines}
        sel_base = st.selectbox("Baseline IFC", list(bo.keys()) or [""],
                                format_func=lambda k: bo.get(k, "-"))

        saved_pools = [r for r in storage.list_runs() if r["status"] == "saved"]
        po = {r["id"]: f"{r['name']} · {METHOD_LABELS.get(r['method'], r['method'])}"
              for r in saved_pools}
        sel_pool = st.selectbox("İhlal havuzu (run)", list(po.keys()) or [""],
                                format_func=lambda k: po.get(k, "-"))

        c1, c2, c3 = st.columns(3)
        inj_n = c1.number_input("Enjekte edilecek ihlal sayısı", 1, 100, 10)
        inj_cat = c2.text_input("Kategori filtresi (boş = hepsi)")
        inj_seed = c3.number_input("Tohum (rastgele seçim)", 0, 10_000, 42)
        inj_model = st.text_input("Enjeksiyon LLM modeli",
                                  value=settings.ifc_llm_model)

        if st.button("İhlal enjekte et", type="primary",
                     disabled=not (sel_base and sel_pool)):
            try:
                pool_vs = storage.get_violations(sel_pool)
                picked = ifc_inject.pick_violations(
                    pool_vs, n=int(inj_n),
                    category=(inj_cat.strip() or None),
                    seed=int(inj_seed),
                )
                if not picked:
                    st.warning("Filtreyle eşleşen ihlal yok.")
                    st.stop()
                with st.spinner(f"{len(picked)} ihlal enjekte ediliyor..."):
                    out = ifc_inject.inject_violations(
                        baseline_id=sel_base, violations=picked,
                        pool_run_id=sel_pool,
                        model=inj_model.strip() or None,
                        selection_filter={"category": inj_cat or None,
                                          "n": int(inj_n),
                                          "seed": int(inj_seed)},
                    )
                st.success(
                    f"Bitti. Uygulanan: {out['summary']['applied']}, "
                    f"atlanan: {out['summary']['skipped']}.\n"
                    f"IFC: {out['ifc_path']}\nLabels: {out['labels_path']}"
                )
            except Exception as e:
                st.error(f"Hata: {e}")

    # -------- Görüntüleme --------
    with ifc_t3:
        st.subheader("Üretilmiş IFC'ler")
        kind = st.radio("Tür", ["baseline", "violated"], horizontal=True)
        models = storage.list_ifc_models(kind)
        if not models:
            st.caption("Bu türde IFC henüz yok.")
        else:
            df = pd.DataFrame([{
                "id": m["id"][:8],
                "name": m["name"],
                "status": m["status"],
                "llm": m["llm_model"],
                "parent": (m["parent_id"] or "")[:8],
                "pool": (m["pool_run_id"] or "")[:8],
                "created_at": m["created_at"],
                "file": m["file_path"],
            } for m in models])
            st.dataframe(df, use_container_width=True)

            opt = {m["id"]: f"{m['name']} · {m['id'][:8]}" for m in models}
            sel = st.selectbox("Detay", list(opt.keys()),
                               format_func=lambda k: opt[k])
            m = storage.get_ifc_model(sel)
            cols = st.columns(3)
            with open(m["file_path"], "rb") as f:
                cols[0].download_button("IFC indir", f,
                                        file_name=Path(m["file_path"]).name)
            if m.get("labels_path"):
                with open(m["labels_path"], "rb") as f:
                    cols[1].download_button("Labels JSON indir", f,
                                            file_name=Path(m["labels_path"]).name)
            if m.get("meta_path"):
                with open(m["meta_path"], "rb") as f:
                    cols[2].download_button("Meta JSON indir", f,
                                            file_name=Path(m["meta_path"]).name)

            if kind == "violated":
                labs = storage.get_ifc_labels(sel)
                if labs:
                    ldf = pd.DataFrame([{
                        "status": l["status"],
                        "title": l["title"],
                        "category": l["category"],
                        "severity": l["severity"],
                        "ifc_type": l["ifc_type"],
                        "ifc_name": l["ifc_name"],
                        "attribute": l["attribute"],
                        "before": l["value_before"],
                        "after": l["value_after"],
                        "reason": l["reason"],
                    } for l in labs])
                    st.markdown("**İhlal etiketleri**")
                    st.dataframe(ldf, use_container_width=True)

            if st.button("Bu kaydı sil", key=f"del_ifc_{sel}"):
                storage.delete_ifc_model(sel)
                st.rerun()
