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

from violation_pool import (
    excel_export, finetune, graph_viewer, ifc_gen, ifc_graph, ifc_inject,
    ifc_viewer, llm, rag, storage,
)
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
st.sidebar.subheader("Token Kullanımı (toplam)")
_tot = storage.usage_totals()
st.sidebar.metric("Toplam token", f"{_tot['total_tokens']:,}",
                  f"{_tot['calls']} çağrı")
st.sidebar.caption(
    f"prompt {_tot['prompt_tokens']:,}  ·  completion {_tot['completion_tokens']:,}"
)

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
    tab_run, tab_view, tab_rag, tab_ft, tab_tok = st.tabs(
        ["Çalıştır", "Havuzu Görüntüle", "RAG / Doküman", "Fine-tune",
         "Token Kullanımı"]
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

        umeta = {"pool_run_id": run_id}
        try:
            with st.spinner("LLM çalışıyor..."):
                rag_chunks = None
                if method == METHOD_RAG:
                    rag_chunks = rag.retrieve(rag_collection, prompt, k=top_k,
                                              usage_meta=umeta)
                    if not rag_chunks:
                        st.warning("RAG koleksiyonu boş veya eşleşme yok.")
                items = llm.generate_chunked(
                    method=method,
                    user_prompt=prompt,
                    model=(effective_llm if method != METHOD_FINETUNE else None),
                    n=int(n_violations),
                    avoid_titles=avoid_titles,
                    usage_meta=umeta,
                    context_chunks=rag_chunks,
                    ft_model_id=(effective_llm if method == METHOD_FINETUNE else None),
                )
        except Exception as e:
            st.error(f"Hata: {e}")
            st.stop()

        added = storage.add_violations(run_id, items)
        st.session_state["draft_run_id"] = run_id
        st.session_state["draft_method"] = method
        st.session_state["draft_is_append"] = bool(append_run)
        tot = storage.usage_totals(pool_run_id=run_id)
        tok_line = (
            f"  ·  token: {tot['total_tokens']} "
            f"(prompt {tot['prompt_tokens']} + completion {tot['completion_tokens']}, "
            f"{tot['calls']} çağrı)"
        )
        if append_run:
            st.success(
                f"{added} ihlal **eklendi**. Toplam: "
                f"{storage.count_violations(run_id)}.{tok_line}"
            )
        else:
            st.success(f"{added} ihlal üretildi.{tok_line}")

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
        meta_cols = st.columns(5)
        meta_cols[0].metric("LLM", run["llm_model"])
        meta_cols[1].metric("Embedding", run.get("embedding_model") or "-")
        meta_cols[2].metric("Koleksiyon", run.get("rag_collection") or "-")
        meta_cols[3].metric("FT model", run.get("finetune_model_id") or "-")
        run_tot = storage.usage_totals(pool_run_id=sel_id)
        meta_cols[4].metric("Token (total)", f"{run_tot['total_tokens']:,}",
                            f"{run_tot['calls']} çağrı")
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
with tab_tok:
    st.subheader("Token Kullanımı")
    st.caption(
        "Her LLM ve embedding çağrısı kaydedilir. Aşağıdan filtreleyebilir, "
        "operasyon/model bazında toplamı görebilirsin."
    )
    fc1, fc2, fc3 = st.columns(3)
    f_op = fc1.selectbox(
        "Operasyon",
        ["", "gen_naive", "gen_optimized", "gen_rag", "gen_finetuned",
         "embed", "ifc_gen", "ifc_inject", "ft_prep"],
    )
    f_pool = fc2.text_input("Pool run id (opsiyonel)")
    f_ifc = fc3.text_input("IFC model id (opsiyonel)")

    rows = storage.list_usage(
        operation=(f_op or None),
        pool_run_id=(f_pool.strip() or None) if f_pool else None,
        ifc_model_id=(f_ifc.strip() or None) if f_ifc else None,
    )
    if rows:
        df_u = pd.DataFrame(rows)
        # özet metrikleri
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Çağrı", len(df_u))
        m2.metric("Toplam token", f"{int(df_u['total_tokens'].sum()):,}")
        m3.metric("Prompt", f"{int(df_u['prompt_tokens'].sum()):,}")
        m4.metric("Completion", f"{int(df_u['completion_tokens'].sum()):,}")

        st.markdown("**Operasyon × Model toplamı**")
        agg = df_u.groupby(["operation", "model"], dropna=False)[
            ["prompt_tokens", "completion_tokens", "total_tokens"]
        ].sum().reset_index()
        agg["calls"] = (
            df_u.groupby(["operation", "model"]).size().reset_index(name="calls")["calls"]
        )
        st.dataframe(agg, use_container_width=True)

        st.markdown("**Kayıtlar (en yeni üstte)**")
        show_cols = ["created_at", "operation", "model", "prompt_tokens",
                     "completion_tokens", "total_tokens",
                     "pool_run_id", "ifc_model_id", "collection", "note"]
        st.dataframe(df_u[show_cols], use_container_width=True, height=380)
    else:
        st.info("Filtreyle eşleşen kayıt yok.")


with top_ifc:
    ifc_t1, ifc_t1b, ifc_t2, ifc_t3 = st.tabs(
        ["Baseline IFC üret", "Gerçek IFC içe aktar",
         "İhlal enjekte et", "IFC'leri görüntüle"]
    )

    # -------- Baseline üretimi --------
    with ifc_t1:
        st.subheader("İhlalsiz baseline IFC üret")
        mode = st.radio(
            "Üretim modu",
            ["parametric", "raw"],
            format_func=lambda m: (
                "Parametrik (önerilen): LLM JSON spec verir, IFC'yi biz kurarız"
                if m == "parametric"
                else "Raw: LLM doğrudan IFC4 STEP yazar (deneysel; çoğunlukla boş)"
            ),
            horizontal=False,
        )
        st.caption(
            "Parametrik modda LLM sadece oda boyutları / kapı-pencere konumları "
            "için JSON döndürür; valid IFC4 geometrisi (duvar/slab/kapı/pencere "
            "extrüzyonları, placement zinciri) ifcopenshell ile inşa edilir. "
            "Bu sayede 3D görselleştirme ve graph zenginleşir."
        )
        c1, c2 = st.columns([2, 1])
        bn_prefix = c1.text_input("İsim öneki", value="House")
        bn_count = c2.number_input("Adet", 1, 20, 1)
        bn_model = st.text_input("IFC LLM modeli", value=settings.ifc_llm_model)
        bn_prompt = st.text_area(
            "Genel promt (her IFC'ye uygulanır)",
            value=(
                "Erişilebilirlik ve kullanılabilirlik açısından sorunsuz, "
                "mevzuata fazlasıyla uygun (ihlalsiz) küçük bir konut spec'i üret."
                if mode == "parametric"
                else "Tek aileli, küçük bir konutun tam IFC4 dosyasını üret. "
                "Tüm boyutlar mevzuata fazlasıyla uygun (ihlalsiz) olsun."
            ),
            height=80,
        )

        vary = st.checkbox(
            "Her IFC için farklı program tipi enjekte et (önerilen)",
            value=True,
        )
        if vary:
            with st.expander("Program tipleri (her satır = bir varyasyon)",
                             expanded=False):
                default_text = "\n".join(ifc_gen.DEFAULT_VARIATIONS)
                vtext = st.text_area(
                    "Varyasyonlar", value=default_text, height=260,
                    key="bn_variations",
                )
                variations = [v.strip() for v in vtext.split("\n") if v.strip()]
            st.caption(
                f"{len(variations)} varyasyon · {bn_count} IFC üretilecek "
                f"(adet > varyasyon ise döngüsel kullanılır)."
            )
        else:
            variations = None
            st.caption("Tüm IFC'ler aynı promtla üretilir.")

        if st.button("Baseline IFC'leri üret", type="primary"):
            with st.spinner(f"{bn_count} adet baseline üretiliyor..."):
                try:
                    res = ifc_gen.generate_baselines(
                        n=int(bn_count), seed_prompt=bn_prompt,
                        model=bn_model.strip() or None, name_prefix=bn_prefix,
                        mode=mode, variations=variations, vary=vary,
                    )
                    df = pd.DataFrame([{
                        "id": r["ifc_model_id"][:8],
                        "status": r["status"],
                        "mode": r.get("mode", "-"),
                        "ifc": r["ifc_path"],
                        "error": r["error"],
                    } for r in res])
                    st.dataframe(df, use_container_width=True)
                except Exception as e:
                    st.error(f"Hata: {e}")

    # -------- Gerçek IFC içe aktar --------
    with ifc_t1b:
        st.subheader("Gerçek dünyadan IFC dosyası içe aktar")
        st.caption(
            "Yüklediğin .ifc dosyaları `ifc_models/imports/` altında saklanır, "
            "ifcopenshell ile parse edilir, graph (NetworkX) hemen üretilir. "
            "Sonrasında bunlara da ihlal enjekte edebilir, 3D/Graph "
            "görselleştirmesini açabilirsin."
        )
        imp_uploads = st.file_uploader(
            "IFC dosyaları (birden fazla seçilebilir)",
            type=["ifc"], accept_multiple_files=True,
            key="imp_uploads",
        )
        imp_dir_input = st.text_input(
            "veya: yerel bir klasör yolu ver (tüm .ifc'leri tara)",
            value="", placeholder="/Users/.../IFCs",
        )
        c_imp1, c_imp2 = st.columns(2)
        if c_imp1.button("Yüklenenleri içe aktar",
                         disabled=not imp_uploads):
            rows = []
            for uf in imp_uploads:
                tmp = settings.ifc_dir / "imports" / f"_upload_{uf.name}"
                tmp.write_bytes(uf.getbuffer())
                try:
                    r = ifc_gen.import_real_ifc(src_path=tmp, name=uf.name)
                    rows.append({"name": uf.name, "id": r["ifc_model_id"][:8],
                                 "status": r["status"], "graph": bool(r["graph_path"]),
                                 "error": r["error"]})
                finally:
                    try: tmp.unlink()
                    except Exception: pass
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        if c_imp2.button("Klasördekileri tara ve içe aktar",
                         disabled=not imp_dir_input.strip()):
            base = Path(imp_dir_input.strip()).expanduser()
            if not base.exists():
                st.error(f"Klasör bulunamadı: {base}")
            else:
                files = sorted(base.rglob("*.ifc"))
                if not files:
                    st.warning("Klasörde .ifc dosyası bulunamadı.")
                else:
                    rows = []
                    with st.spinner(f"{len(files)} dosya içe aktarılıyor..."):
                        for fp in files:
                            try:
                                r = ifc_gen.import_real_ifc(src_path=fp, name=fp.name)
                                rows.append({
                                    "name": fp.name, "id": r["ifc_model_id"][:8],
                                    "status": r["status"], "graph": bool(r["graph_path"]),
                                    "error": r["error"],
                                })
                            except Exception as e:
                                rows.append({
                                    "name": fp.name, "id": "-",
                                    "status": "error", "graph": False,
                                    "error": str(e),
                                })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True)

        st.divider()
        imported = storage.list_ifc_models("imported")
        st.caption(f"İçe aktarılmış IFC: {len(imported)}")
        if imported:
            st.dataframe(pd.DataFrame([{
                "id": m["id"][:8], "name": m["name"], "status": m["status"],
                "graph": bool(m.get("graph_path")),
                "created_at": m["created_at"], "file": m["file_path"],
            } for m in imported]), use_container_width=True)

    # -------- Enjeksiyon --------
    with ifc_t2:
        st.subheader("Baseline'a havuzdan ihlal enjekte et")
        st.caption(
            "İhlalleri tek tek seçip aynı anda toplu enjekte edebilirsin. "
            "İstersen yalancı (decoy) etiketler de eklenir: bunlar gerçek "
            "ihlal değildir ama işaretli görünür — sistemin gerçekleri "
            "decoy'lardan ayırma kabiliyetini ölçmek için."
        )

        sources = [m for m in storage.list_ifc_models()
                   if m["kind"] in ("baseline", "imported") and m["status"] == "ok"]
        if not sources:
            st.warning("Önce geçerli (status=ok) bir baseline veya imported IFC olmalı.")
        bo = {m["id"]: f"[{m['kind']}] {m['name']} · {m['id'][:8]} · {m['created_at']}"
              for m in sources}
        sel_base = st.selectbox("Kaynak IFC (baseline / imported)",
                                list(bo.keys()) or [""],
                                format_func=lambda k: bo.get(k, "-"))

        saved_pools = [r for r in storage.list_runs() if r["status"] == "saved"]
        po = {r["id"]: f"{r['name']} · {METHOD_LABELS.get(r['method'], r['method'])}"
              for r in saved_pools}
        sel_pool = st.selectbox("İhlal havuzu (run)", list(po.keys()) or [""],
                                format_func=lambda k: po.get(k, "-"))

        if sel_pool:
            pool_vs = storage.get_violations(sel_pool)
            # Kategori filtresi
            cats = sorted({v.get("category") or "-" for v in pool_vs})
            f1, f2 = st.columns([1, 2])
            cat_filter = f1.selectbox("Kategori filtresi", ["(hepsi)"] + cats)
            sev_filter = f2.multiselect(
                "Şiddet filtresi (boş = hepsi)",
                ["düşük", "orta", "yüksek", "kritik"],
            )
            filtered = [
                v for v in pool_vs
                if (cat_filter == "(hepsi)" or v.get("category") == cat_filter)
                and (not sev_filter or (v.get("severity") in sev_filter))
            ]
            st.caption(f"Filtre sonrası: {len(filtered)} / {len(pool_vs)} ihlal")

            opt = {
                v["id"]: f"[{(v.get('category') or '-')}] "
                          f"{(v.get('title') or v.get('description') or '')[:90]}"
                          f"  (sev={v.get('severity') or '-'})"
                for v in filtered
            }

            # ---- Toplu seçim (tek state: inj_sel_widget) ----
            # Filtre değiştiğinde stale id'leri temizle
            cur_sel = st.session_state.get("inj_sel_widget", [])
            cur_sel = [i for i in cur_sel if i in opt]
            st.session_state["inj_sel_widget"] = cur_sel

            qc1, qc2, qc3, qc4 = st.columns(4)
            if qc1.button("Tümünü seç", key="inj_btn_all"):
                st.session_state["inj_sel_widget"] = list(opt.keys())
                st.rerun()
            n_rand = qc2.number_input(
                "Rastgele N", 1, max(1, len(filtered)),
                min(10, max(1, len(filtered))),
                key="inj_rand_n",
            )
            r_seed = qc3.number_input("Tohum", 0, 10_000, 42, key="inj_rand_seed")
            if qc4.button("Rastgele N seç", key="inj_btn_rand"):
                import random as _r
                _r.seed(int(r_seed))
                pool_ids = list(opt.keys())
                st.session_state["inj_sel_widget"] = _r.sample(
                    pool_ids, min(int(n_rand), len(pool_ids))
                )
                st.rerun()
            qc5, qc6 = st.columns(2)
            if qc5.button("Seçimi temizle", key="inj_btn_clear"):
                st.session_state["inj_sel_widget"] = []
                st.rerun()
            if qc6.button("Filtreyle eşleşen ilk N", key="inj_btn_topn"):
                st.session_state["inj_sel_widget"] = list(opt.keys())[: int(n_rand)]
                st.rerun()

            selected_ids = st.multiselect(
                "Enjekte edilecek ihlaller (toplu seçim)",
                list(opt.keys()),
                format_func=lambda i: opt.get(i, i),
                key="inj_sel_widget",
            )

            dc1, dc2, dc3 = st.columns(3)
            decoy_ratio = dc1.slider("Decoy oranı (%)", 0, 100, 20) / 100.0
            decoy_seed = dc2.number_input("Decoy tohumu", 0, 10_000, 7,
                                          key="inj_decoy_seed")
            inj_model = dc3.text_input("Enjeksiyon LLM modeli",
                                       value=settings.ifc_llm_model)
            fill_from_pool = st.checkbox(
                "Bir ihlal IFC'ye uymazsa havuzdan başka birini dene "
                "(önerilen)", value=True, key="inj_fill",
            )

            ne_real = len(selected_ids)
            ne_decoys = int(round(ne_real * decoy_ratio))
            st.caption(
                f"Seçim: **{ne_real}** gerçek ihlal + **{ne_decoys}** decoy "
                f"= toplam {ne_real + ne_decoys} işaretli eleman."
            )

            if st.button("Seçilen ihlalleri enjekte et", type="primary",
                         disabled=not (sel_base and selected_ids)):
                try:
                    picked = [v for v in pool_vs if v["id"] in selected_ids]
                    with st.spinner(
                        f"{len(picked)} ihlal sırayla deneniyor "
                        "(her biri için ayrı LLM çağrısı)..."
                    ):
                        out = ifc_inject.inject_violations(
                            baseline_id=sel_base, violations=picked,
                            pool_run_id=sel_pool,
                            model=inj_model.strip() or None,
                            selection_filter={
                                "category": (None if cat_filter == "(hepsi)" else cat_filter),
                                "severity": sev_filter or None,
                                "selected_ids": selected_ids,
                                "decoy_ratio": decoy_ratio,
                                "fill_from_pool": fill_from_pool,
                            },
                            decoy_ratio=decoy_ratio,
                            decoy_seed=int(decoy_seed),
                            fill_from_pool=fill_from_pool,
                        )
                    inj_tot = storage.usage_totals(ifc_model_id=out["ifc_model_id"])
                    s = out["summary"]
                    repl = s.get("replaced_from_pool", 0)
                    repl_txt = (
                        f"; yedek havuzdan: **{repl}**" if repl else ""
                    )
                    st.success(
                        f"Bitti. İstenen: {s['requested']}, "
                        f"uygulanan: **{s['applied']}**, "
                        f"atlanan: {s['skipped']}{repl_txt}, "
                        f"decoy: {s['decoys']}. "
                        f"Token: {inj_tot['total_tokens']:,} "
                        f"({inj_tot['calls']} çağrı)\n"
                        f"IFC: {out['ifc_path']}\nLabels: {out['labels_path']}"
                    )
                except Exception as e:
                    st.error(f"Hata: {e}")

    # -------- Görüntüleme --------
    with ifc_t3:
        st.subheader("Üretilmiş IFC'ler")
        kind = st.radio("Tür", ["baseline", "violated", "imported"],
                         horizontal=True)
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
            ifc_tot = storage.usage_totals(ifc_model_id=sel)
            mcols = st.columns(4)
            mcols[0].metric("LLM", m["llm_model"])
            mcols[1].metric("Status", m["status"])
            mcols[2].metric("Token (bu IFC)", f"{ifc_tot['total_tokens']:,}",
                            f"{ifc_tot['calls']} çağrı")
            mcols[3].metric("Created", m["created_at"])
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

            labs: list[dict] = []
            if kind == "violated":
                labs = storage.get_ifc_labels(sel)
                if labs:
                    import json as _json
                    def _ev_short(ev_json: str | None) -> str:
                        try:
                            evs = _json.loads(ev_json or "[]") or []
                        except Exception:
                            evs = []
                        if not evs:
                            return ""
                        parts = []
                        for e in evs[:3]:
                            d = (e or {}).get("document") or "?"
                            p = (e or {}).get("page")
                            c = (e or {}).get("clause")
                            t = f"{d}"
                            if p is not None: t += f":p{p}"
                            if c: t += f" §{c}"
                            parts.append(t)
                        if len(evs) > 3:
                            parts.append(f"(+{len(evs)-3})")
                        return "; ".join(parts)

                    ldf = pd.DataFrame([{
                        "violation_code": (l.get("violation_id") or "")[:8],
                        "is_decoy": bool(l.get("is_decoy")),
                        "action": l.get("action"),
                        "status": l["status"],
                        "title": l["title"],
                        "category": l["category"],
                        "severity": l["severity"],
                        "evidence": _ev_short(l.get("evidence_json")),
                        "ifc_type": l["ifc_type"],
                        "ifc_name": l["ifc_name"],
                        "attribute": l["attribute"],
                        "before": l["value_before"],
                        "after": l["value_after"],
                        "reason": l["reason"],
                    } for l in labs])
                    n_real = int((~ldf["is_decoy"]).sum())
                    n_dec = int(ldf["is_decoy"].sum())
                    st.markdown(
                        f"**Etiketler** · gerçek ihlal: **{n_real}** · "
                        f"decoy: **{n_dec}**"
                    )
                    st.dataframe(ldf, use_container_width=True)
                    if st.button("Etiketleri Excel'e indir",
                                 key=f"lab_xls_{sel}"):
                        out = excel_export.export_ifc_labels(sel)
                        st.success(f"Yazıldı: {out}")

                    # Detay: bir label için tam kanıt zinciri
                    with st.expander("Bir etiket için tam kanıt zinciri",
                                     expanded=False):
                        opts2 = {l["id"]: f"{(l.get('title') or l.get('description') or l['id'])[:80]}"
                                          + (" [decoy]" if l.get("is_decoy") else "")
                                 for l in labs}
                        ls = st.selectbox("Etiket", list(opts2.keys()),
                                          format_func=lambda k: opts2[k],
                                          key=f"lab_pick_{sel}")
                        lab = next((x for x in labs if x["id"] == ls), None)
                        if lab:
                            try:
                                evs = _json.loads(lab.get("evidence_json") or "[]")
                            except Exception:
                                evs = []
                            st.json({
                                "violation_id (havuz kodu)": lab.get("violation_id"),
                                "title": lab.get("title"),
                                "category": lab.get("category"),
                                "severity": lab.get("severity"),
                                "threshold": lab.get("threshold"),
                                "ifc_global_id": lab.get("ifc_global_id"),
                                "action": lab.get("action"),
                                "is_decoy": bool(lab.get("is_decoy")),
                                "evidence": evs,
                                "reason": lab.get("reason"),
                            })

            # ---- Görselleştirme: 3D IFC ve Graph ----
            hl_guids = set()
            dc_guids = set()
            if kind == "violated":
                for l in labs:
                    g = l.get("ifc_global_id")
                    if not g:
                        continue
                    if l.get("is_decoy") or l.get("status") == "decoy":
                        dc_guids.add(g)
                    elif l.get("status") == "applied":
                        hl_guids.add(g)
            with st.expander("Görselleştir (3D IFC · Graph)", expanded=False):
                view_3d, view_graph = st.tabs(["3D IFC", "Graph"])

                with view_3d:
                    vc1, vc2 = st.columns([1, 3])
                    max_el = vc1.number_input(
                        "Maks. eleman", 100, 20000, 5000, step=100, key=f"max_el_{sel}",
                    )
                    highlight = vc2.checkbox(
                        "Modifiye/eklenen elemanları kırmızı vurgula",
                        value=True, key=f"hl_{sel}", disabled=(kind != "violated"),
                    )
                    if st.button("3D çiz", key=f"draw3d_{sel}"):
                        try:
                            with st.spinner("Geometri tessellate ediliyor..."):
                                fig, stats = ifc_viewer.ifc_to_figure(
                                    m["file_path"],
                                    highlight_guids=hl_guids if highlight else set(),
                                    decoy_guids=dc_guids if highlight else set(),
                                    max_elements=int(max_el),
                                )
                            st.caption(
                                f"Çizilen eleman: {stats['drawn']}  ·  atlanan: {stats['skipped']}"
                                + (f"  ·  ihlal: {len(hl_guids)}"
                                   f"  ·  decoy: {len(dc_guids)}"
                                   if (hl_guids or dc_guids) and highlight else "")
                            )
                            st.plotly_chart(
                                fig, use_container_width=True,
                                config={"scrollZoom": True},
                            )
                        except Exception as e:
                            st.error(f"3D görselleştirme hatası: {e}")

                with view_graph:
                    gpath = m.get("graph_path")
                    needs_build = not gpath or not Path(gpath).exists()
                    if needs_build:
                        st.info("Bu IFC için graph kaydedilmemiş.")
                    if st.button(
                        ("Şimdi üret ve kaydet" if needs_build
                         else "Graph'ı yeniden üret (layout dahil)"),
                        key=f"gbuild_{sel}",
                    ):
                        try:
                            base = Path(m["file_path"]).with_suffix("")
                            gp = base.with_suffix(".graph.json")
                            ifc_graph.build_and_save(m["file_path"], gp)
                            storage.set_ifc_graph_path(sel, str(gp))
                            st.success(f"Graph (yeniden) üretildi: {gp}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Graph üretme hatası: {e}")
                    if not needs_build:
                        hl_graph = st.checkbox(
                            "Modifiye/eklenen düğümleri vurgula",
                            value=True, key=f"hlg_{sel}",
                            disabled=(kind != "violated"),
                        )
                        if st.button("Graph çiz", key=f"drawg_{sel}"):
                            try:
                                with st.spinner("Graph çiziliyor..."):
                                    g = ifc_graph.load_graph(gpath)
                                    gfig = graph_viewer.graph_to_figure(
                                        g,
                                        highlight_guids=(hl_guids if hl_graph else set()),
                                        decoy_guids=(dc_guids if hl_graph else set()),
                                    )
                                st.caption(
                                    f"Düğüm: {g.number_of_nodes()}  ·  "
                                    f"kenar: {g.number_of_edges()}"
                                    + (f"  ·  ihlal: {len(hl_guids)}"
                                       f"  ·  decoy: {len(dc_guids)}"
                                       if (hl_guids or dc_guids) and hl_graph else "")
                                )
                                st.info(
                                    "🖱️ Çekmek için fareyle sürükle (pan modu), "
                                    "kaydırma tekerleğiyle yakınlaştır.",
                                    icon="ℹ️",
                                )
                                st.plotly_chart(
                                    gfig, use_container_width=True,
                                    config={"scrollZoom": True,
                                            "displaylogo": False},
                                )
                            except Exception as e:
                                st.error(f"Graph görselleştirme hatası: {e}")
                        with open(gpath, "rb") as f:
                            st.download_button(
                                "Graph JSON indir", f,
                                file_name=Path(gpath).name, key=f"dlg_{sel}",
                            )

            if st.button("Bu kaydı sil", key=f"del_ifc_{sel}"):
                storage.delete_ifc_model(sel)
                st.rerun()
