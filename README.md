# İhlal Havuzu Oluşturucu

Üç farklı yöntemle (LLM ile) yapı denetim mevzuatına yönelik **ihlal kuralı havuzu**
üretip karşılaştırmak için basit bir Streamlit uygulaması.

## Yöntemler

1. **Naive prompt** — Kullanıcının yazdığı promt LLM'e olduğu gibi gönderilir; ek
   talimat / şablon yoktur. Sadece çıktıyı parse edebilmek için minimum bir JSON
   yönergesi eklenir.
2. **Optimize prompt** — Sabit ve mühendislenmiş bir system promtu (`prompts.OPTIMIZED_PROMPT`)
   + kullanıcının promtu. JSON şeması, kategori/şiddet/eşik kuralları içerir.
3. **RAG + Optimize prompt** — Yöntem 2 ile **aynı promt** kullanılır; ek olarak
   seçilen vektör koleksiyonundan ilgili parçalar getirilip bağlama eklenir.
   Evidence (doküman, sayfa, alıntı) bu yöntemde gerçek belgelerden gelir.

> Yöntem 2 ile Yöntem 3 **aynı promtu** paylaşır — modeller arası karşılaştırma
> bu sayede adil olur.

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # OPENAI_API_KEY vb. doldur
streamlit run app.py
```

## Akış

1. **RAG / Doküman** sekmesinden PDF'leri yükle, bir **koleksiyon adı** ver
   (örn. `imar-yonetmeligi-2024`). Bu ad sonradan RAG çalıştırmasında seçilecek.
2. **Çalıştır** sekmesinden yöntemi seç, promtu yaz, çalıştır.
3. Sonuçları önizle. **Kaydet** dersen havuz kalıcılaşır ve Excel'e yazılır.
   **İptal et** dersen taslak silinir.
4. Aynı isimle yarım kalmış bir çalıştırma varsa, başlatırken **devam et / sıfırdan
   başla** sorulur.
5. Geçmiş havuzlar sol menüde listelenir; oradan açabilir, Excel alabilir veya silebilirsin.

## Saklanan veriler

- `violation_pool.sqlite` — runs (yöntem, promt, model, embedding, koleksiyon,
  doküman listesi, tarih), violations, evidence (document, page, clause, snippet).
- `vectorstore/` — Chroma kalıcı vektör veritabanı (koleksiyon = isim).
- `exports/` — `violations_<name>_<id>.xlsx` (3 sayfa: violations, evidence, run_meta).
- `data/` — yüklenen PDF/TXT kopyaları.

## Notlar

- LLM ve embedding modeli `.env`'den geliyor; UI'dan da değiştirilebilir.
- OpenAI uyumlu herhangi bir endpoint kullanılabilir (`OPENAI_BASE_URL`).
- Uzman jüri / judge katmanı şimdilik yok — sonraya bırakıldı.
