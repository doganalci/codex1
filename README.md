# İhlal Havuzu Oluşturucu

Üç farklı yöntemle (LLM ile) yapı denetim mevzuatına yönelik **ihlal kuralı havuzu**
üretip karşılaştırmak için basit bir Streamlit uygulaması.

## Yöntemler

1. **Tek promt** — Kullanıcının yazdığı promt LLM'e olduğu gibi gönderilir; ek
   talimat / şablon yoktur. Sadece çıktıyı parse edebilmek için minimum bir JSON
   yönergesi eklenir.
2. **İyileştirilmiş tek promt** — Sabit ve mühendislenmiş bir system promtu
   (`prompts.OPTIMIZED_PROMPT`) + kullanıcının promtu. JSON şeması,
   kategori/şiddet/eşik kuralları içerir.
3. **Standart dosyalar + RAG + LLM** — Yöntem 2 ile **aynı promt** kullanılır;
   ek olarak seçilen vektör koleksiyonundan ilgili parçalar getirilip bağlama
   eklenir. Evidence (doküman, sayfa, alıntı) bu yöntemde gerçek belgelerden gelir.
4. **Standart dosyalar + Fine-tune LLM** — Standart dokümanlardan sentetik
   eğitim verisi üretilip OpenAI fine-tune işi başlatılır; sonuçta alınan
   FT model id'siyle (yine Yöntem 2'nin promtu kullanılarak) ihlal havuzu üretilir.

> Yöntem 2, 3 ve 4 **aynı promtu** paylaşır — modeller arası karşılaştırma
> bu sayede adil olur.

## Kurulum (conda)

```bash
conda create -n violation-pool python=3.11 -y
conda activate violation-pool
pip install -r requirements.txt
cp .env.example .env   # OPENAI_API_KEY vb. doldur
streamlit run app.py
```

## Akış

1. **RAG / Doküman** sekmesinden PDF'leri yükle, bir **koleksiyon adı** ver
   (örn. `imar-yonetmeligi-2024`). Bu ad sonradan RAG/FT çalıştırmasında seçilecek.
2. (Yöntem 4 için) **Fine-tune** sekmesinden seçtiğin koleksiyondan eğitim
   verisi üret, FT işini başlat. Bittiğinde model id'yi al.
3. **Çalıştır** sekmesinden yöntemi seç, promtu yaz, çalıştır. Yöntem 4'te FT
   model id'sini gir.
4. Sonuçları önizle. **Kaydet** dersen havuz kalıcılaşır ve Excel'e yazılır.
   **İptal et** dersen taslak silinir.
5. Aynı isimle yarım kalmış bir çalıştırma varsa, başlatırken **devam et / sıfırdan
   başla** sorulur.
6. Geçmiş havuzlar sol menüde listelenir; oradan açabilir, Excel alabilir veya silebilirsin.

## Saklanan veriler

- `violation_pool.sqlite` — runs (yöntem, promt, model, embedding, koleksiyon,
  doküman listesi, FT model id, tarih), violations, evidence (document, page,
  clause, snippet).
- `vectorstore/` — Chroma kalıcı vektör veritabanı (koleksiyon = isim).
- `exports/` — `violations_<name>_<id>.xlsx` (3 sayfa: violations, evidence,
  run_meta) ve FT için `ft_<koleksiyon>.jsonl` eğitim verisi.
- `data/` — yüklenen PDF/TXT kopyaları.

## Notlar

- LLM ve embedding modeli `.env`'den geliyor; UI'dan da değiştirilebilir.
- OpenAI uyumlu herhangi bir endpoint kullanılabilir (`OPENAI_BASE_URL`).
- Uzman jüri / judge katmanı şimdilik yok — sonraya bırakıldı.
