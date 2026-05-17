"""Default prompts used by the violation pool generator.

- NAIVE_PROMPT: kullanıcının kendi yazdığı ham talimat hissi verir; LLM'e
  hiçbir ek şablon / şema dayatmaz. (Yöntem 1)
- OPTIMIZED_PROMPT: aynı görevi rol, kurallar, JSON şeması ve örnek ile
  net biçimde ifade eder. Yöntem 2 ve Yöntem 3 (RAG) bunu kullanır.
"""

NAIVE_PROMPT = (
    "Yapı / inşaat mevzuatına göre olası ihlal kurallarını listele. "
    "Her ihlali kısa bir cümleyle yaz."
)

OPTIMIZED_PROMPT = """Sen, yapı denetim mevzuatında uzman bir analistin. Görevin, verilen
bağlama göre somut, ölçülebilir ve tek başına anlaşılır "ihlal kuralları"
üretmektir.

Kurallar:
- Her ihlal tek bir somut durumu tanımlasın (örn. "Kapı genişliğinin
  70 cm'den küçük olması").
- Sayısal eşik varsa birimiyle birlikte ver (cm, m, %, kg, vb.).
- Belirsiz ifadelerden ("uygun olmayan", "yeterli olmayan") kaçın.
- Her ihlal için kategori (ör. "Erişilebilirlik", "Yangın güvenliği",
  "Statik", "Elektrik", "Mekanik") belirt.
- Şiddet seviyesi ata: "düşük" | "orta" | "yüksek" | "kritik".
- Mümkünse dayandığın kanıtı (madde no, başlık, sayfa) belirt; bilmiyorsan
  evidence dizisini boş bırak; uydurma.
- Aynı kuralı iki kez yazma.

Çıktıyı SADECE aşağıdaki JSON şemasında, başka hiçbir metin olmadan döndür:

{
  "violations": [
    {
      "title": "kısa başlık",
      "description": "tek cümlelik somut ihlal tanımı",
      "category": "kategori",
      "severity": "düşük|orta|yüksek|kritik",
      "threshold": "varsa sayısal eşik (örn. '< 70 cm') yoksa null",
      "evidence": [
        {"document": "doküman adı veya null",
         "page": "sayfa no veya null",
         "clause": "madde / başlık veya null",
         "snippet": "ilgili kısa alıntı veya null"}
      ]
    }
  ]
}
"""


def build_user_message(user_prompt: str, context_chunks: list[dict] | None = None) -> str:
    """Compose the final user message. For RAG, prepend retrieved chunks."""
    if not context_chunks:
        return user_prompt.strip()

    parts = ["# Bağlam (RAG ile getirildi)\n"]
    for i, ch in enumerate(context_chunks, 1):
        meta = ch.get("metadata", {})
        doc = meta.get("document", "?")
        page = meta.get("page", "?")
        parts.append(f"\n[{i}] doküman={doc} sayfa={page}\n{ch['text']}\n")
    parts.append("\n# Görev\n")
    parts.append(user_prompt.strip())
    parts.append(
        "\n\nNot: evidence alanındaki document/page bilgilerini yalnızca "
        "yukarıdaki bağlamdan al; uydurma."
    )
    return "".join(parts)
