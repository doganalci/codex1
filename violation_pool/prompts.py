"""Default prompts used by the violation pool generator.

- NAIVE_PROMPT: kullanıcının kendi yazdığı ham talimat hissi verir; LLM'e
  hiçbir ek şablon / şema dayatmaz. (Yöntem 1)
- OPTIMIZED_PROMPT: aynı görevi rol, kurallar, JSON şeması ve örnek ile
  net biçimde ifade eder. Yöntem 2, 3 (RAG) ve 4 (FT) bunu kullanır.

Konu: yapılı çevrede ERİŞİLEBİLİRLİK ve KULLANILABİLİRLİK
(TS 9111, TS ISO 21542 ve ilgili Türk mevzuatı çerçevesinde).
Yapı denetim / statik / yangın / elektrik / mekanik konuları kapsam DIŞI.
"""

NAIVE_PROMPT = (
    "Yapılı çevrede karşılaşılabilecek erişilebilirlik ve kullanılabilirlik "
    "ihlali kurallarını listele. Her ihlali kısa bir cümleyle yaz."
)

OPTIMIZED_PROMPT = """Sen, yapılı çevrede erişilebilirlik ve kullanılabilirlik
(özellikle TS 9111, TS ISO 21542 ve ilgili Türk mevzuatı) konusunda uzman bir
analistin. Görevin, verilen bağlama göre somut, ölçülebilir ve tek başına
anlaşılır "erişilebilirlik / kullanılabilirlik ihlali" kuralları üretmektir.

Kapsam: yaya erişimi, giriş, kapı/koridor genişlikleri, rampa eğimi,
merdiven, korkuluk/küpeşte, asansör, tuvalet/banyo, mutfak ulaşılabilirliği,
otopark, uyarı yüzeyleri, yönlendirme/işaretleme, görsel/işitsel/dokunsal
ipuçları, kontrast, aydınlatma, manevra alanları, eşik/kot farkları.
(Yapı denetimi, statik, yangın, elektrik, mekanik konularına GİRME — sadece
erişilebilirlik ve kullanılabilirlik.)

Kurallar:
- Her ihlal tek bir somut durumu tanımlasın (örn. "Kapı net geçiş
  genişliğinin 90 cm'den küçük olması").
- Sayısal eşik varsa birimiyle birlikte ver (cm, m, %, lux, vb.).
- Belirsiz ifadelerden ("uygun olmayan", "yeterli olmayan") kaçın.
- Her ihlal için kategori belirt; şu setten seç:
  "Yaya erişimi" | "Giriş" | "Kapı/Koridor" | "Rampa" | "Merdiven" |
  "Korkuluk/Küpeşte" | "Asansör" | "Tuvalet/Banyo" | "Mutfak" |
  "Otopark" | "Uyarı yüzeyi" | "Yönlendirme/İşaretleme" |
  "Görsel/Kontrast" | "Aydınlatma" | "Manevra alanı" | "Eşik/Kot farkı"
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
      "threshold": "varsa sayısal eşik (örn. '< 90 cm') yoksa null",
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
