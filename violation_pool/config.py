import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    data_dir: Path = Path(os.getenv("DATA_DIR", "./data"))
    vectorstore_dir: Path = Path(os.getenv("VECTORSTORE_DIR", "./vectorstore"))
    db_path: Path = Path(os.getenv("DB_PATH", "./violation_pool.sqlite"))
    export_dir: Path = Path(os.getenv("EXPORT_DIR", "./exports"))

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()


METHOD_NAIVE = "naive"
METHOD_OPTIMIZED = "optimized"
METHOD_RAG = "rag"
METHOD_FINETUNE = "finetune"

METHOD_LABELS = {
    METHOD_NAIVE: "1) Tek Promt (LLM, ek işlem yok)",
    METHOD_OPTIMIZED: "2) İyileştirilmiş Tek Promt (LLM)",
    METHOD_RAG: "3) Standart dosyalar + RAG + LLM",
    METHOD_FINETUNE: "4) Standart dosyalar + Fine-tune LLM",
}
