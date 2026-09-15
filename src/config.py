from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv
import os

PROJECT_DIR = Path(__file__).resolve().parents[1]
ROOT_DIR = PROJECT_DIR
DATA_DIR = PROJECT_DIR / "data"
REFERENCE_DOCS_DIR = DATA_DIR / "reference_docs"
OUTPUTS_DIR = PROJECT_DIR / "outputs"
MODELS_DIR = PROJECT_DIR / "models"

# The application .env lives next to app.py at the project root.
load_dotenv(PROJECT_DIR / ".env")

class Settings(BaseModel):
    llm_provider: str = os.getenv("LLM_PROVIDER", "mock").lower()
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2")
    ollama_model_c1: str = os.getenv("OLLAMA_MODEL_C1", os.getenv("OLLAMA_MODEL", "llama3.2"))
    ollama_model_c2: str = os.getenv("OLLAMA_MODEL_C2", "llama3.2")
    ollama_model_c3: str = os.getenv("OLLAMA_MODEL_C3", "qwen2.5:3b")
    ollama_model_c4: str = os.getenv("OLLAMA_MODEL_C4", "mistral")
    ollama_model_c5: str = os.getenv("OLLAMA_MODEL_C5", "mistral")
    ollama_model_c6: str = os.getenv("OLLAMA_MODEL_C6", "llama3.2")
    ollama_model_c7: str = os.getenv("OLLAMA_MODEL_C7", "mistral")
    ollama_timeout_seconds: int = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "6144"))
    ollama_keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "2m")
    llm_document_max_chars: int = int(os.getenv("LLM_DOCUMENT_MAX_CHARS", "22000"))
    ocr_max_pages: int = int(os.getenv("OCR_MAX_PAGES", "40"))
    ocr_target_chars: int = int(os.getenv("OCR_TARGET_CHARS", "80000"))
    ocr_render_scale: float = float(os.getenv("OCR_RENDER_SCALE", "1.15"))
    rag_backend: str = os.getenv("RAG_BACKEND", "chroma").lower()
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma_db")
    chroma_collection: str = os.getenv(
        "CHROMA_COLLECTION",
        "conops_reference_docs",
    )
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    finetuned_model_path: str = os.getenv("FINETUNED_MODEL_PATH", "models/conops-lora")
    chunk_size: int = 1200
    chunk_overlap: int = 180
    top_k: int = 5

    @property
    def chroma_path(self) -> Path:
        path = Path(self.chroma_dir)
        return path if path.is_absolute() else PROJECT_DIR / path

settings = Settings()

AVAILABLE_MODELS = ["llama3.2:latest", "mistral:latest", "qwen2.5:3b"]

STRATEGIES = {
    "C1": {"rag": False, "rules": False, "evaluator": False},
    "C3": {"rag": False, "rules": True, "evaluator": False},
    "C4": {"rag": True, "rules": False, "evaluator": False},
    "C5": {"rag": True, "rules": True, "evaluator": True},
}

KNOWLEDGE_FORMATS = {
    "K1": "simple_list",
    "K2": "definitions",
    "K3": "ontology",
}

for folder in [DATA_DIR, REFERENCE_DOCS_DIR, OUTPUTS_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)
