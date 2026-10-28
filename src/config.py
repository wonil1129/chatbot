import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_TABLE_DIR = BASE_DIR / "chroma_db_koe5"
CHROMA_DIR = CHROMA_TABLE_DIR if CHROMA_TABLE_DIR.exists() else (BASE_DIR / "chroma_db")

# 화면 이름은 환경변수로 바꿀 수 있다 (기본값은 학교명 없는 일반 이름)
APP_TITLE = os.getenv("APP_TITLE", "학사규정 안내 챗봇")
APP_ICON = "🎓"
APP_HEADER = os.getenv("APP_HEADER", "🤖 학사규정 안내 챗봇")
APP_LOGO = os.getenv("APP_LOGO", "logo.png")  # assets/ 안의 파일명, 없으면 로고 없이 표시

# 최적화된 한국어 임베딩 및 리랭커 모델
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

EMBEDDING_MODEL = "nlpai-lab/KoE5"
DEFAULT_OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "exaone3.5:7.8b")


def apply_runtime_env() -> None:
    os.environ.setdefault("OLLAMA_HOST", "http://127.0.0.1:11434")