import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # 경로 설정
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_PATH = os.path.join(BASE_DIR, "data", "items.json")
    CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

    # RAG 파라미터 설정 (설정 파일로 분리하여 조정 가능)
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    DEFAULT_TOP_K = 4

    # 모델 설정
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = "gpt-4o-mini"