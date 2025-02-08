import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    HUGGINGFACE_API_KEY: str = os.getenv("HUGGINGFACE_API_KEY")
    CHUNK_SIZES: list = [2048, 512, 128]
    EMBED_MODEL: str = "local:BAAI/bge-small-en-v1.5"
    RERANK_MODEL: str = "BAAI/bge-reranker-base"

settings = Settings()
