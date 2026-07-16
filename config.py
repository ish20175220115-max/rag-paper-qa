import os
from dotenv import load_dotenv
import dashscope

load_dotenv()


def _get_config(key: str, default: str = None) -> str:
    """优先从 Streamlit Cloud secrets 读取，fallback 到环境变量"""
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


DASHSCOPE_API_KEY = _get_config("DASHSCOPE_API_KEY")
TAVILY_API_KEY = _get_config("TAVILY_API_KEY")

dashscope.api_key = DASHSCOPE_API_KEY

EMBEDDING_MODEL = "text-embedding-v4"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
CHROMA_PERSIST_DIR = "./chroma_db_v2"
SUPPORTED_FILE_TYPES = ["pdf", "txt", "csv", "json"]

ALIBABA_CLOUD_ACCESS_KEY_ID = _get_config("ALIBABA_CLOUD_ACCESS_KEY_ID")
ALIBABA_CLOUD_ACCESS_KEY_SECRET = _get_config("ALIBABA_CLOUD_ACCESS_KEY_SECRET")

MINERU_API_TOKEN = _get_config("MINERU_API_TOKEN")
