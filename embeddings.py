from typing import List
from dashscope import TextEmbedding
from langchain_core.embeddings import Embeddings
from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL


class DashScopeEmbeddings(Embeddings):
    def __init__(self, model: str = EMBEDDING_MODEL, api_key: str = DASHSCOPE_API_KEY):
        self.model = model
        self.api_key = api_key
        self._max_batch_size = 5

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        clean_texts = []
        for t in texts:
            if not isinstance(t, str):
                t = str(t) if t is not None else ""
            clean_texts.append(t.strip() if t.strip() else " ")

        all_embeddings = []
        for i in range(0, len(clean_texts), self._max_batch_size):
            batch = clean_texts[i: i + self._max_batch_size]
            response = TextEmbedding.call(
                model=self.model,
                input=batch,
                api_key=self.api_key,
            )
            if response.status_code != 200:
                raise RuntimeError(f"嵌入失败: {response.message}")
            all_embeddings.extend(
                item["embedding"] for item in response.output["embeddings"]
            )
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]
