from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np 
from openai import OpenAI

class DeepseekAPIGenerator:
    def __init__(self, embedder=None):
        self.embedder = embedder
    def generate(self, query, retrievalChunks: List[str]) -> str:
        client = OpenAI(api_key="sk-fea7631fae524eb98ef05ebe4d82667c", base_url="https://api.deepseek.com")
        llm_model = "deepseek-reasoner"
        context = ""
        for i, chunk in enumerate(retrievalChunks):
            context += f"reference infomation {i+1}: \n{chunk}\n\n"

        prompt = f"根据参考文档回答问题：{query}\n\n{context}"
        messages=[{"role": "system", "content": ""},
                {"role": "user", "content": f"{prompt}"},]

        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=messages,
                stream=False
            )
            return response
        except Exception as e:
            raise ValueError(f"DeepseekAPIGenerator_retrieval error: {e}")


GENERATOR_MAPPING = {
    "DeepseekAPIGenerator": (DeepseekAPIGenerator),
}


class Generator:
    def __init__(self, embedder=None, config: dict=None):
        self.config = config
        self.generator = None
        self._init_components(embedder)

    def _init_components(self, embedder):
        # init generator
        generator_cfg = self.config.get("generator", {})
        self.generator = self._get_generator(embedder, generator_cfg)

    def _get_generator(self, embedder, config: dict):
        generator_type = config.get("type", "generator")
        params = config.get("params", {})
        generator = GENERATOR_MAPPING.get(generator_type)
        if generator is None:
            raise ValueError(f"Indexer_get_chunker -> Unknown chunker type: {generator_type}")
        # 实例化
        return generator(embedder)
