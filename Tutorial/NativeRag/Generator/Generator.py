from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np 
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA

class DeepseekAPIGenerator:
    def __init__(self):
        super().__init__()
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


class DeepseekOllamaGenerator:
    def __init__(self):
        super().__init__()
    def generate(self, query, retrievalChunks: List[str]) -> str:
        llm = Ollama(model="deepseek-r1:70b")
        context = ""
        for i, chunk in enumerate(retrievalChunks):
            context += f"reference infomation {i+1}: \n{chunk}\n\n"

        templatel_prompt = "根据参考文档回答问题{query}\n\n{context}"
        
        # 创建RAG Prompt模板
        QA_PROMPT = PromptTemplate(
            input_variables=["query", "context"],
            template=templatel_prompt
        )

        analysis_chain = LLMChain(
            llm=llm,
            prompt=QA_PROMPT,
            verbose=True
        )

        try:
            # 执行分析流程
            response = analysis_chain.invoke({
                "context": context,
                "query": query,
            })
            return response
        except Exception as e:
            raise ValueError(f"DeepseekAPIGenerator_retrieval error: {e}")


GENERATOR_MAPPING = {
    "DeepseekAPIGenerator": (DeepseekAPIGenerator),
    "DeepseekOllamaGenerator": (DeepseekOllamaGenerator),
}


class Generator:
    def __init__(self, config: dict=None):
        self.config = config
        self.Generator = None
        self._init_components()

    def _init_components(self):
        # init generator
        generator_cfg = self.config.get("generator", {})
        self.Generator = self._get_generator(generator_cfg)

    def _get_generator(self, config: dict):
        generator_type = config.get("type", "generator")
        params = config.get("params", {})
        generator = GENERATOR_MAPPING.get(generator_type)
        if generator is None:
            raise ValueError(f"Indexer_get_chunker -> Unknown chunker type: {generator_type}")
        # 实例化
        return generator()
