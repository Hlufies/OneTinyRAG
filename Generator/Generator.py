#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-08 23:33:50
"""

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
from Mappers.Mappers import GENERATOR_MAPPING



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
    def generate(self, query, retrievalChunks: List[str]) -> str:
        [txtChunks, imgChunks] = retrievalChunks

        if txtChunks is not None and imgChunks is None:
            # 仅文本生成, 目前仅实现如此
            result = self.generator.generate(query, txtChunks)
            return result
        elif txtChunks is not None and imgChunks is not None:
            pass
        else:
            raise ValueError(f"Generator_generate -> Unknown generator type: {self.Generator}")


