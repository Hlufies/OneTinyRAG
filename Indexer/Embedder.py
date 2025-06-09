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
import torch
from tqdm import tqdm

class Embedder(ABC):
    """Base class for embedding models"""
    @abstractmethod
    def embed(self, docs: List[Document]) -> FAISS:
        pass

class BAAIEmbedder(Embedder):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name)
    def embed(self, chunks: List) -> List:
        embedder = []
        for chunk in chunks:
            if isinstance(chunk, str):
                embedding = self.embedder.encode(chunk, normalize_embedder=True)
            elif isinstance(chunk, Document):
                embedding = self.embedder.encode(chunk['text'], normalize_embedder=True)
            else:
                continue
            embedder.append(embedding)
        embedder = np.array(embedder)
        dimension = embedder.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embedder)
        return index

class HuggingFaceEmbedder(Embedder):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)
    def embed(self, chunks: List[str]) -> List:
        embedder = []
        for chunk in chunks:
            embedding = self.embedder.encode(chunk, normalize_embedder=True)
            embedder.append(embedding)
        embedder = np.array(embedder)
        dimension = embedder.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embedder)
        return index

class MetaDataEmbedder(Embedder):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name)
    def embed(self, chunks: List) -> List:
        assert len(chunks)
        embedder = []
        for chunk in tqdm(chunks, desc="Processing Embedder"):
            if isinstance(chunk, str):
                embedding = self.embedder.encode(chunk, normalize_embedder=True)
            elif isinstance(chunk, Document):
                embedding = self.embedder.encode(chunk['text'], normalize_embedder=True)
            elif isinstance(chunk, dict):
                embedding = self.embedder.encode(chunk['page_content'], normalize_embedder=True)
            else:
                continue
            embedder.append(embedding)
        embedder = np.array(embedder)
        dimension = embedder.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embedder)
        return index

