#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-08 23:33:50
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import faiss 
import torch
import numpy as np 


class CosinRetriever:
    def __init__(self, embedder=None, index=None):
        self.embedder = embedder
        self.index = index
    def retrieval_txt(self, query, chunks: List[str], top_k: int = 3) -> List:
        query_embedding = self.embedder.encode(query, normalize_embedder=True)
        query_embedding = np.array([query_embedding])

        distances, indices = self.index.search(query_embedding, top_k)
        retrievalChunks = []
        valid_top_k = min(top_k, len(chunks))
        for i in range(valid_top_k):
            # 获取相似文本块的原始内容
            result_chunk = chunks[indices[0][i]]
            # 获取相似文本块的相似度得分
            # result_distance = distances[0][i]
            retrievalChunks.append(result_chunk)
        return retrievalChunks

    def retrieval_img(self, query, chunks: List[str], top_k: int = 3) -> List:
        # 图像embedder -> [self.processor, self.model] = self.embedder
        device = self.embedder[1].device
        inputs = self.embedder[0](text=query, return_tensors="pt").to(device, torch.float16)
        query_embedding = self.embedder[1].language_model(**inputs)
        print(query_embedding)
        print(query_embedding.shape)
        print(query_embedding.shape)
        exit(0)
        query_embedding = np.array([query_embedding])

        distances, indices = self.index.search(query_embedding, top_k)
        retrievalChunks = []
        for i in range(top_k):
            # 获取相似文本块的原始内容
            result_chunk = chunks[indices[0][i]]
            # 获取相似文本块的相似度得分
            # result_distance = distances[0][i]
            retrievalChunks.append(result_chunk)
        return retrievalChunks


RETRIEVER_MAPPING = {
    "CosinRetriever": (CosinRetriever),
}


class Retriever:
    def __init__(self, DocEmbedder=None, ImgEmbedder=None, textIndex=None, imgIndex=None, config: dict=None):
        self.config = config
        self.Retriever = None
        self._init_components(DocEmbedder, ImgEmbedder, textIndex, imgIndex)
        

    def _init_components(self, DocEmbedder, ImgEmbedder, txtIndex, imgIndex):
        # init retriever
        retriever_cfg = self.config.get("retriever", {})
        self.docRetriever = self._get_retriever(DocEmbedder, txtIndex, retriever_cfg)
        self.imgRetriever = self._get_retriever(ImgEmbedder, imgIndex, retriever_cfg)

    def _get_retriever(self, embedder, index, config: dict):
        retriever_type = config.get("type", "recursive")
        params = config.get("params", {})
        retriever = RETRIEVER_MAPPING.get(retriever_type)
        if retriever is None:
            raise ValueError(f"Indexer_get_chunker -> Unknown chunker type: {retriever_type}")
        if embedder is None:
            return None
        return retriever(embedder, index)

    def retrieval(self, query, txtChunks: List, imgChunks: List, top_k: int = 3) -> List:
        # qurey 默认是文本
        retrievalChunks_txt = None
        retrievalChunks_img = None
        if self.docRetriever is not None:
            retrievalChunks_txt = self.docRetriever.retrieval_txt(query, txtChunks, top_k)
        if self.imgRetriever is not None:
            retrievalChunks_img = self.imgRetriever.retrieval_img(query, imgChunks, top_k)
        return [retrievalChunks_txt, retrievalChunks_img]
    
