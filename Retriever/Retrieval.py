#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-12 23:41:04
    

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



