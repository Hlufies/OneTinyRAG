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
from Mappers.Mappers import RETRIEVER_MAPPING

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
    
