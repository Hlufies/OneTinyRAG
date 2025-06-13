#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-12 23:41:04
    

"""

from Tools.Utils import ApiQuery, OllamaDeepseekQuery
from Tools.Workflow import task_func_default

from Indexer.DataProcessor import PdfProcessor, TxtProcessor, JsonProcessor
from Indexer.Chunker import Chunker, RecursiveChunker, TokenChunker, SemanticSpacyChunker, SemanticNLTKChunker, MetaDataChunker
from Indexer.Embedder import Embedder, HuggingFaceEmbedder, BAAIEmbedder, MetaDataEmbedder

from Agent.Agent import CodeAutoAgent

from Retriever.Retrieval import CosinRetriever

from Generator.Generate import DeepseekAPIGenerator, DeepseekOllamaGenerator


# Query Mapper ======================================
QUERY_MODEL_MAPPING = {
    "ApiQuery" : ApiQuery,
    "OllamaDeepseekQuery" : OllamaDeepseekQuery
}

TASK_FUNC_MAPPING = {
    "task_func_default" : task_func_default
}

TASK_LLM_MAPPING = {
    "ApiQuery" : ApiQuery,
    "OllamaDeepseekQuery" : OllamaDeepseekQuery
}


# Indexer Mapper ======================================
LOADER_MAPPING = {
    ".pdf": (PdfProcessor, {}),
    ".txt": (TxtProcessor, {"encoding": "utf8"}),
    ".json" : (JsonProcessor, {})
}

CHUNER_MAPPING = {
    "recursive": (RecursiveChunker),
    "token": (TokenChunker),
    "SemanticSpacyChunker" : (SemanticSpacyChunker),
    "SemanticNLTKChunker" : (SemanticNLTKChunker),
    "MetaDataChunker": (MetaDataChunker),
}

EMBEDDER_MAPPING = {
    "BAAIEmbedder": (BAAIEmbedder),
    "HuggingFaceEmbedder": (HuggingFaceEmbedder),
    "MetaDataEmbedder": (MetaDataEmbedder)
}

# Retriever Mapper ======================================
RETRIEVER_MAPPING = {
    "CosinRetriever": (CosinRetriever),
}


# Generator Mapper ======================================
GENERATOR_MAPPING = {
    "DeepseekAPIGenerator": (DeepseekAPIGenerator),
    "DeepseekOllamaGenerator": (DeepseekOllamaGenerator),
}
