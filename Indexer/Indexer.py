#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-08 23:33:50
"""

from pathlib import Path
import importlib
import pkgutil
from typing import List
import os

from .DataProcessor import PdfProcessor, TxtProcessor, JsonProcessor
from .Chunker import Chunker, RecursiveChunker, TokenChunker, SemanticSpacyChunker, SemanticNLTKChunker, MetaDataChunker
from .Embedder import Embedder, HuggingFaceEmbedder, BAAIEmbedder, MetaDataEmbedder
from .Agent import CodeAutoAgent

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

class Indexer:
    def __init__(self, config: dict):
        self.config = config
        self.Chunker = None
        self.DocEmbedder = None
        self._init_components()

    def _init_components(self):
        # init Chunker
        chunker_cfg = self.config.get("chunker", {})
        self.Chunker = self._get_chunker(chunker_cfg)

        # init docEmbedder
        embedder_cfg = self.config.get("embedder", {})
        self.DocEmbedder = self._get_Embedder(embedder_cfg)

    def _get_data_processor(self, file_path: str) -> List:
         # auto load processor by suffix  
        if os.path.isdir(file_path) :
            results = []
            for filename in os.listdir(file_path):
                if filename.startswith('.'):
                    continue
                full_path = os.path.join(file_path, filename)
                try:
                    if os.path.isfile(full_path):
                        print("processing file path:", full_path)
                    results += self._get_data_processor(full_path)
                except ValueError as e:
                    print(f"Skipped {full_path}: {str(e)}")
                    continue
            return results
        else:
            ext = Path(file_path).suffix.lower()
            loader_mapping = LOADER_MAPPING.get(ext)
            loader_mapping_counter = 0
            while loader_mapping is None and loader_mapping_counter < 10:
                # # start LLMs' Agent
                loader_mapping_counter += 1
                return []
            processor, loader_args = loader_mapping
            # 返回处理后的文件 + 后缀用来表示是图像还是文本
            return [(processor(**loader_args).process(file_path), file_path.split('.')[-1])]


    def _get_chunker(self, config: dict) -> Chunker:
        chunker_type = config.get("type", "recursive")
        params = config.get("params", {})
        chunker = CHUNER_MAPPING.get(chunker_type)
        if chunker is None:
            raise ValueError(f"Indexer_get_chunker -> Unknown chunker type: {chunker_type}")
        # 实例化
        return chunker(**params)

    def _get_Embedder(self, config: dict) -> Embedder:
        docEmbedder_config = config.get("docEmbedder", {})
        docEmbedder_type = docEmbedder_config.get("type", "BAAIEmbedder")
        docParams = docEmbedder_config.get("params", {})
        docEmbedder = EMBEDDER_MAPPING.get(docEmbedder_type)
        # 实例化
        if docEmbedder is None:
            raise ValueError(f"Indexer_get_Embedder -> Unknown embedder type: {docEmbedder_type}")    
        return docEmbedder(**docParams)
    
    def index(self, file_path: str) -> List:
        datas = self._get_data_processor(file_path)
        chunks = []
        for (data, type) in datas:
            if type in ['jpg', 'jpeg', 'png']:
                pass # 多模态信息除了图像就是文本, unprocess
            else:
                chunks += self.Chunker.chunk(data)
        
        if self.DocEmbedder is not None:
            docEmb = self.DocEmbedder.embed(chunks)
        else:
            docEmb = None
        print(docEmb, chunks)
        exit(0)
        return docEmb, chunks
        
        

    
    

