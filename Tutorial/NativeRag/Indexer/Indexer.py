from pathlib import Path
import importlib
import pkgutil
from types import ModuleType
from typing import List
from langchain_community.vectorstores import FAISS
import faiss 
import numpy as np 
import os
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

from .DataProcessor import PdfProcessor, TxtProcessor
from .DataProcessor import ImgProcessor
from .Chunker import Chunker, RecursiveChunker, TokenChunker, SemanticSpacyChunker, SemanticNLTKChunker
from .Embedder import Embedder, HuggingFaceEmbedder, BAAIEmbedder, BLIPEmbedder
from .Agent import CodeAutoAgent

LOADER_MAPPING = {
    ".pdf": (PdfProcessor, {}),
    ".txt": (TxtProcessor, {"encoding": "utf8"}),

    ".jpg": (ImgProcessor, {}),
    ".jpeg": (ImgProcessor, {}),
    ".png": (ImgProcessor, {}),
}

CHUNER_MAPPING = {
    "recursive": (RecursiveChunker),
    "token": (TokenChunker),
    "SemanticSpacyChunker" : (SemanticSpacyChunker),
    "SemanticNLTKChunker" : (SemanticNLTKChunker),
 
}

EMBEDDER_MAPPING = {
    "BAAIEmbedder": (BAAIEmbedder),
    "HuggingFaceEmbedder": (HuggingFaceEmbedder),
    "Salesforce/blip2-opt-2.7b": (BLIPEmbedder)
}

class Indexer:
    def __init__(self, config: dict):
        self.config = config
        self.Chunker = None
        self.DocEmbedder = None
        self.ImgEmbedder = None
        self._init_components()

    def _init_components(self):
        # init chunker
        chunker_cfg = self.config.get("chunker", {})
        self.Chunker = self._get_chunker(chunker_cfg)

        # init embedder
        embedder_cfg = self.config.get("embedder", {})
        [self.DocEmbedder, self.ImgEmbedder] = self._get_Embedder(embedder_cfg)



    def _get_data_processor(self, file_path: str) -> List:
         # auto load processor by suffix  
        if os.path.isdir(file_path) :
            results = []
            for filename in os.listdir(file_path):
                if filename.startswith('.'):
                    continue
                full_path = os.path.join(file_path, filename)
                try:
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
        imgEmbedder_config = config.get("imgEmbedder", {})

        docEmbedder_type = docEmbedder_config.get("type", "BAAIEmbedder")
        docParams = docEmbedder_config.get("params", {})
        docEmbedder = EMBEDDER_MAPPING.get(docEmbedder_type)
        
        if docEmbedder is None:
            raise ValueError(f"Indexer_get_Embedder -> Unknown embedder type: {docEmbedder_type}")
        
        imgEmbedder_type = imgEmbedder_config.get("type", "")
        imgParams = imgEmbedder_config.get("params", {})
        imgEmbedder = EMBEDDER_MAPPING.get(imgEmbedder_type)
        
        if imgEmbedder is None:
            raise ValueError(f"Indexer_get_Embedder -> Unknown embedder type: {imgEmbedder_type}")
        
        # 实例化
        return docEmbedder(**docParams), imgEmbedder(**imgParams)
    
    def index(self, file_path: str) -> List:
        datas = self._get_data_processor(file_path)
        chunks = []
        images = []
        # 这里主要是为了区分文本和图像
        for (data, type) in datas:
            if type in ['jpg', 'jpeg', 'png']:
                images.append(data)
            else:
                chunks += self.Chunker.chunk(data)
        # docEmb = self.DocEmbedder.embed(chunks)
        # imgEmb = self.ImgEmbedder.embed(images)
        
        return self.DocEmbedder.embed(chunks), chunks, self.ImgEmbedder.embed(images), images, 
     
        
        

    
    

