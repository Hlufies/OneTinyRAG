from pathlib import Path
import importlib
import pkgutil
from types import ModuleType
from typing import List
from langchain_community.vectorstores import FAISS
import faiss 
import numpy as np 

from .DataProcessor import DataProcessor, PdfProcessor, TxtProcessor
from .Chunker import Chunker, RecursiveChunker, TokenChunker, SemanticSpacyChunker, SemanticNLTKChunker
from .Embedder import Embedder, HuggingFaceEmbedder, BAAIEmbedder

DOCUMENT_LOADER_MAPPING = {
    ".pdf": (PdfProcessor, {}),
    ".txt": (TxtProcessor, {"encoding": "utf8"}),
}

CHUNER_MAPPING = {
    "recursive": (RecursiveChunker),
    "token": (TokenChunker),
    "SemanticSpacyChunker" : (SemanticSpacyChunker),
    "SemanticNLTKChunker" : (SemanticNLTKChunker)
}

EMBEDDER_MAPPING = {
    "BAAIEmbedder": (BAAIEmbedder),
    "HuggingFaceEmbedder": (HuggingFaceEmbedder),
}

class Indexer:
    def __init__(self, config: dict):
        self.config = config
        self._init_components()
        self.Chunker = None
        self.Embedder = None
        self._init_components()

    def _init_components(self):
        # init chunker
        chunker_cfg = self.config.get("chunker", {})
        self.Chunker = self._get_chunker(chunker_cfg)

        # init embedder
        embedder_cfg = self.config.get("embedder", {})
        self.Embedder = self._get_embedder(embedder_cfg)

    def _get_data_processor(self, file_path: str) -> str:
         # auto load processor by suffix
        ext = Path(file_path).suffix.lower()
        loader_mapping = DOCUMENT_LOADER_MAPPING.get(ext)
        if loader_mapping is None:
            raise ValueError(f"Indexer_get_data_processor error: {ext}")
        processor, loader_args = loader_mapping
        return processor(**loader_args).process(file_path)


    def _get_chunker(self, config: dict) -> Chunker:
        chunker_type = config.get("type", "recursive")
        params = config.get("params", {})
        chunker = CHUNER_MAPPING.get(chunker_type)
        if chunker is None:
            raise ValueError(f"Indexer_get_chunker -> Unknown chunker type: {chunker_type}")
        # 实例化
        return chunker(**params)

    def _get_embedder(self, config: dict) -> Embedder:
        embedder_type = config.get("type", "BAAIEmbedder")
        params = config.get("params", {})
        embedder = EMBEDDER_MAPPING.get(embedder_type)
        if embedder is None:
            raise ValueError(f"Indexer_get_embedder -> Unknown embedder type: {embedder_type}")
        # 实例化
        return embedder(**params)
    def index(self, file_path: str) -> List:
        text = self._get_data_processor(file_path)
        chunks = self.Chunker.chunk(text)
        return self.Embedder.embed(chunks), chunks
     
        
        

    
    

