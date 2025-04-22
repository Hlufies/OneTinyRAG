from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np 
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

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

class BLIPEmbedder(Embedder):
    def __init__(self, model_name="blip2-opt-2.7b"):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,  # 半精度节省显存
            device_map=self.device           # 自动分配设备（CPU/GPU）
        )
        self.embedder = [self.process, self.model]
    def embed(self, imgs):
        embedder = []
        for img in imgs:
            inputs = self.processor(images=img, return_tensors="pt").to(self.device, torch.float16)
            with torch.no_grad():
                embedding = self.embedder.vision_model(**inputs).last_hidden_state[:, 0, :].cpu().numpy()                
            embedder.append(embedding.squeeze(0))
        embedder = np.array(embedder)
        dimension = embedder.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embedder)
        return index

