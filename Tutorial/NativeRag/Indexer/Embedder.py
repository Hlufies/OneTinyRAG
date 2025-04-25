from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np 

class Embedder(ABC):
    """Base class for embedding models"""
    @abstractmethod
    def embed(self, docs: List[Document]) -> FAISS:
        pass

class BAAIEmbedder(Embedder):
    def __init__(self, model_name="sentence-transformers/all-mpnet-base-v2"):
        self.embedder = SentenceTransformer(model_name)
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

