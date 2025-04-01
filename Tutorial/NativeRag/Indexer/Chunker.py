from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document

class Chunker(ABC):
    @abstractmethod
    def chunk(self, docs: List[Document]) -> List[Document]:
        pass

class RecursiveChunker(Chunker):
    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n## ", "\n# ", "\n\n", "\n", "。", "!", "?", " ", ""]
        )

    def chunk(self, docs: str) -> List:
        return self.splitter.split_text(docs)

class TokenChunker(Chunker):
    def __init__(self, chunk_size=512, chunk_overlap=64):
        self.splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    def chunk(self, docs: List[Document]) -> List[Document]:
        return self.splitter.split_documents(docs)