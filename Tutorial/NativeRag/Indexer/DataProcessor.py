from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
from langchain_community.document_loaders import PyPDFLoader
import re

class DataProcessor(ABC):
    """Base class for all data processors"""
    @abstractmethod
    def process(self, file_path: str) -> List[Document]:
        pass

class PdfProcessor(DataProcessor):
    def process(self, file_path: str) -> str:
        try:
            loader = PyPDFLoader(file_path)
            documents = [doc.page_content for doc in loader.load()]
            text = ''
            for document in documents:
                text += clean_text(document)
            return text
        except Exception as e:
            raise ValueError(f"PdfProcessor error: {e}")

class TxtProcessor(DataProcessor):
    def process(self, file_path: str=None, encoding='utf8') -> str:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = ''
                for line in f:
                    text += clean_text(line)
            return text
        except Exception as e:
            raise ValueError(f"TxtProcessor error: {e}")


def clean_text(text: str) -> str:
    """
    文本清洗函数：
    1. 合并被换行断开的单词（如 xxx-\nxxx → xxxxxx）
    2. 将换行符转换为空格
    """
    # 第一步：处理连字符换行
    text = re.sub(r'-\n', '', text)
    
    # 第二步：处理普通换行
    text = re.sub(r'\n', ' ', text)
    
    return text.strip()