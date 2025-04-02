from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
import spacy
from langchain.text_splitter import TextSplitter
from typing import List, Optional
from typing import List
import nltk
from nltk.tokenize import sent_tokenize
import jieba

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



class SemanticSpacyChunker(Chunker):
    """基于spaCy语义分析的智能文本分割器"""
    def __init__(
        self,
        model_name: str = "zh_core_web_sm",  # 支持中英文模型切换： en_core_web_sm
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        use_sentence: bool = True  # 是否基于句子拆分
    ):
        self.nlp = spacy.load(model_name)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_sentence = use_sentence

    def split_text(self, text: str) -> List[str]:
        """核心分割逻辑"""
        doc = self.nlp(text)
        
        if self.use_sentence:
            sentences = [sent.text for sent in doc.sents]
        else:
            sentences = [token.text for token in doc if not token.is_punct]
        # 动态合并句子/词块
        current_chunk = []
        current_length = 0
        chunks = []

        for sent in sentences:
            sent_length = len(sent)
            
            # 判断是否超过阈值
            if current_length + sent_length > self.chunk_size:
                if current_chunk:
                    # 中文用空字符串连接
                    chunks.append("".join(current_chunk))
                    
                    # 精确计算重叠字符数
                    overlap_buffer = []
                    overlap_length = 0
                    # 逆向遍历寻找重叠边界
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) > self.chunk_overlap:
                            break
                        overlap_buffer.append(s)
                        overlap_length += len(s)
                    # 恢复原始顺序
                    current_chunk = list(reversed(overlap_buffer))
                    current_length = overlap_length
                    
            current_chunk.append(sent)
            current_length += sent_length

        # 处理剩余内容
        if current_chunk:
            chunks.append("".join(current_chunk))
        return chunks

    def chunk(self, docs: str) -> List[str]:
        """文档处理入口"""
        chunks = self.split_text(docs)
        return chunks


class SemanticNLTKChunker(Chunker):
    """基于NLTK的智能语义分块器，支持中英文混合文本"""
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        language: str = "chinese",
        use_jieba: bool = True
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language = language
        self.use_jieba = use_jieba

        # 初始化中文分词器
        if self.language == "chinese" and self.use_jieba:
            jieba.initialize()
    def _chinese_sentence_split(self, text: str) -> List[str]:
        """基于结巴分词的智能分句"""
        if not self.use_jieba:
            return [text]
            
        delimiters = {'。', '！', '？', '；', '…'}
        sentences = []
        buffer = []
        
        for word in jieba.cut(text):
            buffer.append(word)
            if word in delimiters:
                sentences.append(''.join(buffer))
                buffer = []
        
        if buffer:  # 处理末尾无标点的句子
            sentences.append(''.join(buffer))
        return sentences

    def split_text(self, text: str) -> List[str]:
        """多语言分句逻辑"""
        sentences = []
        if self.language == "chinese":
            sentences =  self._chinese_sentence_split(text)
        else:
            nltk.download('punkt_tab')
            sentences =  sent_tokenize(text, language=self.language)

        """动态合并句子并保留字符重叠"""
        chunks = []
        current_chunk = []
        current_length = 0
        overlap_buffer = []

        for sent in sentences:
            sent_len = len(sent)
            
            # 触发分块条件
            if current_length + sent_len > self.chunk_size:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    
                    # 计算重叠部分
                    overlap_buffer = []
                    overlap_length = 0
                    for s in reversed(current_chunk):
                        if overlap_length + len(s) > self.chunk_overlap:
                            break
                        overlap_buffer.append(s)
                        overlap_length += len(s)
                        
                    current_chunk = list(reversed(overlap_buffer))
                    current_length = overlap_length
            
            current_chunk.append(sent)
            current_length += sent_len

        for chunk in chunks:
            print(len(chunk))

        # 处理剩余内容
        if current_chunk:
            chunks.append("".join(current_chunk))
        return chunks

    def chunk(self, docs: str) -> List[str]:
        chunks = self.split_text(docs)
        return chunks

    