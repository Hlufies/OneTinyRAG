#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-08 23:33:50
"""

from abc import ABC, abstractmethod
from typing import List, Union
from langchain.docstore.document import Document
import re
from typing import Tuple, List
from tqdm import tqdm
import random
from PIL import Image
from PyPDF2 import PdfReader
import json
from langchain_community.document_loaders import (
    PyPDFLoader, 
    PDFPlumberLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
    UnstructuredXMLLoader,
    UnstructuredHTMLLoader,
)

class DataProcessor(ABC):
    """Base class for all data processors"""
    @abstractmethod
    def process(self, file_path: str) -> List[Document]:
        pass

class PdfProcessor(DataProcessor):
    def process(self, file_path: str) -> Union[str, List[str], List[Document]]:
        # rand_num = random.randint(1, 3)
        rand_num = 1
        try:
            if rand_num == 1:
                # 返回 文本
                loader = PyPDFLoader(file_path)
                documents = [doc.page_content for doc in tqdm(loader.load())]
                text = ''
                for document in documents:
                    text += clean_text(document)
                return text
            elif rand_num == 2:
                # 返回 Document 对象
                def get_pdf_metadata(file_path):
                    reader = PdfReader(file_path)
                    metadata = reader.metadata
                    return {k : v for k, v in metadata.items() if v != ''}
                metadata = get_pdf_metadata(file_path)
                documents = [Document(page_content=doc["text"], metadata=metadata) for doc in tqdm(loader.load())]
                return documents
            else:
                # 返回每页的文本列表
                loader = PyPDFLoader(file_path)
                documents = [doc["text"] for doc in tqdm(loader.load())]
                return documents
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

class AutoProcessor(DataProcessor):
    def process(self, file_path):
        DOCUMENT_LOADER_MAPPING = {
            ".pdf": (PDFPlumberLoader, {}),
            ".txt": (TextLoader, {"encoding": "utf8"}),
            ".doc": (UnstructuredWordDocumentLoader, {}),
            ".docx": (UnstructuredWordDocumentLoader, {}),
            ".ppt": (UnstructuredPowerPointLoader, {}),
            ".pptx": (UnstructuredPowerPointLoader, {}),
            ".xlsx": (UnstructuredExcelLoader, {}),
            ".csv": (CSVLoader, {}),
            ".md": (UnstructuredMarkdownLoader, {}),
            ".xml": (UnstructuredXMLLoader, {}),
            ".html": (UnstructuredHTMLLoader, {}),
        }

        ext = Path(file_path).suffix.lower()
        loader_tuple = DOCUMENT_LOADER_MAPPING.get(ext) 

        if loader_tuple:
            processor_class, args = loader_tuple 
            processor = processor_class(file_path, **args) 
            documents = [doc.page_content for doc in loader.load()]
            text = ''
            for document in documents:
                text += clean_text(document)
            return text
        else:
            raise ValueError(f"no match processor error: {e}")

class JsonProcessor(DataProcessor):
    """
    This class is a placeholder for JSON processing about abstract.
    The format and structure:
        "title": "G proteins as drug targets.",
        "abstract": "The structure and function of heterotrimeric G protein subunits is known in considerable detail. Upon stimulation of a heptahelical receptor by the appropriate agonists, the cognate G proteins undergo a cycle of activation and deactivation; the alpha-subunits and the beta gamma-dimers interact sequentially with several reaction partners (receptor, guanine nucleotides and effectors as well as regulatory proteins) by exposing appropriate binding sites. For most of these domains, low molecular weight ligands have been identified that either activate or inhibit signal transduction. These ligands include short peptides derived from receptors, G protein subunits and effectors, mastoparan and related insect venoms, modified guanine nucleotides, suramin analogues and amphiphilic cations. Because compounds that act on G proteins may be endowed with new forms of selectivity, we propose that G protein subunits may therefore be considered as potential drug targets.",
        "journal_info": "Cell Mol Life Sci",
        "pub_info": "1999 Feb;55(2):257-70. doi: 10.1007/s000180050288.",
        "authors": [
            {
                "name": "Höller C",
                "institute": [
                    "Institute of Pharmacology, University of Vienna, Austria."
                ]
            }
        ],
        "doi": "10.1007/s000180050288",
        "pmid": "10188585",
        "pmcid": "PMC11147085"
    }
    """
    def process(self, file_path: str) -> Union[str, List[str], List[Document]]:
        def parse_json_objects(json_str):
            decoder = json.JSONDecoder()
            offset = 0
            parsed_objects = []
            while offset < len(json_str):
                # 跳过空白字符（如换行、空格）
                while offset < len(json_str) and json_str[offset].isspace():
                    offset += 1
                if offset >= len(json_str):
                    break
                try:
                    # 解析单个JSON对象并更新偏移量
                    obj, offset = decoder.raw_decode(json_str, idx=offset)
                    parsed_objects.append(obj)
                except json.JSONDecodeError as e:
                    print(f"解析错误，位置 {offset}: {e}")
                    break  # 遇到错误时终止，可根据需要调整
            return parsed_objects
        try:
            with open(file_path, 'r', encoding='utf8') as f:
                content = f.read()
            data = parse_json_objects(content)
            documents = []
            if isinstance(data, list):
                for item in data:
                    # 摘要信息 -> text，标题，作者，机构 -> metadata
                    if 'abstract' in item:
                        doc = clean_text(item['abstract'])
                    else:
                        doc = ''
                    metadata = {
                        # 一次过滤的Tag
                        'title': item.get('title', ''),                # chunk上下文标题
                        'authors': item.get('authors', []),            # chunk上下文作者信息
                        'journal_info': item.get('journal_info', ''),  # chunk上下文杂志信息
                        'pub_info': item.get('pub_info', ''),          # chunk上下文出版信息
                        'doi': item.get('doi', ''),                    # chunk上下文DOI
                        'pmid': item.get('pmid', ''),                  # chunk上下文PMID
                        'pmcid': item.get('pmcid', ''),                # chunk上下文PMCID
                        'institutes': [author.get('institute', []) for author in item.get('authors', [])],  # chunk上下文作者机构
                        "author_names": [author.get('name', '') for author in item.get('authors', [])],     # chunk上下文作者姓名
                        
                        # 二次过滤的tag
                        'keywords': item.get('keywords', []), 
                    }
                    documents.append(Document(page_content=doc, metadata=metadata))
        except Exception as e:
            raise ValueError(f"JsonProcessor error: {e}")
        return documents

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