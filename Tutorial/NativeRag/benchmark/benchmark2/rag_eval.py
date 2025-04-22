from tqdm.auto import tqdm
import pandas as pd
from typing import Optional, List, Tuple
import json
import os
import datasets
from abc import ABC, abstractmethod
from typing import List
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from sentence_transformers import SentenceTransformer
import faiss 
import numpy as np 
from openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import pandas as pd
from langchain_core.vectorstores import VectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document as LangchainDocument
# ====================== config ================================
# export OLLAMA_MODELS="/newdata/HTY/BIO_LLM/DS_70b"
# export HF_ENDPOINT=https://hf-mirror.com
# 设置OLLAMA_MODELS环境变量
pd.set_option("display.max_colwidth", None)
os.environ['OLLAMA_MODELS'] = "/newdata/HTY/BIO_LLM/DS_70b/models"
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
pd.set_option("display.max_colwidth", None)
custom_path = "/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/eval_dataset"  # 自定义路径
client = OpenAI(api_key="sk-7861e2c852df45278daaa80e9f6057e6", base_url="https://api.deepseek.com")
# ====================== config ================================

# 读
# 
eval_dataset = datasets.load_dataset("m-ric/huggingface_doc_qa_eval", split="train",cache_dir=custom_path)