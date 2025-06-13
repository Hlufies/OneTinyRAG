#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-08 23:33:50
"""

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import Ollama
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from collections import deque, defaultdict
from abc import ABC, abstractmethod
from typing import List
from openai import OpenAI
import faiss 
import numpy as np 
import json
import re
import asyncio
import sys
import os
from .Workflow import run, analyze_workflow, print_workflow_results,workflow, task_func_default
from .Utils import format_template, save_dict, extract_json_blocks
from .Utils import ApiQuery, OllamaDeepseekQuery
from Mappers.Mappers import QUERY_MODEL_MAPPING, TASK_FUNC_MAPPING, TASK_LLM_MAPPING
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)



# 查询优化
def Query(user_query, config=None):
    assert config is not None
    config = config['query']
    with open(os.path.join(current_dir,config["template_query"]), "r") as f:
        query_template = json.load(f)
    query_template["Initialization"] += user_query
    user_dict = format_template(query_template)[1:-1]
    task_dict = QUERY_MODEL_MAPPING[config["query_model"]](user_dict)
    # 分析task流
    task_graph = analyze_workflow(task_dict)
    task_funcs = {k: TASK_FUNC_MAPPING[config["task_func"]] for k, _ in task_dict.items()}
    task_llms  = {k: TASK_LLM_MAPPING[config["task_llm"]] for k, _ in task_dict.items()}
    # 解析分析结果
    print_workflow_results(task_graph)
    try:
        final_dict = asyncio.run(
            run(
                user_query=user_query,   # 用户查询 
                task_dict=task_dict,     # 任务字典
                task_graph=task_graph,   # 任务图
                task_funcs=task_funcs,   # 任务方法
                task_llms=task_llms,      # 任务模型
                template=config["template_workflow"]
            )
        )
        
    except Exception as e:
        print("Query error: ", e)
        return user_query, task_dict, task_dict
    save_dict("query_before.json", task_dict)
    save_dict("query_after.json", final_dict)
    return user_query, task_dict, final_dict





