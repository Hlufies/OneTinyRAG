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
from .Tools import run, analyze_workflow, print_workflow_results,workflow, extract_json_blocks, format_template
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def ApiQuery(query)->dict:
    client = OpenAI(api_key="sk-94a5591a82a0409889c4d00f1c09edc0", base_url="https://api.deepseek.com")
    llm_model = "deepseek-chat"

    messages=[{"role": "system", "content": ""},{"role": "user", "content": f"{query}"},]
    print(query)
    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        stream=False
    )
    result = response.choices[0].message.content
    
    result = extract_json_blocks(result)
    return result
def save_dict(path, results):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
def Query(user_query):
    with open(os.path.join(current_dir,"query_template_v1.json"), "r") as f:
        query_template = json.load(f)
    query_template["Initialization"] += user_query
    user_dict = format_template(query_template)[1:-1]
    task_dict = ApiQuery(user_dict)
    save_dict("query_before.json", task_dict)
    task_graph = analyze_workflow(task_dict)
    print_workflow_results(task_graph)
    try:
        final_dict = asyncio.run(run(user_query,  task_dict, task_graph, {}))
    except Exception as e:
        print("Query error: ", e)
        return user_query, task_dict, task_dict
    save_dict("query_after.json", final_dict)
    return user_query, task_dict, final_dict





