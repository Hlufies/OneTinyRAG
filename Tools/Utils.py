#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-08 23:33:50
"""

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
import sys
import os
from collections import defaultdict, deque
import asyncio
from collections import deque
from typing import Dict, List, Any, Callable, Coroutine, Optional
import asyncio
import json
from typing import Dict, List, Any, Callable, Coroutine
from openai import OpenAI
import ollama
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def save_dict(path, results):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def format_template(data: dict, indent: int = 0, compact: bool = True) -> str:
    """
    将字典格式化为紧凑的文本形式
    
    参数:
        data: 要格式化的字典数据
        indent: 当前缩进级别
        compact: 是否使用紧凑格式（减少换行）
        
    返回:
        格式化后的文本字符串
    """
    # 缩进字符串
    indent_str = " " * indent
    
    if isinstance(data, dict):
        # 处理空字典
        if not data:
            return "{}"
        
        # 处理嵌套字典
        parts = []
        for key, value in data.items():
            # 处理键
            formatted_key = f'"{key}"' if isinstance(key, str) else str(key)
            
            # 处理值
            if isinstance(value, (dict, list)):
                # 对于嵌套结构，增加缩进
                formatted_value = format_template(value, indent + 4, compact)
                
                # 如果值较短，使用紧凑格式
                if compact and len(str(value)) < 60:
                    parts.append(f"{indent_str}{formatted_key}: {formatted_value.strip()}")
                else:
                    parts.append(f"{indent_str}{formatted_key}: {formatted_value}")
            else:
                # 基本数据类型
                formatted_value = f'"{value}"' if isinstance(value, str) else str(value)
                parts.append(f"{indent_str}{formatted_key}: {formatted_value}")
        
        # 添加逗号分隔符
        content = ",\n".join(parts)
        return f"{{\n{content}\n{' ' * indent}}}"
    
    elif isinstance(data, list):
        # 处理空列表
        if not data:
            return "[]" 
        # 处理嵌套列表
        parts = []
        for item in data:
            formatted_item = format_template(item, indent + 4, compact)
            # 如果值较短，使用紧凑格式
            if compact and len(str(item)) < 40:
                parts.append(formatted_item.strip())
            else:
                parts.append(indent_str + formatted_item.strip())
        # 添加逗号分隔符
        content = ", ".join(parts)
        return f"[{content}]"
    elif isinstance(data, str):
        # 处理字符串值
        return f'"{data}"'
    elif isinstance(data, (int, float)):
        # 处理数值
        return str(data)
    elif data is None:
        return "null"
    else:
        # 处理其他类型
        return str(data)

def extract_json_blocks(text):
    start = 0
    end = len(text) - 1

    while start < end:
        if text[start] != '{':
            start +=1
        if text[end] != '}':
            end -= 1
        if text[start] == '{' and text[end] == '}':
            break
    if start >= end:
        return {}
    try:
        return json.loads(text[start:end+1])
    except Exception as e:
        print("extract json error: ", e)
        return {}


def merge_branch(dict_a: dict, dict_b: dict) -> dict:
    """
    递归合并两个字典，支持嵌套结构，处理特殊键和类型转换
    :param dict_a: 主字典（被修改）
    :param dict_b: 待合并字典
    :return: 合并后的字典（即修改后的 dict_a）
    """
    skip_keys = {'required_steps', 'complexity'}
    
    for key, val_b in dict_b.items():
        # 跳过特定键
        if key in skip_keys:
            continue
        
        # 处理 confidence 键（特殊逻辑）
        if key == 'confidence':
            if isinstance(val_b, (int, float)) and isinstance(dict_a.get(key), (int, float)):
                dict_a[key] = (dict_a[key] + val_b) / 2
            continue
        
        # 递归处理嵌套字典
        if key in dict_a and isinstance(dict_a[key], dict) and isinstance(val_b, dict):
            merge_branch(dict_a[key], val_b)
            continue
        
        # 统一转换为列表处理
        val_a = dict_a.get(key)
        if isinstance(val_a, str):
            val_a = [val_a]
        if isinstance(val_b, str):
            val_b = [val_b]
        
        # 合并列表并去重
        if isinstance(val_a, list) and isinstance(val_b, list):
            # 使用集合去重（注意：仅适用于不可变元素）
            combined = list(set(val_a) | set(val_b))
            dict_a[key] = combined
        else:
            # 非列表类型直接覆盖
            dict_a[key] = val_b
    
    return dict_a
    
# def merge_branch(dict_a, dict_b) -> Any:
#     keys_a = set(dict_a.keys())
#     keys_b = dict_b.keys()
    
#     for kb in keys_b:
#         if kb == 'required_steps':
#             continue
#         if kb in 'complexity':
#             continue
#         if kb in keys_a:
#             ctx_a = dict_a[kb]
#             ctx_b = dict_b[kb]

#             if kb == 'confidence':
#                 if isinstance(ctx_a, float) and isinstance(ctx_b, float):
#                     dict_a[kb] = (ctx_a + ctx_b) / 2
#                     continue
#                 else:
#                     try:
#                         ctx_a = float(ctx_a)
#                         ctx_b = float(ctx_b)
#                         dict_a[kb] = (ctx_a + ctx_b) / 2
#                     except Exception as e:
#                         continue
#                     continue
#             if isinstance(ctx_a, str):
#                 ctx_a = [ctx_a]
#             if isinstance(ctx_b, str):
#                 ctx_a = [ctx_b]
#             # 同时lists 暂时只考虑这种情况
#             if isinstance(ctx_a, list) and isinstance(ctx_b, list):
#                 if ctx_b not in ctx_a:
#                     ctx_a += ctx_b
#                 dict_a[kb] = ctx_a

#             # 同时dict 暂时只考虑这种情况
#             if isinstance(ctx_a, dict) and isinstance(ctx_b, dict):
#                 for k, v in ctx_b.items():
#                     ctx_a[k] = v
#                 dict_a[kb] = ctx_a
#         else:
#             dict_a[kb] = dict_b[kb]                
#     return dict_a


def ApiQuery(query, llm_model="deepseek-chat", api_key="sk-xxx", base_url="https://api.deepseek.com")->dict:
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages=[{"role": "system", "content": ""},{"role": "user", "content": f"{query}"},]
    
    response = client.chat.completions.create(
        model=llm_model,
        messages=messages,
        stream=False
    )
    result = response.choices[0].message.content
    result = extract_json_blocks(result)
    return result

def OllamaDeepseekQuery(query, model_name="deepseek-r1:70b")->dict:
    response = ollama.generate(
        model=model_name,
        prompt=query
    )
    result = response["response"]
    result = extract_json_blocks(result)
    return result





    
  
  



