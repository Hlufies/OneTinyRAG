#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    @Copyright: © 2025 Junqiang Huang. 
    @Version: OneRAG v3
    @Author: Junqiang Huang
    @Time: 2025-06-08 23:33:50
    

"""

import os
import sys
import json
from Indexer.Indexer import Indexer
from Retriever.Retriever import Retriever
from Generator.Generator import Generator
from Tools.Query import Query
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)



os.environ["TOKENIZERS_PARALLELISM"] = "false"
with open('Config/config7.json', 'r') as js:
    config = json.load(js)
query="西红柿炒蛋怎么做的？"
user_query, task_dict, final_dict = Query(user_query=query, config=config)
exit(0)
indexer = Indexer(config)
textIndex, txtChunks = indexer.index("Dataset")
retriever = Retriever(indexer.DocEmbedder.embedder, textIndex, config)
retrievalChunks = retriever.retrieval(query, txtChunks, top_k=3)
generator = Generator(config)
result = generator.generate(query, retrievalChunks)
# print(result)
