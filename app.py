import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import json
from Indexer.Indexer import Indexer
from Retriever.Retriever import Retriever
from Generator.Generator import Generator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_CACHE"]="/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/models"
with open('config/config7.json', 'r') as js:
    config = json.load(js)
query="以下有哪些靶蛋白"
indexer = Indexer(config)
textIndex, txtChunks = indexer.index("dataset")
retriever = Retriever(indexer.DocEmbedder.embedder, textIndex, config)
retrievalChunks = retriever.retrieval(query, txtChunks, top_k=3)
generator = Generator(config)
result = generator.generate(query, retrievalChunks)
# print(result)
