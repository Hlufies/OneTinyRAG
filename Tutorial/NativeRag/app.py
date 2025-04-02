import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
import json
from Indexer.Indexer import Indexer
from Retriever.Retriever import Retriever
from Generator.Generator import Generator

os.environ["TOKENIZERS_PARALLELISM"] = "false"
with open('config/config5.json', 'r') as js:
    config = json.load(js)
query="下面报告中涉及了哪几个行业的案例以及总结各自面临的挑战？"
indexer = Indexer(config)
index, chunks = indexer.index("dataset/docs.pdf")
retriever = Retriever(indexer.Embedder.embedder, index, config)
retrievalChunks = retriever.Retriever.retrieval(query, chunks, 3)
generator = Generator(config)
result = generator.Generator.generate(query, retrievalChunks)
print(result)
