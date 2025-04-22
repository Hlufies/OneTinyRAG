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
with open('config/config6.json', 'r') as js:
    config = json.load(js)
query="下面报告中涉及了哪几个行业的案例以及总结各自面临的挑战？"
indexer = Indexer(config)
textIndex, textChunks, imgIndex, imgChunks = indexer.index("/newdata/HJQ/Cultural_Tourism_Program/RAGDemo/OneRAG/Tutorial/NativeRag/dataset")
retriever = Retriever(indexer.DocEmbedder.embedder, indexer.ImgEmbedder.embedder, textIndex, imgIndex, config)
retrievalChunks_txt = retriever.docRetriever.retrieval_txt(query, textChunks, 3)
# unprocessed
# retrievalChunks_img = retriever.imgRetriever.retrieval_img(query, imgChunks, 3)

# generator = Generator(config)
# result = generator.Generator.generate(query, retrievalChunks)
# print(result)
