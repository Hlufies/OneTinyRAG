
<p align="center">
  <img src="poster.png" alt="封面图" />
</p>

# OneRAG - Intelligent Document Assistant 

<p align="center">
  <a href="https://github.com/Hlufies/OneRAG/blob/main/Tutorial/NativeRag/Readme.md">English</a> | 
  <a href="https://github.com/Hlufies/OneRAG/blob/main/Tutorial/NativeRag/Readme_zh.md">简体中文</a>
</p>


Implementing core retrieval-augmented generation capabilities. Now open for community-driven enhancements!

# Tutorial-NativeRAG

A RAG Tutorial based on LangChain+Deepseek+Faiss implementation.

## Todo
- [2025.4.20]  RL RAG 🔄
- [2025.4.20]  CRAG 🔄
- [2025.4.20]  Self RAG 🔄
- [2025.4.20]  Multi-Modal RAG 🔄
- [2025.4.20]  Adaptive Retrieval 🔄
- [2025.4.20]  HyDE 🔄
- [2025.4.20]  Query Transformations 🔄
- [2025.4.15]  Relevant Segment Extraction 🔄
- [2025.4.15]  Contextual Compression 🔄
- [2025.4.15]  Fusion Retrieval 🔄
- [2025.4.15]  Contextual Chunk Header 🔄
- [2025.4.23] 🔥 Mutil-Modal process
- [2025.4.23] 🔥 BLIP2
- [2025.4.23] 🔥 MetaData Chunk
- [2025.4.10] 🔥 Multi-format File Auto-processing ✅
- [2025.4.10] 🔥 Ollama + Deepseek Local Deployment ✅
- [2025.4.10] 🔥 SemanticChunker Extension (Spacy & NLTK) ✅

## Quick Start
### Install Dependencies
```bash
conda activate DeepseekRag
pip install -r requirements.txt
```

### Run Demo
```bash
python app.py
```

## Project Structure

```
/NativeRag/
├── app.py                 # Main entry point
├── Readme.md              # English documentation
├── Readme_zh.md           # Chinese documentation
├── Indexer/               # Indexing module
│   ├── __init__.py
│   ├── Indexer.py         # Index management core
│   ├── Embedder.py        # Embedding model abstraction
│   ├── DataProcessor.py   # Document processors
│   └── Chunker.py         # Text chunking strategies
├── Generator/             # Generation module
│   └── Generator.py       
├── Retriever/             # Retrieval module
│   └── Retriever.py
└── config/                # Configuration directory
    └── config1.json
```

## Core Components

```python
indexer = Indexer(config)
retriever = Retriever(indexer.embedder.embedder, index, config)
generator = Generator(indexer.embedder, config)
```

## Configuration
Configure settings via `config/config1.json`:
• Embedding model parameters
• Chunking size/overlap
• Similarity threshold
• LLM API endpoints

## Extension Guide
1. Register in `Indexer.py`
2. Add document processors in `DataProcessor.py`
3. Implement new text splitters in `Chunker.py`
4. Extend embedding models in `Embedder.py`
5. Register in `Retriever.py`
6. Develop new retrievers in `Retriever.py`
7. Register in `Generator.py`
8. Implement new generators in `Generator.py`





## TODO

## 🚀 Current Capabilities

### Core Features
✅ **Document Processing**  
- Supported formats: PDF/TXT 
- Chunking strategies: Fixed-size sliding window (512 tokens)  
- Metadata extraction: File name, timestamp  

✅ **Vector Engine**  
- Embedding models: text-embedding-3-small (API)  
- Vector storage: FAISS with HNSW indexing  
- Similarity search: Cosine similarity  

✅ **Generation Pipeline**  
- LLM integration: Deepseek API  
- Contextual fusion: Dynamic prompt templating  
- Response validation: Basic hallucination checks  


## 🔭 Future Roadmap
### Phase 1: Core Enhancements
**Document Processing**  
- [ ] Multi-modal support (DOCX/PPTX/HTML)  
- [ ] Layout-aware PDF parsing  
- [ ] Dynamic chunk sizing based on content  

**Vector Engine**  
- [ ] Hybrid search (keyword + vector)  
- [ ] Local embedding models (bge-small-en)  
- [ ] Version-controlled indexes  

**Generation**  
- [ ] Local LLM deployment (Deepseek-chat 7B)  
- [ ] Response citation generation  
- [ ] Multi-turn conversation support  

### Phase 2: Enterprise Features (Q4 2024)
- [ ] Role-based access control  
- [ ] Audit logging subsystem  
- [ ] GPU-accelerated indexing  
- [ ] Cluster deployment mode  

### Phase 3: Intelligent Features (2025)
- [ ] Cross-document reasoning  
- [ ] Automated fact verification  
- [ ] Dynamic knowledge graph  
- [ ] Anomaly detection alerts  

## 🛠️ How to Contribute

### Extension Points
| Module          | Extension Methods                      |
|-----------------|-----------------------------------------|
| DocumentLoader  | Implement `BaseProcessor` in DataProcessor.py |
| Chunker         | Add new class in Chunker.py             | 
| Embedder        | Extend `EmbeddingWrapper` in Embedder.py|
| Retriever       | Develop new strategies in Retriever.py  |
| Generator       | Create response formatters in Generator.py |



## 📊 Performance Benchmarks


