

<p align="center">
  <a href="https://github.com/Hlufies/OneRAG/blob/main/Tutorial/NativeRag/Readme.md">English</a> | 
  <a href="https://github.com/Hlufies/OneRAG/blob/main/Tutorial/NativeRag/Readme_zh.md">简体中文</a>
</p>

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
- [2025.4.15]  MetaData Chunk 🔄
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
• [ ] Deepseek local deployment (Ollama)  
• [ ] Support 10+ file formats  
• [ ] Web UI visualization  
• [ ] Hybrid search strategies  
• [ ] Performance benchmarking  
