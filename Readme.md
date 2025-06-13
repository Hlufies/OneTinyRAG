# OneRAG - Intelligent Document Assistant 

<p align="center">
  <a href="https://github.com/Hlufies/OneRAG/blob/main/Readme.md">English</a> | 
  <a href="https://github.com/Hlufies/OneRAG/blob/main/Readme_zh.md">简体中文</a>
</p>


Implementing core retrieval-augmented generation capabilities. Now open for community-driven enhancements!


## Todo
- [2025.4.23] 🔥 Mutil-Modal process
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

### Run
```bash
bash app.sh
```

## Project Structure

```
/OneRAG/
├── app.sh                 # Main bash file
├── app.py                 # Main entry point
├── Readme.md              # English documentation
├── Readme_zh.md           # Chinese documentation
├── Indexer/               # Indexing module
│   ├── Indexer.py         # Index management core
│   ├── Embedder.py        # Embedding model abstraction
│   ├── DataProcessor.py   # Document processors
│   └── Chunker.py         # Text chunking strategies
├── Generator/             # Generation module
│   └── Generator.py       
├── Retriever/             # Retrieval module
│   └── Retriever.py
├── Tools/                 # Tools module
│   ├── Query.py           # Rewritte Query
│   ├── Search.py          # Search
│   ├── Utils.py           # Utils
│   └── Workflow.py        # Workflow
└── Config/                # Configuration directory
│   └── config.json
├── Dataset/               # Retrieval module
│   └── text
├── Agent/              
├── Critic/                
├── Tutorial/                
│
└── Tutorial/              # Other Tutorials
  ├── Ollama_zh.md         
  └── Ollama.md
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

## 📊 Performance Benchmarks


