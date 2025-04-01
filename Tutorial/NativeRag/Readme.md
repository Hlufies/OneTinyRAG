# Tutorial-NativeRAG

A RAG Tutorial based on LangChain+Deepseek+Faiss implementation.

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
