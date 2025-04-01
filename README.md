
# OneRAG - Intelligent Document Assistant 

Implementing core retrieval-augmented generation capabilities. Now open for community-driven enhancements!

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


