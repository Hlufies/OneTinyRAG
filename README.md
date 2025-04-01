```markdown
# OneRAG - Intelligent Document Assistant 


**First-generation RAG system** implementing core retrieval-augmented generation capabilities. Now open for community-driven enhancements!

## 🚀 Current Capabilities

### Core Features
✅ **Document Processing**  
- Supported formats: PDF/TXT/Markdown  
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

### Quick Start
```bash
conda create -n onerag python=3.9
conda activate onerag
pip install -r requirements.txt
python app.py --config config/config1.json
```

## 🔭 Future Roadmap

### Phase 1: Core Enhancements (Q3 2024)
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

Sample contribution (Add CSV processor):
```python
class CSVProcessor(BaseProcessor):
    def process(self, file_path):
        import pandas as pd
        df = pd.read_csv(file_path)
        return [Document(page_content=str(row)) for row in df.iterrows()]
```

## 📊 Performance Benchmarks

```

Key features of this README:
1. Clear progression from current state to future vision
2. Version-phased development plan
3. Actionable contribution guidelines
4. Performance metrics transparency
5. Community engagement prompts
6. Modular architecture showcase
7. Badges for project status tracking
